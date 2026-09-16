"""Cascade AI doctor: Stage-1 RD → Stage-2 macula → evidence + CaseResult."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision.io as io
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import erdes  # noqa: E402

sys.modules["src"] = erdes

from ai_doctor.case_result import DISCLAIMER_ZH, build_case_result, risk_tier  # noqa: E402
from ai_doctor.evidence import EvidenceEngine  # noqa: E402
from ai_doctor.report import build_report  # noqa: E402
from erdes.data.components.unsharp_masking import apply_unsharp_masking  # noqa: E402
from erdes.data.components.utils import resize  # noqa: E402
from erdes.models.components.factory import build_3d_architecture  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RD_CANDIDATES = [
    _ROOT / "weights" / "um_decoupled_best.ckpt",
    Path("D:/ERDES/checkpoints/um_decoupled_50ep_local/um_decoupled_best.ckpt"),
]
_DEFAULT_MACULA_CANDIDATES = [
    _ROOT / "weights" / "macula_unet3d_best.ckpt",
    Path("D:/ERDES/checkpoints/macula_unet3d_50ep_local/macula_unet3d_best.ckpt"),
]


def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.is_file():
            return p
    return paths[0]


DEFAULT_RD_CKPT = _first_existing(_DEFAULT_RD_CANDIDATES)
DEFAULT_MACULA_CKPT = _first_existing(_DEFAULT_MACULA_CANDIDATES)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _adapt_state_dict(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    adapted = {}
    for k, v in state_dict.items():
        if k.startswith("logit_adjuster."):
            continue
        key = k.replace("net.", "", 1) if k.startswith("net.") else k
        adapted[key] = v
    model_keys = set(model.state_dict().keys())
    if model_keys == set(adapted.keys()):
        return adapted
    if any(k.startswith("model.") for k in model_keys) and not any(
        k.startswith("model.") for k in adapted
    ):
        return {f"model.{k}": v for k, v in adapted.items()}
    if any(k.startswith("enc.") for k in model_keys) and not any(
        k.startswith("enc.") for k in adapted
    ):
        new_adapted = {}
        for k, v in adapted.items():
            if k.startswith("fc1.") or k.startswith("fc2."):
                new_adapted[f"cls.{k}"] = v
            else:
                new_adapted[f"enc.{k}"] = v
        return new_adapted
    return adapted


def _load_net(
    model_name: str,
    ckpt_path: Path,
    device: str,
    pooling: str = "topk",
    topk_ratio: float = 0.5,
) -> torch.nn.Module:
    net = build_3d_architecture(
        model_name, num_classes=1, pooling=pooling, topk_ratio=topk_ratio
    )
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = _adapt_state_dict(ckpt["state_dict"], net)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys (%s): %s", model_name, missing[:8])
    if unexpected:
        log.warning("Unexpected keys (%s): %s", model_name, unexpected[:8])
    net = net.to(device).eval()
    return net


def disease_from_flags(has_rd: bool, macula_detached: Optional[bool]) -> Dict[str, str]:
    if not has_rd:
        return {
            "code": "non_rd",
            "label_zh": "未见明显视网膜脱离",
            "label_en": "No clear retinal detachment",
            "summary_zh": "一阶段未检出视网膜脱离征象。当前无需进入黄斑分期；建议结合临床与其它检查综合判断。",
        }
    if macula_detached is True:
        return {
            "code": "rd_macula_detached",
            "label_zh": "视网膜脱离（黄斑已脱离）",
            "label_en": "Retinal detachment with macular detachment",
            "summary_zh": "一阶段检出视网膜脱离，二阶段提示黄斑已脱离。临床优先级通常更高，请尽快专科评估。",
        }
    return {
        "code": "rd_macula_intact",
        "label_zh": "视网膜脱离（黄斑完整）",
        "label_en": "Retinal detachment with macula intact",
        "summary_zh": "一阶段检出视网膜脱离，二阶段提示黄斑尚未脱离。仍需专科随访与治疗决策。",
    }


class AIDoctor:
    """Two-stage ocular ultrasound AI doctor with L2 evidence + L3 report."""

    def __init__(
        self,
        rd_ckpt: Union[str, Path] = DEFAULT_RD_CKPT,
        macula_ckpt: Union[str, Path] = DEFAULT_MACULA_CKPT,
        rd_model_name: str = "resnet3d",
        macula_model_name: str = "unet3d",
        device: Optional[str] = None,
        video_size: Tuple[int, int, int] = (96, 128, 128),
        rd_threshold: float = 0.5,
        macula_threshold: float = 0.5,
        apply_um_for_rd: bool = True,
        um_strength: float = 1.5,
        pooling: str = "topk",
        topk_ratio: float = 0.5,
        explain: bool = True,
        evidence_top_k: int = 3,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.video_size = video_size
        self.rd_threshold = rd_threshold
        self.macula_threshold = macula_threshold
        self.apply_um_for_rd = apply_um_for_rd
        self.um_strength = um_strength
        self.explain = explain
        self.resize_tf = resize(video_size)
        self.rd_model_name = rd_model_name
        self.macula_model_name = macula_model_name

        rd_ckpt = Path(rd_ckpt)
        macula_ckpt = Path(macula_ckpt)
        if not rd_ckpt.is_file():
            raise FileNotFoundError(f"RD checkpoint not found: {rd_ckpt}")
        if not macula_ckpt.is_file():
            raise FileNotFoundError(f"Macula checkpoint not found: {macula_ckpt}")

        log.info("Loading RD model (%s) from %s", rd_model_name, rd_ckpt)
        self.rd_model = _load_net(
            rd_model_name, rd_ckpt, self.device, pooling=pooling, topk_ratio=topk_ratio
        )
        log.info("Loading Macula model (%s) from %s", macula_model_name, macula_ckpt)
        self.macula_model = _load_net(
            macula_model_name, macula_ckpt, self.device, pooling=pooling, topk_ratio=topk_ratio
        )
        self.evidence_engine = EvidenceEngine(
            self.rd_model, self.device, top_k=evidence_top_k
        ) if explain else None
        log.info("AI Doctor ready on %s (explain=%s)", self.device, explain)

    def _normalize_volume(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 4:
            raise ValueError(f"Expected [C,D,H,W], got {tuple(video.shape)}")
        if video.shape[0] == 3:
            video = video.mean(dim=0, keepdim=True)
        elif video.shape[0] != 1:
            video = video[:1]
        video = self.resize_tf(video.float())
        if float(video.max()) > 1.5:
            video = video / 255.0
        return video.unsqueeze(0)

    def preprocess_path(self, path: Union[str, Path]) -> torch.Tensor:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        ext = path.suffix.lower()
        if ext in VIDEO_EXTS:
            video, _, _ = io.read_video(str(path), pts_unit="sec")
            video = video.float().permute(3, 0, 1, 2)
            return self._normalize_volume(video)
        if ext in IMAGE_EXTS:
            img = Image.open(path).convert("L")
            frame = to_tensor(img)
            t = self.video_size[0]
            video = frame.unsqueeze(1).repeat(1, t, 1, 1)
            return self._normalize_volume(video * 255.0)
        raise ValueError(f"Unsupported file type: {ext}")

    def preprocess_image_paths(self, paths: List[Path]) -> torch.Tensor:
        frames = []
        for p in sorted(paths, key=lambda x: x.name.lower()):
            img = Image.open(p).convert("L")
            frames.append(to_tensor(img))
        if not frames:
            raise ValueError("No images provided")
        video = torch.stack(frames, dim=1)
        return self._normalize_volume(video * 255.0)

    def diagnose_tensor(
        self,
        video: torch.Tensor,
        *,
        source: str = "upload",
        input_type: str = "video",
    ) -> Dict[str, Any]:
        """Run cascade + optional evidence/report → flat UI fields + case_result."""
        video = video.to(self.device)
        n_frames = int(video.shape[2])

        rd_in = video
        if self.apply_um_for_rd:
            um = apply_unsharp_masking(video.squeeze(0), strength=self.um_strength)
            rd_in = um.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            rd_prob = float(torch.sigmoid(self.rd_model(rd_in)).item())
            has_rd = rd_prob >= self.rd_threshold

            macula_prob: Optional[float] = None
            macula_detached: Optional[bool] = None
            stage2_ran = False
            if has_rd:
                stage2_ran = True
                macula_prob = float(torch.sigmoid(self.macula_model(video)).item())
                macula_intact = macula_prob >= self.macula_threshold
                macula_detached = not macula_intact

        disease = disease_from_flags(has_rd, macula_detached)
        tier = risk_tier(rd_prob, has_rd, macula_detached)

        evidence: List[Dict[str, Any]] = []
        evidence_meta: Dict[str, Any] = {}
        if self.explain and self.evidence_engine is not None:
            evidence, evidence_meta = self.evidence_engine.build(
                rd_in, has_rd=has_rd, include_overlays=True
            )

        report = build_report(
            has_rd=has_rd,
            rd_prob=rd_prob,
            risk_tier=tier,
            macula_detached=macula_detached,
            stage2_ran=stage2_ran,
            evidence=evidence,
            disease_label_zh=disease["label_zh"],
        )

        case = build_case_result(
            input_type=input_type,
            n_frames=n_frames,
            source=source,
            has_rd=has_rd,
            rd_prob=rd_prob,
            macula_detached=macula_detached,
            macula_intact_prob=macula_prob,
            stage2_ran=stage2_ran,
            evidence=[{k: v for k, v in e.items() if k != "overlay_png"} for e in evidence],
            report=report,
            disease=disease,
            rd_model_name="um_decoupled_50ep",
            macula_model_name="macula_unet3d_50ep",
        )

        # Compact overlays for UI (keep base64 here, not in case_result audit blob)
        overlays = [
            {
                "id": e["id"],
                "frame_idx": e["frame_idx"],
                "text": e["text"],
                "source": e["source"],
                "score": e["score"],
                "png": e.get("overlay_png"),
            }
            for e in evidence
            if e.get("overlay_png")
        ]

        return {
            "has_rd": has_rd,
            "rd_probability": round(rd_prob, 4),
            "rd_threshold": self.rd_threshold,
            "risk_tier": tier,
            "stage2_ran": stage2_ran,
            "macula_detached": macula_detached,
            "macula_intact_probability": (
                None if macula_prob is None else round(macula_prob, 4)
            ),
            "macula_threshold": self.macula_threshold if stage2_ran else None,
            "disease_code": disease["code"],
            "disease_label_zh": disease["label_zh"],
            "disease_label_en": disease["label_en"],
            "summary_zh": disease["summary_zh"],
            "disclaimer_zh": DISCLAIMER_ZH,
            "pipeline": [
                {"stage": 1, "name": "视网膜脱离筛查", "result": "阳性" if has_rd else "阴性"},
                {
                    "stage": 2,
                    "name": "黄斑状态",
                    "result": (
                        "未执行（一阶段阴性）"
                        if not stage2_ran
                        else ("黄斑已脱离" if macula_detached else "黄斑完整")
                    ),
                },
            ],
            "findings": report["findings"],
            "impression": report["impression"],
            "advice": report["advice"],
            "evidence_overlays": overlays,
            "evidence_meta": evidence_meta,
            "case_result": case,
            "source": source,
        }

    def diagnose_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        video = self.preprocess_path(path)
        return self.diagnose_tensor(
            video, source=Path(path).name, input_type="video"
        )

    def diagnose_images(self, paths: List[Path]) -> Dict[str, Any]:
        video = self.preprocess_image_paths(paths)
        return self.diagnose_tensor(
            video, source=f"{len(paths)} images", input_type="image"
        )


_doctor: Optional[AIDoctor] = None


def get_doctor() -> AIDoctor:
    global _doctor
    if _doctor is None:
        rd = os.environ.get("AI_DOCTOR_RD_CKPT", str(DEFAULT_RD_CKPT))
        macula = os.environ.get("AI_DOCTOR_MACULA_CKPT", str(DEFAULT_MACULA_CKPT))
        _doctor = AIDoctor(rd_ckpt=rd, macula_ckpt=macula)
    return _doctor
