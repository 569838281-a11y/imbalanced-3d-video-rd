"""L2 Evidence engine: C1 Grad-CAM + C2 frame quality + C4 rule filter."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from erdes.visualization.gradcam3d import GradCAM3D, overlay_cam_on_frame

log = logging.getLogger(__name__)


def _frame_quality_scores(video_chw: np.ndarray) -> np.ndarray:
    """
    Per-frame quality in [0,1]: contrast × edge energy × mid-tone preference.
    video_chw: [C, D, H, W] float in [0,1]
    """
    gray = video_chw[0] if video_chw.ndim == 4 else video_chw
    d, h, w = gray.shape
    scores = np.zeros(d, dtype=np.float32)
    for t in range(d):
        frame = np.clip(gray[t], 0.0, 1.0)
        u8 = (frame * 255).astype(np.uint8)
        contrast = float(frame.std())
        edges = cv2.Canny(u8, 40, 120)
        edge_density = float(edges.mean()) / 255.0
        # Prefer frames that are not nearly black/white washed-out.
        mean = float(frame.mean())
        mid = 1.0 - abs(mean - 0.45) * 1.6
        mid = max(0.05, mid)
        scores[t] = contrast * (0.35 + edge_density) * mid
    if scores.max() > 1e-8:
        scores = scores / (scores.max() + 1e-8)
    return scores


def _cam_box_norm(cam_hw: np.ndarray, thr_q: float = 0.85) -> Optional[List[float]]:
    """Normalized [x0,y0,x1,y1] from CAM threshold; None if empty."""
    h, w = cam_hw.shape
    thr = float(np.quantile(cam_hw, thr_q))
    mask = cam_hw >= max(thr, 0.35)
    ys, xs = np.where(mask)
    if len(xs) < 8:
        return None
    x0, x1 = float(xs.min()) / w, float(xs.max() + 1) / w
    y0, y1 = float(ys.min()) / h, float(ys.max() + 1) / h
    # Expand slightly
    pad = 0.02
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(1.0, x1 + pad), min(1.0, y1 + pad)
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def _filter_box(box: Optional[List[float]], min_area: float = 0.01) -> Optional[List[float]]:
    """C4: drop tiny / near-full-frame / edge-artifact boxes."""
    if box is None:
        return None
    x0, y0, x1, y1 = box
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area < min_area or area > 0.85:
        return None
    # Reject boxes glued to all four borders (UI chrome / full FOV).
    if x0 < 0.02 and y0 < 0.02 and x1 > 0.98 and y1 > 0.98:
        return None
    return box


def _png_b64(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class EvidenceEngine:
    """Build findings[] + overlay previews from RD net Grad-CAM and frame quality."""

    def __init__(self, rd_net: torch.nn.Module, device: str, top_k: int = 3) -> None:
        self.rd_net = rd_net
        self.device = device
        self.top_k = top_k
        self._gradcam: Optional[GradCAM3D] = None
        try:
            self._gradcam = GradCAM3D(rd_net)
        except Exception as exc:
            log.warning("GradCAM3D unavailable: %s", exc)

    def close(self) -> None:
        if self._gradcam is not None:
            self._gradcam.close()
            self._gradcam = None

    def build(
        self,
        video_1cthw: torch.Tensor,
        has_rd: bool,
        include_overlays: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        video_1cthw: [1,C,T,H,W] in [0,1] (prefer UM-aligned tensor for RD).
        Returns (evidence_list, meta).
        """
        vol = video_1cthw.detach()
        video_cdhw = vol.squeeze(0).detach().cpu()  # C,D,H,W
        gray_np = video_cdhw.numpy()
        quality = _frame_quality_scores(gray_np)
        d = gray_np.shape[1]

        cam: Optional[np.ndarray] = None
        cam_energy = np.zeros(d, dtype=np.float32)
        if self._gradcam is not None:
            label = 1 if has_rd else 0
            try:
                with torch.enable_grad():
                    cam = self._gradcam(vol.to(self.device), label=label)
                cam_energy = cam.reshape(d, -1).mean(axis=1).astype(np.float32)
                if cam_energy.max() > 1e-8:
                    cam_energy = cam_energy / (cam_energy.max() + 1e-8)
            except Exception as exc:
                log.warning("Grad-CAM failed: %s", exc)
                cam = None

        # Combined rank: CAM energy × quality (ONSD-style quality prior).
        if cam is not None:
            combined = 0.65 * cam_energy + 0.35 * quality
        else:
            combined = quality

        order = np.argsort(-combined)
        # Prefer frames that are also among top quality OR top CAM.
        top_q = set(np.argsort(-quality)[: max(8, self.top_k * 3)].tolist())
        top_c = set(np.argsort(-cam_energy)[: max(8, self.top_k * 3)].tolist()) if cam is not None else set(range(d))
        selected: List[int] = []
        for idx in order.tolist():
            if idx in top_q or idx in top_c:
                selected.append(int(idx))
            if len(selected) >= self.top_k:
                break
        if len(selected) < self.top_k:
            for idx in order.tolist():
                if idx not in selected:
                    selected.append(int(idx))
                if len(selected) >= self.top_k:
                    break

        evidence: List[Dict[str, Any]] = []
        for i, frame_idx in enumerate(selected):
            box = None
            source = "quality"
            text = f"成像质量较高的第 {frame_idx} 帧（客观选帧）"
            score = float(combined[frame_idx])
            if cam is not None:
                box = _filter_box(_cam_box_norm(cam[frame_idx]))
                source = "cam" if frame_idx in top_c else "quality"
                if has_rd:
                    text = (
                        f"第 {frame_idx} 帧：模型关注区提示可疑膜样/高回声区域"
                        "（事后归因，非解剖金标准）"
                    )
                else:
                    text = (
                        f"第 {frame_idx} 帧：相对关注区（阴性病例亦供对照，"
                        "不代表病变定位）"
                    )

            item: Dict[str, Any] = {
                "id": f"E{i + 1}",
                "frame_idx": frame_idx,
                "box_norm": box,
                "score": round(score, 4),
                "cam_energy": round(float(cam_energy[frame_idx]), 4),
                "quality_score": round(float(quality[frame_idx]), 4),
                "source": source,
                "text": text,
            }
            if include_overlays:
                frame = np.clip(gray_np[0, frame_idx], 0.0, 1.0)
                if cam is not None:
                    rgb = overlay_cam_on_frame(frame, cam[frame_idx], alpha=0.45)
                else:
                    rgb = (np.stack([frame, frame, frame], axis=-1) * 255).astype(np.uint8)
                # Draw box if present
                if box is not None:
                    h, w = rgb.shape[:2]
                    x0, y0, x1, y1 = box
                    p0 = (int(x0 * w), int(y0 * h))
                    p1 = (int(x1 * w), int(y1 * h))
                    cv2.rectangle(rgb, p0, p1, (255, 220, 80), 2)
                item["overlay_png"] = _png_b64(rgb)
            evidence.append(item)

        meta = {
            "cam_available": cam is not None,
            "n_frames": d,
            "topk_frames": selected,
        }
        return evidence, meta
