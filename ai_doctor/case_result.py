"""CaseResult schema helpers (TECH_PIPELINE_FINAL §三)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


DISCLAIMER_ZH = (
    "本结果为科研辅助筛查输出，不能替代执业医师面诊与诊疗决策。"
    "依据区热力图仅为模型关注线索，不等于病灶解剖边界或病因定位。"
    "三阶段解剖亚型在当前数据下仅为原型演示，不可单独作为临床分型依据。"
)


def risk_tier(rd_prob: float, has_rd: bool, macula_detached: Optional[bool]) -> str:
    if not has_rd:
        return "low" if rd_prob < 0.3 else "mid"
    if macula_detached is True:
        return "high"
    if rd_prob >= 0.85:
        return "high"
    return "mid"


def build_case_result(
    *,
    input_type: str,
    n_frames: int,
    source: str,
    has_rd: bool,
    rd_prob: float,
    macula_detached: Optional[bool],
    macula_intact_prob: Optional[float],
    stage2_ran: bool,
    evidence: List[Dict[str, Any]],
    report: Dict[str, Any],
    disease: Dict[str, str],
    stage3_ran: bool = False,
    stage3_branch: Optional[str] = None,
    subtype: Optional[str] = None,
    subtype_prob: Optional[float] = None,
    rd_model_name: str = "um_decoupled_50ep",
    macula_model_name: str = "macula_unet3d_50ep",
    stage3_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not has_rd:
        macula_status = "skipped"
    elif macula_detached is True:
        macula_status = "detached"
    elif macula_detached is False:
        macula_status = "intact"
    else:
        macula_status = "unknown"

    tier = risk_tier(rd_prob, has_rd, macula_detached)
    models: Dict[str, Any] = {
        "rd": {"name": rd_model_name, "version": "local", "skipped": False},
        "macula": {
            "name": macula_model_name,
            "version": "local",
            "skipped": not stage2_ran,
        },
    }
    if stage3_model_name or stage3_ran:
        models["stage3"] = {
            "name": stage3_model_name or "stage3_swin_ssl",
            "version": "local",
            "branch": stage3_branch,
            "skipped": not stage3_ran,
        }

    return {
        "case_id": uuid.uuid4().hex[:12],
        "input": {"type": input_type, "n_frames": n_frames, "source": source},
        "models": models,
        "prediction": {
            "rd_label": "RD" if has_rd else "non_RD",
            "rd_prob": round(float(rd_prob), 4),
            "risk_tier": tier,
            "macula_status": macula_status,
            "macula_intact_prob": (
                None if macula_intact_prob is None else round(float(macula_intact_prob), 4)
            ),
            "subtype": subtype,
            "subtype_prob": (
                None if subtype_prob is None else round(float(subtype_prob), 4)
            ),
            "stage3_branch": stage3_branch,
            "disease_code": disease["code"],
            "disease_label_zh": disease["label_zh"],
            "disease_label_en": disease["label_en"],
            "multi_disease": [],
        },
        "evidence": evidence,
        "report": report,
        "dialogue_slots": {
            "duration_days": None,
            "photopsia": None,
            "field_defect": None,
            "trauma": None,
        },
    }
