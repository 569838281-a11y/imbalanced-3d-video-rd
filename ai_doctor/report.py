"""L3 ReportBuilder: rule-based Findings / Impression / Advice (evidence-bound)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_doctor.case_result import DISCLAIMER_ZH


def build_report(
    *,
    has_rd: bool,
    rd_prob: float,
    risk_tier: str,
    macula_detached: Optional[bool],
    stage2_ran: bool,
    evidence: List[Dict[str, Any]],
    disease_label_zh: str,
) -> Dict[str, Any]:
    frames = [e["frame_idx"] for e in evidence]
    frame_phrase = (
        "、".join(str(f) for f in frames)
        if frames
        else "若干关键帧"
    )
    cam_texts = [e["text"] for e in evidence[:3]]

    # Findings = objective imaging language only
    findings_parts = [
        f"眼部超声体积已按训练分布重采样；依据引擎关注帧：第 {frame_phrase} 帧。",
    ]
    if evidence:
        findings_parts.append(cam_texts[0])
        if len(cam_texts) > 1:
            findings_parts.append("其它依据帧亦见模型/质量分联合提示的可疑高亮区。")
    if has_rd:
        findings_parts.append(
            "一阶段判别提示存在与视网膜脱离相符的影像模式（概率见 Impression）。"
        )
    else:
        findings_parts.append(
            "一阶段判别未提示明确视网膜脱离影像模式；阴性不能完全排除早期/局限脱离。"
        )
    if stage2_ran and macula_detached is True:
        findings_parts.append("二阶段级联提示黄斑受累倾向（影像分类结果，非裂孔定位）。")
    elif stage2_ran and macula_detached is False:
        findings_parts.append("二阶段级联提示黄斑完整性相对保留。")

    findings = " ".join(findings_parts)

    # Impression = tendency + uncertainty, not etiology
    if not has_rd:
        impression = (
            f"倾向印象：{disease_label_zh}（RD 概率 {rd_prob:.1%}，风险分层 {risk_tier}）。"
            "当前证据不支持作出视网膜脱离的肯定诊断；若症状持续或出现闪光/遮挡，仍需眼科评估。"
        )
    elif macula_detached is True:
        impression = (
            f"倾向印象：{disease_label_zh}（RD 概率 {rd_prob:.1%}，风险分层 {risk_tier}）。"
            "级联结果提示黄斑可能已受累；请尽快专科面诊，勿将本输出视为手术指征本身。"
        )
    else:
        impression = (
            f"倾向印象：{disease_label_zh}（RD 概率 {rd_prob:.1%}，风险分层 {risk_tier}）。"
            "级联结果提示黄斑可能尚未脱离，但仍属视网膜脱离相关高风险范畴，需专科确认。"
        )

    advice = _advice_templates(risk_tier, has_rd, macula_detached)

    return {
        "findings": findings,
        "impression": impression,
        "advice": advice,
        "disclaimer": DISCLAIMER_ZH,
    }


def _advice_templates(
    risk_tier: str,
    has_rd: bool,
    macula_detached: Optional[bool],
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if risk_tier == "high" or (has_rd and macula_detached is True):
        items.append(
            {
                "level": "urgent",
                "template_id": "rd_macula_urgent",
                "text": "建议尽快（当日/急诊窗口）联系眼科或视网膜专科评估，避免自行延误。",
            }
        )
    elif has_rd:
        items.append(
            {
                "level": "soon",
                "template_id": "rd_prompt_visit",
                "text": "建议尽快安排眼科专科就诊，携带本超声资料供医师复核。",
            }
        )
    else:
        items.append(
            {
                "level": "routine",
                "template_id": "non_rd_observe",
                "text": "可先按常规眼科随访；若出现突发视野遮挡、闪光感或视力骤降，请立即就医。",
            }
        )

    items.append(
        {
            "level": "info",
            "template_id": "disclaimer_repeat",
            "text": "AI 输出仅供辅助参考；最终诊断与治疗方案以执业医师意见为准。",
        }
    )
    return items
