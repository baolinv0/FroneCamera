"""Readable reports over the shared payload; no independent inference in the renderer."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

from portrait_eval.report_model import challenge_matrix

if TYPE_CHECKING:
    from portrait_eval.reporting import ReportPayload


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _anchor(value: Any) -> str:
    # Prefix and hex encode to keep IDs exact, unique and valid for arbitrary user labels.
    return "e-" + str(value).encode().hex()


def _scene_links(ids: list[str]) -> str:
    return (
        ", ".join(f'<a href="#{_anchor(scene)}">{_text(scene)}</a>' for scene in ids)
        or "无 / unknown"
    )


def _claim(item: dict[str, Any], index: int, *, anchor: bool = True) -> str:
    identity = f' id="{_anchor(item.get("claim_id", f"legacy-{index}"))}"' if anchor else ""
    return f"""<article{identity}><b>{_text(item.get("statement", "缺少主张文本"))}</b>
<p>{_text(item.get("status", "legacy/unreviewed"))} · {_text(item.get("attribute", "unknown"))} · {_text(item.get("region", "unknown"))}</p>
<p>支持：{_scene_links(item.get("supporting_scene_ids", item.get("evidence", [])))}<br>
反例：{_scene_links(item.get("contradicting_scene_ids", []))}<br>
未知：{_scene_links(item.get("unknown_scene_ids", []))}；不可比：{_scene_links(item.get("incomparable_scene_ids", []))}</p>
<p>家族分母：{_text(item.get("evidence_counts", {}))}；独立支持家族：{_text(item.get("independent_family_count", "unknown"))}</p>
<small>范围：{_text(item.get("conditions", []))} · 规则标签 {_text(item.get("grade", "unknown"))}，未校准为可信概率。</small></article>"""


def render_evidence_html(payload: ReportPayload, status: str) -> str:
    claims = payload.all_claims or payload.findings
    approved = [
        item
        for item in payload.findings
        if item.get("status")
        not in {
            "DISPUTED",
            "INSUFFICIENT_EVIDENCE",
            "NOT_COMPARABLE",
            "INSUFFICIENT_SCENE_COVERAGE",
        }
    ]
    summary = (
        "".join(_claim(item, i, anchor=False) for i, item in enumerate(approved[:5]))
        or "<p>目前没有通过证据门槛的结论；待确认项保留在完整附录。</p>"
    )
    matrix = "".join(
        f"<tr><td>{_text(row['label'])}</td><td>{_scene_links(row['scene_ids'])}</td></tr>"
        for row in (payload.challenge_matrix or challenge_matrix(payload.scene_results))
    )
    sections = []
    for scene in payload.scene_results:
        key = scene["group_id"]
        photos = "".join(
            f'<figure><img src="{_text(asset["path"])}" loading="lazy" alt="{_text(key)} {_text(asset.get("device_name", asset.get("device_id", "asset")))}"><figcaption>{_text(asset.get("device_name", asset.get("device_id", "asset")))}</figcaption></figure>'
            for asset in payload.visual_assets
            if asset.get("scene_id") == key and asset.get("path")
        )
        verdicts = (
            "".join(
                _claim(item, i, anchor=False)
                for i, item in enumerate(scene.get("findings", []))
                if item.get("status") == "SUPPORTED_SCOPED"
            )
            or "<p>本场景暂无已支持结论。</p>"
        )
        crops = []
        for device, metrics in scene.get("metrics", {}).items():
            for face in metrics.get("faces", []):
                if face.get("native_crop_path"):
                    crops.append(
                        f'<figure><a href="{_text(face["native_crop_path"])}"><img src="{_text(face["native_crop_path"])}" loading="lazy" alt="{_text(device)} {_text(face["face_id"])} native crop"></a><figcaption>{_text(device)} / {_text(face["face_id"])} {_text(face.get("role", "unknown"))} · 原生像素裁剪，页面缩放显示</figcaption></figure>'
                    )
        sections.append(
            f'<section id="{_anchor(key)}"><h3>{_text(key)} · {_text(scene.get("audit", {}).get("challenge", {}).get("primary", "unknown"))}</h3>{verdicts}<div class="photos">{photos}</div><div class="photos">{"".join(crops)}</div><p>缺失来源：{_text(scene.get("missing_devices", []))}；家族：{_text(scene.get("scene_family") or "unknown")}</p><details><summary>完整观察、测量和分歧</summary><pre>{_text(json.dumps(scene, ensure_ascii=False, indent=2))}</pre></details></section>'
        )
    profiles = "".join(
        f"<article><h3>{_text(item.get('device_name', 'Device'))}</h3><p>{_text(item.get('summary', '证据不足'))}</p><p>适用范围：{_scene_links(item.get('scope', []))}；反例：{_scene_links(item.get('counterexamples', []))}</p><p>优先目标：{_text(item.get('priority', 'unknown'))}<br>收益与代价：{_text(item.get('tradeoffs', 'unknown'))}</p></article>"
        for item in payload.device_profiles
    )
    causes = "".join(
        f"<article><b>{_text(item.get('statement', item.get('observable_effect', '未知现象')))}</b><p>竞争假设：{_text(item.get('candidate_causes', []))}；替代解释：{_text(item.get('alternative_explanations', []))}</p><p>最小鉴别实验：{_text(item.get('minimal_experiment', '未定义'))}</p><small>证据归属：{_text(item.get('scene_id', 'unknown'))} / {_text(item.get('image_id', 'unknown'))}</small></article>"
        for item in payload.attributions
    )
    bias = "".join(
        f"<article><b>{_text(item.get('status', 'unknown'))}</b><p>{_text(item.get('reasons', []))}</p><p>{_text(item.get('recommended_actions', []))}</p></article>"
        for item in payload.capture_bias
    )
    external = "".join(
        f'<li>{_text(item.get("device_id", "unknown"))} / {_text(item.get("claim_id", "legacy unbound"))}: <a href="{_text(item.get("url", "#"))}">{_text(item.get("title", "Source"))}</a> · {_text(item.get("evidence_scope", "background"))}</li>'
        for item in payload.external_validation + payload.hardware_context
    )
    appendix = "".join(_claim(item, i) for i, item in enumerate(claims))
    limits = "".join(f"<li>{_text(note)}</li>" for note in payload.limitations)
    banner = "FINAL REPORT" if status == "final" else "DRAFT"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{_text(payload.project_name)}</title><style>
body{{font:16px/1.65 "Noto Sans CJK SC",Arial,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#202a34;background:#fafbf9}}h1,h2,h3{{line-height:1.3}}h2{{margin-top:48px;border-top:3px solid #2b6656;padding-top:18px}}article,details{{background:white;border:1px solid #d5ded8;border-radius:8px;padding:18px;margin:14px 0}}.banner{{background:#fff1cb;padding:15px}}.photos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}}figure{{margin:0}}img{{width:100%;height:auto;max-height:600px;object-fit:contain}}small,figcaption{{color:#56645c}}a{{color:#16624c}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #d5ded8;padding:10px;text-align:left}}@media print{{details{{display:block}}article,figure{{break-inside:avoid}}}}
</style></head><body><header><p>FRONT CAMERA · EVIDENCE REVIEW</p><h1>{_text(payload.project_name)}</h1><p class="banner">{banner} · 模型辅助观察报告 · 人工验证：{_text(payload.review_provenance.get("human_review", "pending"))}。归档为 final 不代表已完成校准。</p><p>来源：{_text(", ".join(payload.devices))}</p></header>
<h2>一页结论</h2>{summary}<p>质量缺陷、风格偏好和内部机制分别判断；当前没有人工校准的准确率。</p>
<h2>前摄难点与评判原则</h2><p>人脸明亮与光照塑形；肤色与环境色；洁净与纹理；主脸与群像；夜间可读性与氛围。评价围绕区域与实际拍摄挑战展开。</p>
<h2>数据与场景挑战矩阵</h2><p>{_text(payload.coverage)}</p><table><tr><th>输入挑战</th><th>已覆盖场景</th></tr>{matrix}</table><p>挑战来自拍摄记录或人工标注；成片亮暗只作为输出特征。</p>
<h2>按场景展开证据</h2>{"".join(sections)}
<h2>各来源的取舍与边界</h2>{profiles or "<p>跨场景证据不足，暂不形成来源画像。</p>"}
<h2>竞争机制假设与最小排查实验</h2>{causes or "<p>暂无可追溯的机制假说。</p>"}{bias}
<h2>外部背景资料</h2><ul>{external or "<li>没有可比较的外部证据。</li>"}</ul>
<h2>完整主张与证据附录</h2>{appendix or "<p>无主张。</p>"}
<h2>最终结论与使用范围</h2>{summary}<ul>{limits}</ul><p>本报告保留失败、反例、未知和缺失；模型感知能力尚须独立人工验证。</p></body></html>"""
