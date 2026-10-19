# 评测升级实施与验收记录

2026-10-19。实现基线：`cdf39c6438ee1cee5b54894a4a62652c12b48171`。

## 审查结论

原修改意见的五类 P0 风险均能在实际运行代码中定位。本轮已修改工程并接入回归，但完整评测方案尚未全部验收。

ICC 转换统一的是 SDR 渲染口径，不证明相机色彩准确；同模型多轮复核不是独立证据。旧 A/B/C 和 confidence 字段是未校准的规则标签/分数。本轮自动结论最高为 `SUPPORTED_SCOPED / B`；有反例则争议，缺少指标则保持 unknown。

## 实际执行路径

- `/api/projects/{id}/run?sync=true` → `EvaluationPipeline.run`。
- `/api/projects/{id}/run-full-evaluation?sync=true` → `run_evaluation_workflow` → 同一管线。
- 异步任务 → worker → `run_evaluation_workflow` → 同一管线。
- 管线读取已确认版本的配对快照，检查原图哈希；标注绑定配对版本。原文件发生变化需要重新扫描确认。
- 图像记录 → 匿名视觉/指标复核 → 命题裁决 → 家族归纳 → 内部冻结 → 外部资料 → 共用 ReportPayload → HTML/JSON/CSV/DOCX。

每轮诊断和报告使用独立目录，不覆盖旧产物。原有 `core/models_v2.py` 严格 2.0 存储契约保持不变；本轮扩展的是实际运行/报告契约 **0.2**，不代表完整 v2 推理迁移已完成。历史报告 0.1 保留兼容，不批量覆盖旧判断。

## 逐项状态

| 原任务 | 状态 | 本轮实现与未完成边界 |
|---|---|---|
| T01 | 核心完成 | 冻结配对、哈希、真实入口接入及重复运行产物隔离 |
| T02 | 核心完成 | ICC→sRGB、EXIF 1–8、源精度与解码精度、帧 0、Y/Y′、转换版本；缺失/损坏 ICC 阻止颜色命题。仅 SDR 基图，不分析 gain map；无可用 ICC 的 HEIF 保留颜色未知 |
| T03 | 部分完成 | 所有 Haar 人脸、人工框/角色/位置、无损 PNG 原生裁剪、实际模型输入编码/尺度、HTML/DOCX 展示。真实遮挡/侧脸、强检测器及跨设备人物对应验收未完成 |
| T04–T05 | 核心完成 | 区域/属性/方向/比较对象/条件匹配；反向命题冲突；取消固定客观满分。仅两个明确整图统计有公式支持，其余观察保留待复核 |
| T06 | 核心完成 | 反例阻止确认；同家族混合表现只计一个反例结果；未知家族不计独立覆盖；原始支持/反例/未知/不可比集合保留 |
| T07 | 核心完成 | claim ID＋版本＋设备＋属性绑定；片段只是背景，不计独立同图支持；外部驳回不改变内部结论 |
| T08–T09 | 工程接口完成 | 六挑战＋unknown、人工 scene_family、属性级可比性；限制同时约束观察与跨场景统计；成片亮暗不定义采集照度 |
| T10 | 核心完成 | 完整主张表、家族分母、未分配家族、复核关系及依赖；摘要/附录可追溯 |
| T11 | 部分完成 | 画像只从已裁决主张生成，带范围与反例；目前仍以编码亮度/近白比例两类窄命题为主，未完成肤色、塑形、群像等丰富策略归纳 |
| T12–T13 | 核心完成，真实交付待验收 | HTML 结论/难点/挑战/场景/画像/机制/附录；DOCX 修复字段、全脸证据与整图边界，多源分批展示；JSON 同源；线上图片受权限/签名保护。真实长报告分页仍需复核 |
| T14–T15 | 未执行 | 当前工作区没有 Camera0717 的 143 张原始资产/24 对应位置，也没有独立人审结果 |
| T16–T20 | 未执行 | 人工标注、家族划分、模型/提示词冻结和感知能力校准未开展；不能用工程夹具宣称准确率 |
| T21–T25 | 未执行 | 四臂编辑、风格/保真双门、误放行及成本统计未实现或运行；不产生自动调校准入结论 |

## 独立审查

独立审查的三项重要问题均已以先失败后通过的回归修复：外部来源驳回连带删除内部主张；跨场景统计绕过人工可比性限制；DOCX 丢失逐脸证据并裁掉整图边界。

同时修复了查看记录解盲 ID 不一致、主张 ID 缺比较对象/条件、视觉首轮收到指标派生提示、原生裁剪经 JPEG 重压缩、线上图片路径无效、模板页脚固定六机文字，以及 RAW 占位较多时的配对崩溃。存在位置歧义时仍要求人工确认。

## 验证

基线：89 项 pytest 通过。最终：`python -m pytest -q` 为 **134 passed**（1 个上游弃用警告）；`python -m ruff check src tests scripts`、`python -m ruff format --check src scripts`、`python -m mypy src/portrait_eval` 均通过，`git diff --check` 无问题。

回归覆盖 sRGB/Display-P3 色块（两码值容差）、损坏/缺失 ICC、EXIF 1–8、16-bit RGB PNG、Y/Y′、帧选择、多脸裁剪逐像素一致、错误设备/越界 ROI、版本冲突、相反命题、3 支持/21 反例、家族混合/去重、unknown 分母、跨设备/版本外部关联、HTTP 图片及签名隔离、报告字段及原生脸证据。

- 前端：Vitest 3 项通过；TypeScript/Vite 构建通过。当前标注通过 API，尚无新增可视化标注编辑器。
- 导出：使用含结构、纹理、前后两张脸和缺失源的程序化夹具；DOCX 经 LibreOffice 渲染为 4 页，逐页检查中文、图片、表格和裁剪。版式沿用现有模板，部分节标题的页尾衔接仍可优化；真实长报告须再次逐页检查。
- HTML：本地图片/锚点和实际 HTTP 访问回归通过。未安装匹配 Playwright 版本的 Chromium，未完成浏览器截图检查，不记为通过。
- 测试环境有一个 Starlette/httpx 的上游弃用警告，未静默过滤。
- 未调用收费视觉/编辑模型，未上传私人图像，未重新运行真实相机测评。

## 新标注接口

先扫描并确认配对，读取当前版本和设备 ID，再提交：

```http
POST /api/projects/{project_id}/scenes/G001/annotation
```

```json
{
  "expected_pairing_version": 3,
  "expected_revision": 0,
  "author": "reviewer",
  "reason": "核对现场记录与原图",
  "scene_family": "room-a",
  "challenge": "mixed_light",
  "attribute_comparability": {"global_exposure": "COMPARABLE_WITH_CONFOUNDERS"},
  "faces": {
    "实际设备ID": [{"bbox_xywh": [10, 20, 120, 160], "role": "SECONDARY", "position": "rear"}]
  }
}
```

`challenge`：`uniform_light / high_contrast / mixed_light / multi_person / low_light / strong_light / unknown`。坐标针对 EXIF 方向已校正的分析图。GET 同一路径读取最新修订，再次提交须提供其 revision。标注追加保存，重新确认配对后不自动套用旧标注。

Linux 中文导出需要 Noto Sans CJK 等实际可用字体。模型输入超过最大边长会记录缩放，不能宣称原生像素已经查看。Haar 漏检不等于照片没有人物。

## 实施决策

- 用户已指定方案并要求修改，本轮直接实现，不增加设计审批轮次。
- 保持 0.1 兼容并扩展实际运行契约 0.2；代价是完整严格 v2 迁移仍需后续工作。
- final 保留为文件归档状态，始终显示模型辅助、人工验证待完成；旧调用方需区分文档状态与验证状态。
- 真实数据、人审和校准结果缺失时不补写成功，后续仍须完成 T14–T25。

## 构建可重复性补充

GitHub 首次前端作业在 Node 22 自带 npm 10 的依赖解析阶段出现 `edgesOut` 异常，后端作业已通过。本轮把 CI 与前端 Docker 构建统一到本地已验证的 Node 24，提交来自 npm 公共仓库、包含完整 integrity 的 lockfile，并改用 `npm ci`；安装日志也纳入 CI 产物。新的干净安装、3 项测试和构建已在本地通过。Docker 镜像构建未在当前环境执行。

提交审阅：[PR #6](https://github.com/baolinv0/FroneCamera/pull/6)，目标分支 `agent/openai-auto-pairing-final`。
