---
document_id: WORK-REQUIREMENTS-001
document_version: 0.2.0
status: DRAFT
project: Qwen-TMQA
scope: multimodel-visual-review-and-human-calibration
---

# Original Requirements

## 1. Goal

交付一套可运行、可追溯、可验证的 Tone Mapping / RGB Enhancement 质量评估系统，使工程人员能够：

1. 查看当前数据集、场景、Alpha 档位和各评估阶段的结果；
2. 查看多个独立视觉大模型对同一图像或序列给出的维度分数、决策、置信度、问题区域和简明理由；
3. 在可视化面板中直接检查每次推理实际使用的 Prompt、Prompt 模板、输入图像顺序、模型版本、原始响应和解析结果；
4. 仅将模型分歧、质量尾部和高风险样本交给人工盲审；
5. 在人工判断成功持久化后揭示模型结果，并计算模型与人工的误差；
6. 基于人工盲审结果统计模型可靠性，为后续阈值、Prompt、模型角色和融合策略优化提供证据。

本版本的首要目标是 **Evaluation Observability**，次要目标是 **Human-in-the-loop Evaluator Calibration**。

## 2. Core decision question

> 系统是否能够在不泄露模型结论给盲审人员的前提下，完整展示和追溯当前多阶段、多模型评测结果，只将分歧、尾部和高风险样本送入人工审核，并利用保存成功的人工判断产生可复现的模型可靠性统计？

只有该问题的所有强制条件均得到实现、测试和执行证据支持，任务才可判定完成。

## 3. Mandatory requirements

### R-001 — 九级 Alpha 数据发现与完整性

系统必须支持以下默认九级目录：

```text
a_m100, a_m075, a_m050, a_m025,
a_000,
a_p025, a_p050, a_p075, a_p100
```

要求：

- `a_000` 为默认基准档位；
- 相同相对路径表示同一场景；
- `strict_complete=true` 时，必须同时验证九个目录均存在且每个场景九张图完整；
- 非图像文件不得被发现为场景；
- `recursive=false` 时不得递归发现子目录文件；
- 重复 Alpha、非法目录名称、缺失基准或不完整场景必须显式失败，不得静默跳过。

### R-002 — 确定性客观评价与阶段结果

系统必须为每个 Alpha 图像生成至少以下客观指标：

- luminance / EV；
- P20、P50、P90；
- clipping ratio；
- shadow ratio；
- contrast；
- color drift；
- edge similarity。

系统必须为完整 Alpha 序列生成至少以下控制指标：

- Spearman 单调性；
- violation rate；
- smoothness；
- dead-zone ratio；
- endpoint range；
- clipping growth；
- control score。

系统必须保存 Integrity、Hard Gate、Tone/Color、Fidelity、Control、Model Judges 等阶段的状态、分数和证据。

### R-003 — 任意数量的独立模型评估器

系统必须通过配置中的 `judges: [...]` 支持任意数量独立 Judge，并至少支持：

- deterministic mock adapter；
- OpenAI-compatible multimodal endpoint adapter。

一个 Judge 的请求、解析或验证失败不得中断其他 Judge 或整个场景；失败 Judge 必须以 `available=false`、错误信息和完整 Prompt Trace 记录。

### R-004 — 完整模型结果记录

每个模型结果必须至少包含：

- model ID、role、version/checkpoint；
- synthetic 标记；
- available 状态；
- Tone、Color、Fidelity、Control、Preference、Overall 六个分数；
- Decision；
- Confidence；
- Issues、fatal 标记、region/bbox；
- 简明、基于证据的 rationale；
- raw response；
- parsed response；
- latency；
- error。

真实 Judge 返回缺失字段、非法分值、非法决策或非法 Issue Schema 时必须 fail closed，不能使用部分结果继续评分。

### R-005 — Prompt Inspector 与推理追踪

每次模型推理必须保存并在工程调试视图中展示：

- prompt ID；
- immutable prompt version；
- rendered prompt；
- prompt template；
- rendered variables；
- prompt hash；
- output schema version；
- inference parameters；
- input manifest；
- raw response；
- parsed JSON；
- prompt version diff。

工程人员必须能够不进入源代码即可查看上述内容。

### R-006 — 明确的多图输入顺序与实际 Payload 追踪

Input Manifest 中每张图必须记录：

- contiguous index；
- role：source / baseline / candidate / crop / difference；
- Alpha；
- level；
- source path；
- source file SHA-256；
- 原始宽高；
- 发送宽高；
- 实际发送 payload SHA-256；
- payload MIME type；
- 编码或压缩参数。

Prompt 和模型请求中的图片顺序必须与 Manifest 完全一致。EXIF Orientation 必须在客观指标、Dashboard 和模型请求中保持一致。

### R-007 — Prompt 和 Schema 版本不可变

已经发布的 Prompt 版本不得原地修改。

当 Prompt 文本、必填输出字段、输出 Schema 或语义发生变化时，必须：

- 新增 Prompt 版本；
- 新增或升级 output schema version；
- 保留旧版本可读取和可对比；
- 更新默认配置；
- 在 Dashboard 中可查看版本差异。

### R-008 — 安全的最终决策逻辑

最终决策必须为：

```text
KEEP / REGENERATE / REVIEW / REJECT
```

必须满足：

- 客观 fatal 或任一可信模型报告 fatal issue 时，不得被其他高分补偿；
- 所有 Judge 不可用时必须为 `REVIEW`；
- 模型决策发生分歧时必须为 `REVIEW`；
- 所有可用模型一致 `REJECT` 时不得输出 `KEEP`；
- 所有可用模型一致 `REGENERATE` 时不得输出 `KEEP`；
- Judge unavailable 必须增加 uncertainty 并进入 review reason；
- 所有阈值必须来自版本化配置，不得在测试中临时放宽。

### R-009 — 可视化结果展示

Dashboard 必须包含：

1. Dataset Overview；
2. Scene Inspector；
3. Alpha Filmstrip；
4. Difference Map；
5. Stage Trace；
6. Model-by-dimension Matrix；
7. Prompt Inspector；
8. Focused Review Queue；
9. Human Blind Review；
10. Model Reliability。

必须显示：

- 数据集场景数、分数统计和决策分布；
- 每档 Alpha 图像及其客观指标；
- 各阶段评分和证据；
- 不同模型各维度分数、Decision、Confidence、Unavailable 状态和分差；
- 人工审核后的模型—人工差值；
- 模型可靠性、决策一致率和融合权重。

界面必须为 UTF-8，可在主流桌面浏览器使用，不得出现乱码。

### R-010 — 真正的数据层人工盲审

人工 Gold Label 模式在提交成功前，审核客户端不得接收到：

- 模型名称和角色；
- 模型分数；
- 模型决策；
- 模型 Prompt；
- 模型 rationale；
- 系统最终 Decision；
- 完整工程评估 JSON。

仅通过 CSS、DOM 隐藏或 JavaScript 状态隐藏不算盲审。

系统必须拆分：

- reviewer-safe payload；
- engineering payload；
- post-submit reveal payload。

只有服务端确认人工记录成功后，才允许返回并显示模型结果和 Prompt。

### R-011 — 聚焦人工审核队列

人工审核页面只能选择自动生成的 Review Queue 中的场景。

Queue 至少由以下原因产生：

- model score disagreement；
- model decision disagreement；
- low-tail score；
- fatal / hard risk；
- configured high-risk evidence；
- Judge unavailable；
- OOD 或 branch conflict（若实现）。

普通干净样本不得出现在人工盲审选择器中，但可在工程 Scene Inspector 中查看。

前端和服务端都必须验证 Queue 授权；仅前端过滤不算满足要求。

### R-012 — 人工审核记录完整性

人工审核记录必须至少包含：

- review ID；
- scene ID；
- non-empty reviewer ID；
- blind-review 标记；
- Decision；
- Overall 和可选维度分数；
- issues；
- regions；
- rationale；
- confidence；
- synthetic 标记；
- server-generated received timestamp。

真实 Gold Label 必须为 blind review。记录必须 append-only；不得静默覆盖原始 JSONL。

同一 `(scene_id, reviewer_id)` 的重复提交在校准视图中仅使用服务器接收时间最新的一条；不同 reviewer 必须作为独立证据保留。

### R-013 — Synthetic 与真实人工证据隔离

Synthetic Judge 和 synthetic human review 必须显式标记。

生产校准默认不得使用 synthetic review。只有显式实验开关允许 synthetic 数据参与，并且输出必须醒目标记为实验结果。

混合 synthetic 和真实人工记录时，系统必须拒绝或要求显式选择，不能把 synthetic 记录隐式计入真实模型可靠性。

### R-014 — 人工校准与可靠性统计

系统必须基于有效人工盲审计算：

- Overall MAE；
- per-dimension MAE；
- Decision agreement；
- pairwise model gap；
- sample count；
- reliability/fusion weight。

校准输出必须记录所使用 review 集合的类型、数量和生成时间，并明确说明小样本和 synthetic 结果不可作为生产权重。

### R-015 — 服务端安全与失败行为

本地 Review Server 默认只监听 `127.0.0.1`。

必须：

- 校验 Content-Length；
- 校验 JSON 和 Pydantic Schema；
- 拒绝 Queue 外场景；
- 拒绝空 Reviewer ID；
- 拒绝非盲真实 Gold Label；
- 由服务端生成接收时间；
- 不把 API key、Authorization header 或其他 secret 写入 Dashboard、JSON 或日志；
- 存储失败时不得提前揭示模型结果。

### R-016 — CLI 和可复现实验

系统必须提供并验证：

```text
qwen-tmqa validate-config
qwen-tmqa make-example
qwen-tmqa evaluate
qwen-tmqa visualize
qwen-tmqa serve
qwen-tmqa simulate-human
qwen-tmqa calibrate
qwen-tmqa audit
```

Synthetic experiment 必须可重复运行，并覆盖至少：

- 正常序列；
- 高光裁剪；
- 色彩偏移；
- 结构变化；
- Alpha 反转；
- Dead Zone；
- Judge endpoint failure；
- invalid JSON；
- invalid output schema；
- unanimous REJECT / REGENERATE；
- Queue 外人工提交；
- synthetic/real calibration isolation。

### R-017 — 自动需求审计与可追溯性

机器审计必须逐项检查所有 MUST 需求的关键输出和契约，但机器审计不得仅通过搜索 HTML 字符串判定安全属性。

每个 MUST Requirement 必须映射到：

- implementation location；
- test location；
- execution command；
- measurable threshold；
- actual evidence；
- final status。

任何 MUST 未通过时，审计结果必须为 FAIL。

### R-018 — 构建、兼容性和 CI

必须支持 Python 3.10 和 3.12。

必须通过：

- compileall；
- Ruff；
- full pytest；
- Wheel 和 sdist build；
- Wheel isolated install/import；
- CLI smoke test；
- synthetic end-to-end experiment；
- Dashboard JavaScript syntax check；
- requirements audit；
- GitHub Actions on branch head。

CI 未运行或失败时不得宣称完成或合并。

### R-019 — 证据边界与非误导性声明

Synthetic Judge 或 synthetic human experiment 只能证明数据流、可视化、审核和校准代码可运行，不能证明真实 Qwen、InternVL、Ovis、MiniCPM 的评分准确率。

在没有真实模型 endpoint 和人工 Gold Set 时，不得声称：

- 真实模型可靠性已验证；
- 当前融合权重可用于生产；
- 当前系统是训练型 Reward Model；
- 当前系统具有物理正确的 HDR source-aware 评价能力。

## 4. Optional requirements

- `O-001`: 支持真实 Qwen3-VL、InternVL、Ovis、MiniCPM endpoint 的小规模 canary；
- `O-002`: 支持语义区域、face、OCR、identity 和 local crop 专用检测器；
- `O-003`: 支持 OOD 检测、风险覆盖曲线和校准置信度；
- `O-004`: 支持 reviewer assignment、双人一致性和专家仲裁；
- `O-005`: 支持将人工标签导出为 Pointwise、Pairwise 和 Sequence 训练数据；
- `O-006`: 支持真实 HDR / linear source adapter 和 source-conditioned metrics；
- `O-007`: 支持缓存、并行、断点续跑和大规模数据集进度显示。

Optional requirements 未实现不得阻止本版本完成，也不得被默认描述为已实现。

## 5. Non-goals

本版本不要求：

- 训练或微调专用 Qwen-IQA / Reward Model；
- 证明任何真实大模型的 Tone Mapping 评分准确率；
- 自动调用图像生成模型完成重生成闭环；
- 物理 HDR、RAW、linear-domain source fidelity；
- 生产级用户认证、RBAC 或公网部署；
- 复杂多人标注任务分配系统；
- 将 synthetic review 当作真实 Gold Label；
- 以 Overall 均值替代 fatal hard gate；
- 修改冻结需求、验收阈值或历史 Prompt 来使测试通过。

## 6. Inputs

### 6.1 Dataset input

- 根目录包含版本化 Alpha level 子目录；
- 默认期望九级目录由 R-001 定义；
- 支持 `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, `.webp`；
- 图像必须可由 Pillow 解码；
- 同一场景各档位必须具有一致的语义内容和可比较尺寸；
- 可选 source 图像只能作为追踪和模型输入使用，除非未来版本实现真实 source-conditioned metrics。

### 6.2 Configuration input

YAML 必须定义：

- dataset settings；
- expected levels；
- baseline level；
- Judge list；
- model role、adapter、model、version、endpoint；
- Prompt ID/version；
- inference parameters；
- review thresholds；
- visualization settings。

未知关键字段、非法阈值和缺失 remote endpoint 必须在启动前失败。

### 6.3 Model response input

真实 Judge 必须返回满足版本化 JSON Schema 的对象。非法 JSON、缺失字段、非法范围和未知决策必须被隔离为 unavailable evidence。

### 6.4 Human review input

只接受服务端授权 Queue 中的 scene ID，并满足 R-012、R-013 和 R-015。

## 7. Outputs

系统至少生成：

```text
results/
├── evaluations.json
├── summary.json
├── errors.json
└── scenes.csv

dashboard/
├── index.html
├── assets/images/
└── data/
    ├── engineering_evaluations.json
    ├── reviewer_payload.json
    └── review_queue.json

human_reviews.jsonl
calibration.json
requirements_audit.json
```

输出必须：

- 使用 UTF-8；
- 包含 schema/version 信息；
- 可追溯到输入文件、Prompt、模型版本和配置；
- 不泄露 API key 或 secret；
- reviewer payload 不包含盲审禁用字段；
- 失败场景记录错误，不得伪装为成功。

## 8. Invariants

以下属性必须始终成立：

1. Fatal 内容或身份错误不可被其他维度高分补偿；
2. 未成功保存人工判断前，不向审核客户端发送模型结果；
3. Synthetic review 默认不参与真实校准；
4. Prompt 版本不可原地修改；
5. `strict_complete=true` 时九级目录和九张图必须完整；
6. 所有 Judge unavailable 时不能输出 KEEP；
7. 所有模型一致 REJECT 或 REGENERATE 时不能输出 KEEP；
8. 单 Judge 失败不能删除或覆盖其他 Judge 结果；
9. Append-only 原始人工记录不得被重写；
10. 不允许删除、skip、xfail、削弱测试或放宽阈值来取得 PASS；
11. 不允许把静态代码检查描述为真实执行结果；
12. 不允许把 synthetic experiment 描述为真实模型准确率证据。

## 9. Constraints

- Python：3.10、3.12；
- 核心运行环境：Linux 和 Windows 可安装；
- 浏览器：当前主流 Chromium/Edge；
- Dashboard：UTF-8，自包含或通过本地 server 提供；
- Review Server 默认仅绑定 localhost；
- Synthetic experiment 必须在固定输入与配置下确定性复现；
- 测试不得依赖外部真实模型 endpoint；
- 真实 endpoint 测试必须与 synthetic/mock 证据明确区分；
- 任何未执行检查必须标记为 `NOT_RUN` 或 `BLOCKED`；
- 性能和内存仅在存在明确验收阈值时作为阻塞条件。

## 10. Allowed change scope

允许修改：

- `src/qwen_tmqa/**`；
- `tests/**`；
- `configs/**`；
- `scripts/**`；
- `docs/**`；
- `.github/workflows/**`；
- `pyproject.toml`；
- README 和版本化发布说明。

允许新增：

- Prompt / Schema 新版本；
- review-safe API；
- server-side reveal API；
- 新测试、实验和验收证据。

## 11. Forbidden change scope

未经人类批准，不得：

- 修改本文件中的 MUST 需求或完成阈值；
- 将 `workflow/01_SPEC_BASELINE.json` 标记为 FROZEN；
- 修改历史 Prompt 版本的内容；
- 删除或降低原有测试覆盖；
- 修改 Gold Label 以使模型看起来更准确；
- 将 synthetic 输出改标为真实；
- 绕过 Queue 授权或盲审隔离；
- 在输出中写入 secret；
- 合并到发布分支或部署到生产环境。

需求变更必须通过 `workflow/05_CHANGE_REQUESTS.md`、版本升级、重新生成 Hash 和人类重新批准。

## 12. Deliverables

必须交付：

1. 完整可浏览源代码；
2. 版本化配置文件；
3. 所有自动测试；
4. 真实 Judge adapter 和 mock/synthetic adapter；
5. Dataset、Scene、Stage、Model、Prompt、Review、Queue、Reliability Dashboard；
6. reviewer-safe 数据接口和 post-submit reveal 接口；
7. Append-only review storage；
8. Calibration output；
9. 自动 Requirements Audit；
10. Synthetic end-to-end experiment；
11. Wheel 和 sdist；
12. README、实验报告、需求矩阵和限制说明；
13. GitHub Actions 执行证据；
14. Requirement-to-code/test/evidence traceability；
15. 所有已知残余风险和未执行检查清单。

## 13. Done when

任务仅在以下条件全部满足时完成：

1. 本文件和对应 Acceptance Matrix、Verification Plan 已由人类审查；
2. `workflow/01_SPEC_BASELINE.json` 由人类标记为 `FROZEN`；
3. 所有 MUST Requirement 均具有实现、测试、执行命令、阈值和证据映射；
4. compileall、Ruff、完整 pytest、build、isolated Wheel import、CLI smoke、synthetic E2E、JavaScript syntax check 和 requirements audit 全部 exit code 0；
5. branch-head GitHub Actions 全部通过；
6. 真盲审数据隔离测试通过，审核客户端在提交前无法取得模型结果；
7. unanimous REJECT / REGENERATE、all-unavailable、fatal issue 和 decision-disagreement 回归测试通过；
8. 九级 strict completeness 测试通过；
9. Prompt version immutability 测试通过；
10. Synthetic/real calibration isolation 测试通过；
11. 未解决 P0/P1 数量为 0；
12. 未解决 P2 已由人类在 Risk Acceptance 中明确接受；
13. Independent Critic 返回 `PASS_TO_EVALUATOR`；
14. Fresh Evaluator 返回 `PASS`；
15. Project Sources、Hash 和版本清单已刷新并由人类确认。

“页面存在”“测试文件存在”“需求审计字符串为 PASS”本身不构成完成证据；必须有真实行为测试和执行日志支持。
