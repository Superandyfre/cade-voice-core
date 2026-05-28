# CADE Voice Core 项目描述（代码对齐版）

更新时间：2026-05-22
代码基线：`/home/pinggu/audio/cade-voice-core` 当前工作树

## 0. 本次修订结论

本文件已按当前代码做逐模块对齐，重点修正了以下问题：

1. 旧文档中存在目录结构漂移（列出了当前不存在的文件）。
2. `order.confirmed` 投递语义描述错误（旧文档把 `published` 写成了直接 `delivered`）。
3. ZeroMQ 命令/事件列表不完整（缺失 `order.confirmed.ack`、`outbox.retry` 等）。
4. 测试统计过时（旧文档写 305；当前 `pytest --collect-only -q` 为 352）。
5. 部分配置项和评测脚本遗漏（如 `eval_audio_replay.py`、`evals/audio/manifest.json`）。

**已完成改进：**

1. P0：统一 ROUTER ACK schema（所有 ACK 含 `ok/accepted/reason/duplicate/state/session_id/last_event_seq`）；修复 `full_pipeline_test.py` 旧 `payload.ok` 检查；实现 `OutboxRetryWorker`（后台扫描 pending/published、重发、超限 dead_letter）；接入 metrics 真实递增（outbox_*/order_recovered/asr_echo_block）；统一 `cade.__version__` 从包元数据读取。
2. P1：增强 `transition_journal.jsonl`，每步记录 `order_id`、`idempotency_key`，支持 `failed/skipped` 状态判定。
3. P2：验证语义事件集合已稳定（`valid_order/confirm_correct/confirm_wrong/cancel_request/repeat_request/menu_question/out_of_menu_item/empty_or_noise/unknown`），`item_id` 为主键，modification 信号优先走 `confirm_wrong + fix_order`。
4. P3：扩展 `cade-order-doctor` 增加版本一致性检查与 stale outbox 检测；扩展 `cade-fsm-graph --full` 展示 journal 子步骤、outbox 状态流和 Recovery 子流程。
5. **开放菜单解析**：移除分类器对非菜单项的硬拦截（`OUT_OF_MENU_ITEM` → `VALID_ORDER`），移除 LLM response_format 的 `enum` 约束，更新 LLM prompt 允许提取任意食物项（菜单名仅用于规范化），放宽后验证不再拒绝非菜单项。系统现可处理任意菜单内容。
6. **Outbox 去重重试**：`find_retryable` 和 `find_undelivered` 改为按 `order_id` 取最新条目（append-only 日志去重），修复同订单历史条目导致的重试风暴。
7. **TTS 与状态转移解耦**：`_run_ask_stage()` 和 `_run_repeat_stage()` 中状态转移移至 TTS 播放之前，`_processing_input` 在转移后立即清除。TTS 期间 FSM 已处于目标状态，用户在 TTS 结束 + 回声窗口（300ms）后即可输入，不再等待数秒。
8. **云端 LLM 支持**：已验证 Mimo v2.5-pro（`https://token-plan-cn.xiaomimimo.com/v1`）兼容 OpenAI Chat Completions 协议。10 轮全链路测试结果：规则解析路径 ~160ms/轮，LLM fallback 路径 ~2s/轮，10/10 全部通过。

---

## 1. 审计范围与方法

本次对齐基于实际代码与脚本，不依赖历史描述。已核对：

1. 顶层结构、`pyproject.toml`、`.env.example`、`menu.yml`。
2. 核心源码：`cade/config.py`、`cade/cli.py`、`cade/controller.py`、`cade/voice/session.py`。
3. ASR/TTS/LLM/FSM 全链路模块（含 `cade/fsm/runtime`、`cade/fsm/parsing`、`cade/fsm/storage`）。
4. 所有测试文件与测试收集结果（352 tests collected）。
5. 评测脚本与样本：`scripts/`、`evals/bootstrap/`、`evals/audio/manifest.json`。

---

## 2. 旧文档与代码不符清单（逐项）

### 2.1 目录结构不符

旧文档列出但当前不存在的文件：

1. `cade/fsm/runtime/callbacks.py`
2. `cade/fsm/storage/local.py`
3. `cade/fsm/storage/protocols.py`

对应实现实际在：

1. `CallbackTTSSink` / `CallbackEventSink`：`cade/fsm/order_fsm.py`
2. `LocalOrderStorage` / `LocalOrderIdProvider`：`cade/fsm/order_fsm.py`
3. `OrderStorage` / `OrderIdProvider` / `TTSSink` / `EventSink` 协议：`cade/fsm/order_fsm.py`

### 2.2 Outbox 状态流描述错误

旧文档将确认后 outbox 写为 `pending -> delivered`，与代码不符。  
当前真实流程（`_hook_commit_order`）是：

1. `pending`
2. 发布 `order.confirmed`
3. `published`

`delivered` / `dead_letter` 需要外部发 `order.confirmed.ack` 才会落盘。

### 2.3 ZeroMQ 命令列表不完整

旧文档缺失以下实际已实现命令：

1. `order.confirmed.ack`
2. `outbox.retry`

并且未清晰说明 `events.get_since` 的分页字段：`from_seq`、`to_seq`、`next_from_seq`、`has_more`、`count`。

### 2.4 事件列表有“声明实现但代码未发布”的项

在 `OrderSubFSM` 的事件发布路径中，当前不会发布：

1. `tts.playback_started`
2. `tts.interrupted`

`OrderSubFSM` 实际发布的是 `tts.request`、`tts.completed`、`tts.failed`、`tts.cache_hit/miss`、`tts.fallback_used` 等。

### 2.5 测试统计过时

旧文档写”305 项全部通过”。当前代码基线下：

1. `pytest --collect-only -q`：**352 tests collected**
2. 旧统计数已失效，不应继续写死

### 2.6 遗漏项

旧文档遗漏：

1. `scripts/eval_audio_replay.py`
2. `evals/audio/manifest.json`
3. `OrderFSMConfig` 里的 `outbox_retry_sec` / `outbox_max_attempts` 字段
4. `Config` 里的 `ECHO_SIMILARITY_THRESHOLD` / `ECHO_SIMILARITY_WINDOW_SEC`

### 2.7 已知代码现状（旧文档未说明）

以下字段此前定义但未被业务逻辑递增，**现已全部接入真实递增点**：

1. `order_recovered_total` — 在 `_recover_pending_confirmed` 恢复路径中递增 ✓
2. `outbox_delivered_total` — 在 `order.confirmed.ack` 标记 delivered 时递增 ✓
3. `outbox_dead_letter_total` — 在 `order.confirmed.ack` 标记 dead_letter 时递增 ✓
4. `asr_echo_block_total` — 在 `OrderingVoiceRuntime` ASR 回调被 SpeakingGate 拦截时递增 ✓
5. `outbox_pending_total` — 在 `_hook_commit_order` 写 pending outbox 时递增 ✓
6. `outbox_published_total` — 在 `_hook_commit_order` 写 published outbox 时递增 ✓

此外两个工程一致性问题**已修复**：

1. ~~`pyproject.toml` 版本是 `0.2.0`，`cade/__init__.py` 仍是 `0.1.0`~~ → `cade.__version__` 现从 `importlib.metadata` 读取，兜底 `"0.2.0"` ✓
2. ~~`cade.fsm.full_pipeline_test` 仍按旧 ACK 结构检查 `payload.ok`~~ → 已改为按命令类型验证 `accepted`（业务类）或 `ok`（命令类）✓

---

## 3. 项目定位与主链路

`cade-voice-core` 是 ROS-free 语音交互核心，包含两条主链：

1. 普通对话链：`ASR -> RobotController -> LLM(JSON) -> RobotAction -> TTS`
2. 点餐链：`serving_state.update(PAUSED_ORDERING) -> ASK -> LISTEN -> REPEAT -> CHECK -> FINISH -> order.confirmed`

---

## 4. 实际顶层结构（当前）

```text
cade-voice-core/
  pyproject.toml
  requirements.txt
  .env.example
  PLAN.md
  PROJECT_DESCRIPTION.md
  asr-promote-plan.md
  menu.yml
  cade/
    config.py
    cli.py
    controller.py
    voice/session.py
    asr/{engine.py,text_norm.py}
    tts/{engine.py,router.py,cache.py,normalizer.py,chunker.py,playback.py,backends/*}
    brain/{llm_client.py,prompts.py,response_formats.py,structured_backend.py,schema_export.py,schemas.py,router.py,context.py}
    body/{robot_interface.py,robot.py,safety.py,world_state.py}
    agent/{state.py,graph.py}
    fsm/
      order_fsm.py
      config.py
      events.py
      states.py
      cli.py
      graph_export.py
      full_pipeline_test.py
      full_pipeline_standalone.py
      parsing/{input_classifier.py,order_parser.py,menu_context.py,pipeline.py}
      runtime/{voice_runtime.py,zmq_runtime.py}
      storage/{outbox.py,journal.py}
      diagnostics/{checks.py}
      *_shim.py（input_classifier.py/menu_context.py/order_parser.py/voice_runtime.py/zmq_runtime.py/diagnostics.py）
  scripts/
    audio_monitor.py
    check_llm_structured_capabilities.py
    eval_audio_replay.py
    eval_llm.py
    test_nemotron_wav.py
    download_voice_models.sh
  evals/
    bootstrap/*.jsonl
    audio/manifest.json
  tests/*.py
  data/orders/
  fsm/   # 历史归档目录（含旧嵌套仓库）
```

---

## 5. 配置模型（当前代码）

### 5.1 全局配置：`cade.config.Config`

配置来源：优先 `.env`，不存在则加载 `.env.example`。  
核心分组：

1. LLM：`CADE_MODE`、`CADE_*_BASE_URL`、`CADE_*_API_KEY`、`CADE_*_MODEL`、`CADE_TEMPERATURE`、`CADE_MAX_TOKENS`、`CADE_TIMEOUT`
2. 音频：`CADE_INPUT_DEVICE`、`CADE_OUTPUT_DEVICE`、`CADE_INPUT_SOURCE`
3. ASR：主模型 + fallback + `CADE_ASR_REPLACEMENTS`
4. TTS：router/cache/chunking/playback/backends 全部开关和路径
5. 回声相关：`CADE_ECHO_SUPPRESS_MS`、`CADE_ECHO_SUPPRESS_AFTER_MS`、`CADE_ECHO_SIMILARITY_THRESHOLD`、`CADE_ECHO_SIMILARITY_WINDOW_SEC`
6. 结构化输出：`CADE_LLM_STRUCTURED_PROFILE`、`CADE_LLM_REQUIRE_STRUCTURED`

### 5.2 点餐配置：`cade.fsm.config.OrderFSMConfig`

除常规 prompt、重试、通道、菜单配置外，当前还包含：

1. `CADE_ORDER_MENU_FILE`
2. `CADE_ORDER_SESSION_SNAPSHOT_FILE`
3. `CADE_ORDER_IDEMPOTENCY_CACHE_FILE`
4. `CADE_ORDER_IDEMPOTENCY_TTL_SEC`
5. `CADE_ORDER_OUTBOX_RETRY_SEC`（由 `OutboxRetryWorker` 消费）
6. `CADE_ORDER_OUTBOX_MAX_ATTEMPTS`（由 `OutboxRetryWorker` 消费）

---

## 6. 普通语音对话链路（非点餐）

入口：

1. `cade-text-chat`：纯文本调试，走 `RobotController`
2. `cade-voice-chat`：`VoiceSession` 实时语音闭环

关键实现点：

1. `RobotController.process_input()` 产出 `decision/action_result/spoken_text/timings`
2. 安全门控在 `ActionSafetyGate.validate()`，拒绝返回 `blocked_by_safety`
3. `VoiceSession` 在 ASR 回调层做 speaking + tail window 抑制
4. `CADE_BARGE_IN_ENABLED=true` 时允许打断 TTS（调用 `TTSEngine.stop()`）

---

## 7. ASR 模块（`cade/asr/engine.py`）

支持模型：

1. `whisper`
2. `sense_voice`
3. `paraformer`
4. `transducer`
5. `streaming_zipformer`
6. `streaming_nemotron`

输入路径：

1. `sounddevice`（`start_listening`）
2. Pulse/PipeWire `parec`（`start_listening_pulse`）

真实行为：

1. streaming 模型靠 endpoint detection 切句
2. offline 模型走 Silero VAD 切段
3. Nemotron 路径默认做 `text_norm`（小写、末尾标点处理、可配置替换）
4. 支持主模型加载失败自动 fallback

---

## 8. TTS 模块（`cade/tts/*`）

`TTSEngine` 是 facade，内部组合：

1. `TTSRouter`（default/fast/fallback + 负载回退）
2. `TextNormalizer`
3. `TTSCache`（WAV 缓存 + index）
4. backend（`SherpaKokoroBackend` / `SherpaVitsBackend` / `NullBackend`）
5. `PlaybackManager`（`sounddevice_stream` + `paplay` fallback）

关键点：

1. `speak()` 保持旧接口：返回 `(playback_duration_s, audio_duration_s)`
2. `speak_detailed()` 返回 `TTSResult`（含 backend/cache/rtf/fallback 元信息）
3. 长文本可按 `SentenceChunker` 分段合成播放
4. 预热线程仅在 cache + prewarm 启用时启动

---

## 9. LLM 与结构化输出

### 9.1 `LLMClient`

关键行为：

1. OpenAI-compatible Chat Completions
2. Qwen3 `no_think` 注入逻辑（普通调用）
3. structured 请求下的渐进降级：`json_schema -> json_object -> none`
4. JSON 提取采用平衡括号扫描 + 多候选回退

### 9.2 `OpenAICompatibleStructuredBackend`

用于统一 schema 调用与统计，返回：

1. 解析后的 Pydantic 模型
2. `StructuredCallStats`（format mode、attempts、latency、fallback stage）

### 9.3 `schema_export.py`

把 Pydantic schema 清洗成更兼容的 LLM-facing JSON schema（处理 `$defs/anyOf/allOf/additionalProperties`）。

---

## 10. 点餐 FSM 核心（`cade/fsm/order_fsm.py`）

### 10.1 状态与转移

状态：`NOT_PERMITTED / PERMITTED / ASK / LISTEN / REPEAT / CHECK / FINISH`

关键转移：

1. `session.permitted`：`NOT_PERMITTED -> PERMITTED`
2. `ask.begin`：`PERMITTED -> ASK`
3. `ask.completed`：`ASK -> LISTEN`
4. `order.extracted`：`LISTEN -> REPEAT`
5. `repeat.completed`：`REPEAT -> CHECK`
6. `order.fixed`：`CHECK -> REPEAT`
7. `check.correct`：`CHECK -> FINISH`（after hook: `_hook_commit_order`）
8. `session.reset`：几乎全状态可回 `NOT_PERMITTED`

### 10.2 输入处理

1. 仅 `LISTEN/CHECK` 接受 live input
2. `input_channel_mode` 过滤 source
3. 去重窗口过滤重复文本
4. `_processing_input` 防并发重入（在状态转移后、TTS 播放前即清除，SpeakingGate 提供回声保护）
5. 使用 `InputPipeline` 先规则分类，再决定是否 fallback LLM
6. **开放菜单**：分类器不再硬拦截非菜单项为 `OUT_OF_MENU_ITEM`，而是标记为 `VALID_ORDER`（低置信度）交由 LLM 处理；LLM prompt 允许提取任意食物/饮料项，菜单名仅用于规范化；后验证不再拒绝非菜单项

### 10.3 订单提交与可靠性

`check.correct` after-hook 的提交子步骤：

1. `save_order_group`
2. `append_event`
3. `outbox_pending`
4. `publish_confirm`
5. `outbox_published`
6. `tts_finish`（soft）

并写入 `transition_journal.jsonl`，支持崩溃后步级恢复判定。

### 10.4 Outbox 重试机制

`OutboxRetryWorker`（`cade/fsm/storage/outbox.py`）在 `ZmqRuntime` 启动时随之后台运行：

1. 按 `outbox_retry_sec` 周期扫描所有 `pending/published` outbox 条目
2. **按 `order_id` 去重**：`find_retryable` 和 `find_undelivered` 只取每个订单的最新条目，避免 append-only 日志中历史条目导致的重试风暴
3. 对满足 `next_retry_ts <= now` 且 `attempt_count < outbox_max_attempts` 的条目执行重发
4. 重发后写 `published` 条目，递增 `attempt_count`、更新 `last_attempt_ts`、计算 `next_retry_ts`
5. `attempt_count >= outbox_max_attempts` 的条目标记为 `dead_letter`
6. `delivered` 条目不参与重试

Outbox 条目完整字段：`status`、`topic`、`order_id`、`idempotency_key`、`attempt_count`、`last_attempt_ts`、`next_retry_ts`、`ts`、`foods`、`foods_with_qty`、`order`。

### 10.5 恢复语义

`transition_journal.jsonl` 每步记录包含 `journal_id`、`step`、`status`（`started/committed/failed/skipped`）、`ts`、`order_id`、`idempotency_key`。

恢复规则：
- `committed` 子步骤不重跑。
- `failed` 子步骤可按策略重试或进入恢复状态。
- `pending` 子步骤先检查副作用是否已存在。
- 已确认订单（`finish_confirmed/committed`）不重复发布 `order.confirmed`，不重复 append business event。
- `order.confirmed` 可以重发（outbox retry），保持同一 `order_id/idempotency_key`。

---

## 11. 解析管线（`cade/fsm/parsing/*`）

`InputPipeline` 统一处理两类入口：

1. `process_listen`
2. `process_check`

内部组件：

1. `OrderInputClassifier`：12 类快速分类（cancel/menu/noise/out_of_scope 等）；疑似食物请求但非菜单项时返回 `VALID_ORDER`（低置信度）交由 LLM fallback
2. `DeterministicOrderParser`：规则提取订单
3. `ConfirmationParser`：确认语义解析（含 modification 优先级）
4. `MenuContextProvider`：候选菜单上下文

---

## 12. ZeroMQ 运行时与协议

### 12.1 运行时

`ZmqRuntime`：

1. 启动 PUB + ROUTER
2. 接管 `OrderSubFSM` 事件发布
3. 带 `IdempotencyStore` 的命令幂等 ACK 缓存
4. 维护 `event_seq` 与 `events.get_since` 重放
5. **统一 ACK schema**：所有 ACK 均含 `ok`、`accepted`、`reason`、`duplicate`、`state`、`session_id`、`last_event_seq`；命令类（`serving_state.update` 等）用 `ok`，业务类（`user_text.*`）用 `accepted/reason`
6. **OutboxRetryWorker**：后台线程按 `outbox_retry_sec` 扫描 `pending/published` 条目，重新发布 `order.confirmed`，递增 `attempt_count`，超过 `outbox_max_attempts` 标记 `dead_letter`；随 `ZmqRuntime.start()` 启动、`stop()` 停止
7. `health.get` 返回 `version` 字段（来自 `cade.__version__`）

### 12.2 命令（当前实现）

1. `serving_state.update`
2. `user_text.primary`
3. `user_text.secondary`
4. `order_id.propose`
5. `snapshot.get`
6. `events.get_since`
7. `health.get`
8. `session.cancel`
9. `outbox.undelivered`
10. `metrics.get`
11. `order.confirmed.ack`
12. `outbox.retry`

### 12.3 事件（`OrderSubFSM` 发布）

1. `order.state`
2. `order.confirmed`
3. `order.metrics`
4. `order.cancelled`
5. `order.warning`
6. `order.invalid_transition`
7. `order.error`
8. `order.llm_candidate`
9. `order.heartbeat`
10. `tts.request`
11. `tts.completed`
12. `tts.failed`
13. `tts.normalized`
14. `tts.backend_selected`
15. `tts.cache_hit` / `tts.cache_miss`
16. `tts.synthesis_completed`
17. `tts.playback_completed`
18. `tts.fallback_used`

---

## 13. 持久化文件布局（每个订单目录）

```text
<CADE_ORDER_BASE_DIR>/<order_id>/
  order_group.json
  events.jsonl
  outbox.jsonl
  session_snapshot.json
  transition_journal.jsonl
```

说明：

1. `order_group.json` 和 `session_snapshot.json` 为原子写
2. `events.jsonl` / `outbox.jsonl` / `transition_journal.jsonl` 为追加日志
3. outbox 目标状态是 `pending/published/delivered/dead_letter`

---

## 14. CLI 入口（来自 `pyproject.toml`）

1. `cade-text-chat`
2. `cade-voice-chat`
3. `cade-bench`
4. `cade-order-fsm`
5. `cade-order-voice`
6. `cade-order-test`
7. `cade-order-e2e`
8. `cade-order-doctor`
9. `cade-fsm-graph`

---

## 15. 测试与评测现状

### 15.1 测试

当前仓库测试文件：`tests/*.py` 共 22 个。
当前收集结果（2026-05-21）：`352 tests collected`。

可靠性提升新增测试（13 项）：
1. OutboxRetryWorker：重发 published 条目、超限 dead_letter、跳过 delivered
2. Outbox 条目含 idempotency_key
3. Metrics 递增：outbox_pending/published/delivered、order_recovered_total
4. SpeakingGate 阻断计数
5. Journal 记录 order_id/idempotency_key
6. Journal failed 步骤检测
7. 崩溃恢复步级跳过验证
8. 已确认订单不重复 outbox 条目

### 15.2 评测脚本

1. `scripts/eval_llm.py`：bootstrap JSONL 用例评测（mock/live）
2. `scripts/eval_audio_replay.py`：`InputPipeline` 语义回放评测
3. `scripts/check_llm_structured_capabilities.py`：schema 能力探针
4. `scripts/test_nemotron_wav.py`：Nemotron vs Zipformer 对比

---

## 16. 当前环境观测（本机）

仅记录与代码运行直接相关的观测结果：

1. Python：`3.13.9`
2. 已装包版本（抽样）：`sherpa_onnx 1.13.2`、`sounddevice 0.5.5`、`soundfile 0.13.1`、`psutil 7.0.0`、`openai 2.37.0`、`pydantic 2.12.4`、`zmq 27.1.0`
3. ASR 模型目录存在：Nemotron + Zipformer + `silero_vad.onnx`
4. TTS 目录当前仅存在：`vits-piper-en_US-libritts_r-medium-int8`

---

## 17. 未完成项与建议

### 17.1 已闭环项（本轮 P0-P3 完成）

1. ~~`outbox_retry_sec` / `outbox_max_attempts` 目前未实际参与重试调度~~ → `OutboxRetryWorker` 已实现并集成到 `ZmqRuntime` 生命周期 ✓
2. ~~多个 metrics 字段目前只定义未递增~~ → 全部接入真实递增点（见 2.7）✓
3. ~~包版本号存在双源不一致~~ → `cade.__version__` 从 `importlib.metadata` 读取，兜底 `”0.2.0”` ✓
4. ~~`full_pipeline_test` 旧 `payload.ok` 检查~~ → 已按命令类型分别验证 `ok` 或 `accepted` ✓
5. ~~transition_journal 缺少 order_id/idempotency_key~~ → 每步写入 `order_id` 和 `idempotency_key`，支持 `failed/skipped` 状态 ✓
6. ~~非菜单项（如 lemonade）被分类器硬拦截，LLM 无法处理~~ → 分类器返回 `VALID_ORDER`（低置信度），LLM prompt 允许提取任意食物项，后验证不再拒绝 ✓
7. ~~Outbox 重试风暴（同订单历史条目被重复处理）~~ → `find_retryable`/`find_undelivered` 按 `order_id` 去重，只取最新条目 ✓
8. ~~TTS 播放阻塞状态转移，用户需等待数秒~~ → `_run_ask_stage`/`_run_repeat_stage` 中状态转移移至 TTS 前，`_processing_input` 转移后立即清除 ✓

### 17.2 后续可选改进方向

1. `order_fsm.py` 的 storage/protocol/sink/recovery 逐步抽离到 `fsm/storage`、`fsm/runtime`、`fsm/recovery`，但当前小步优先，避免引入行为回归。
2. shim 文件短期保留；完成 import 迁移后再加 deprecation warning，最后删除。
3. modifier、多语言、库存热更新、dashboard 等业务扩展暂不在当前范围内。
4. LLM fallback 路径延迟优化：当前非菜单项需走 LLM（~2s 云端 / ~25s 本地），可考虑更激进的规则匹配或缓存机制。
5. ASR 输入缓冲：当前 `_processing_input=True` 期间 ASR 文本直接丢弃，可改为排队等待处理完成后重放。

### 17.3 文档维护建议

1. 不再写死”测试总数/全通过数”，改为”收集数 + 命令”。
2. 把协议作为”代码即真相”，文档只放当前命令/事件清单。
3. 每次改 `OrderSubFSM` 转移或 `ZmqRuntime` handler 时同步更新本文件。

