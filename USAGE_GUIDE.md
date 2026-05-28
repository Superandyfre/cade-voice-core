# CADE Voice Core 使用说明

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 环境准备](#2-环境准备)
- [3. 快速开始](#3-快速开始)
- [4. CLI 命令详解](#4-cli-命令详解)
- [5. ZeroMQ 协议详解](#5-zeromq-协议详解)
- [6. 点餐 FSM 工作流](#6-点餐-fsm-工作流)
- [7. 配置参考](#7-配置参考)
- [8. 菜单配置](#8-菜单配置)
- [9. 持久化与可靠性](#9-持久化与可靠性)
- [10. 测试与诊断](#10-测试与诊断)
- [11. 目录结构](#11-目录结构)
- [12. 常见问题](#12-常见问题)

---

## 1. 项目简介

CADE Voice Core 是一个 **ROS-free** 的语音交互核心模块，运行在 Linux 上，提供两条主链路：

| 链路 | 说明 | 入口命令 |
|------|------|----------|
| 普通对话 | `ASR → LLM → TTS` 自由对话 | `cade-voice-chat` |
| 点餐服务 | `ASK → LISTEN → REPEAT → CHECK → FINISH` 受限 FSM | `cade-order-voice` / `cade-order-fsm` |

### 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  CADE Voice Core                     │
│                                                     │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐          │
│  │   ASR   │──▶│   LLM   │──▶│   TTS    │          │
│  │ Engine  │   │ Client  │   │ Engine   │          │
│  └─────────┘   └─────────┘   └──────────┘          │
│       ▲                           │                  │
│       │                           ▼                  │
│  ┌─────────┐              ┌──────────┐              │
│  │ Mic /   │              │ Speaker /│              │
│  │ Pulse   │              │ Pulse    │              │
│  └─────────┘              └──────────┘              │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │           点餐 Sub-FSM                       │    │
│  │  NOT_PERMITTED → PERMITTED → ASK → LISTEN   │    │
│  │  → REPEAT → CHECK → FINISH → NOT_PERMITTED  │    │
│  └─────────────────────────────────────────────┘    │
│       │ ZMQ PUB (5555)    ▲ ZMQ ROUTER (5556)       │
└───────┼────────────────────┼────────────────────────┘
        │                    │
        ▼                    │
  ┌───────────┐              │
  │ 外部系统   │──────────────┘
  │ (平板等)   │  发送命令
  └───────────┘
```

---

## 2. 环境准备

### 2.1 系统要求

- **操作系统**：Linux（需要 PulseAudio 或 PipeWire 音频服务）
- **Python**：3.10+
- **音频设备**：至少一个输入（麦克风）和一个输出（扬声器）

### 2.2 安装

```bash
cd /home/pinggu/audio/cade-voice-core

# 安装依赖
pip install -r requirements.txt

# 以可编辑模式安装本项目（注册 CLI 命令）
pip install -e .
```

### 2.3 环境变量

复制示例文件并按需修改：

```bash
cp .env.example .env
```

**必填项**（仅 CLOUD 模式）：

```bash
CADE_MODE=CLOUD
CADE_CLOUD_API_KEY=sk-your-api-key
```

**使用本地 LLM 时**：

```bash
CADE_MODE=LOCAL
CADE_LOCAL_BASE_URL=http://127.0.0.1:8080/v1
```

### 2.4 模型文件

需要下载以下模型到对应目录：

| 组件 | 默认目录 | 说明 |
|------|----------|------|
| ASR 主模型 | `models/asr/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-*` | Nemotron streaming |
| ASR fallback | `models/asr/sherpa-onnx-streaming-zipformer-en-20M-*` | Zipformer 轻量 |
| VAD | `models/asr/silero_vad.onnx` | Silero VAD |
| TTS (VITS) | `models/tts/vits-piper-en_US-libritts_r-medium-int8` | 默认 TTS |
| TTS (Kokoro) | `models/tts/kokoro-en-v0_19` | 高质量 TTS（可选） |
| TTS (Piper) | `models/tts/vits-piper-en_US-lessac-medium` | 快速 TTS（可选） |

### 2.5 验证安装

```bash
# 运行诊断检查
cade-order-doctor --json

# 运行测试套件
pytest tests/ -q
```

---

## 3. 快速开始

### 3.1 纯文本对话（无需音频设备）

```bash
# 需要 LLM 服务（云端或本地）
cade-text-chat
# 输入文字，机器人回复，输入 quit 退出
```

### 3.2 语音对话（需要麦克风和扬声器）

```bash
# 使用云端 LLM
cade-voice-chat

# 使用本地 LLM
cade-voice-chat   # 需在 .env 设置 CADE_MODE=LOCAL
```

### 3.3 使用本地 LLM 测试（三终端启动）

使用 llama-server 作为本地 OpenAI-compatible LLM 推理服务，无需云端 API 即可完成全链路测试。

**前提条件**：已编译 llama.cpp 并准备好 GGUF 格式模型文件。

#### 第一步：终端 1 — 启动本地 LLM 服务

```bash
# 启动 llama-server（OpenAI-compatible 接口）
/home/pinggu/audio/llama.cpp/build/bin/llama-server \
  -m /home/pinggu/audio/Qwen3.5-9B-UD-Q8_K_XL.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 4096
```

启动成功后会显示：
```
llama server listening at http://127.0.0.1:8080
```

> **参数说明**：
> - `-m`：GGUF 模型文件路径，支持 Qwen、LLaMA、Mistral 等主流模型
> - `-ngl 99`：全部层加载到 GPU（无 GPU 可改为 `-ngl 0` 纯 CPU 推理）
> - `-c 4096`：上下文窗口长度，4096 足够点餐场景使用

#### 第二步：终端 2 — 启动点餐 FSM 服务

```bash
# 使用本地模式启动 FSM（Headless，无音频设备）
CADE_MODE=LOCAL cade-order-fsm
```

启动后显示：
```
CADE Ordering Sub-FSM (headless ZeroMQ)
  LLM: Local (http://127.0.0.1:8080/v1)
  PUB:  tcp://0.0.0.0:5555
  ROUTER: tcp://0.0.0.0:5556
```

#### 第三步：终端 3 — 运行集成测试

```bash
# 自动走完完整点餐流程：健康检查 → 开启服务 → 点餐 → 确认 → 验证
CADE_MODE=LOCAL cade-order-test
```

测试通过后输出：
```
✓ health check passed
✓ snapshot received (state=NOT_PERMITTED)
✓ serving_state update → PAUSED_ORDERING
✓ state reached LISTEN
✓ user text accepted → REPEAT
✓ state reached CHECK
✓ user text accepted → FINISH
✓ order.confirmed received
7/7 checks passed
```

#### 其他本地 LLM 启动方式

```bash
# 带真实麦克风 + 扬声器的语音点餐
CADE_MODE=LOCAL cade-order-voice

# 真实设备端到端测试（需要实际说话）
CADE_MODE=LOCAL cade-order-e2e

# 文本对话测试（不需要 FSM）
CADE_MODE=LOCAL cade-text-chat
```

### 3.4 点餐服务（Headless ZMQ 模式）

```bash
# 终端 1：启动点餐 FSM 服务（无音频，仅 ZMQ）
cade-order-fsm

# 终端 2：运行集成测试
cade-order-test
```

### 3.5 点餐服务（真实音频模式）

```bash
# 使用真实麦克风 + 扬声器
cade-order-voice

# 跳过音频设备探测（调试用）
cade-order-voice --skip-audio-probe

# 指定 LLM 模式
cade-order-voice --local
cade-order-voice --cloud
```

---

## 4. CLI 命令详解

### 4.1 `cade-text-chat` — 文本对话

纯文本 REPL，用于测试 LLM 决策，不涉及 ASR/TTS。

```bash
cade-text-chat
```

交互方式：输入文字 → 回车 → 机器人回复。输入 `quit` 或 `exit` 退出。

### 4.2 `cade-voice-chat` — 语音对话

完整语音闭环：麦克风 → ASR → LLM → TTS → 扬声器。

```bash
cade-voice-chat
```

按 `Ctrl+C` 退出。支持通过 `CADE_BARGE_IN_ENABLED=true` 开启打断功能。

### 4.3 `cade-bench` — 性能基准测试

```bash
# 全部 smoke test
cade-bench --full

# 单独测试
cade-bench --llm-smoke                    # LLM JSON 决策
cade-bench --order-llm-smoke              # 点餐 LLM 对抗测试
cade-bench --tts-smoke                    # TTS 合成
cade-bench --tts-cpu                      # TTS CPU benchmark（含多后端对比）
cade-bench --asr-smoke --wav test.wav     # ASR 转录测试

# 输出 JSON 报告
cade-bench --tts-cpu --output-json results.json
```

### 4.4 `cade-order-fsm` — 点餐 FSM（Headless）

启动后暴露 ZMQ PUB + ROUTER 接口，等待外部系统发送命令。自身无音频设备。

```bash
cade-order-fsm
```

启动后显示：
```
CADE Ordering Sub-FSM (headless ZeroMQ)
  Robot: LARA
  LLM: Cloud (deepseek-chat)
  PUB:  tcp://0.0.0.0:5555
  ROUTER: tcp://0.0.0.0:5556
  Order dir: data/orders
```

### 4.5 `cade-order-voice` — 点餐服务（真实音频）

带真实 ASR/TTS 的点餐运行时，同时暴露 ZMQ 接口。

```bash
cade-order-voice [选项]

选项：
  --pub ADDR           PUB 绑定地址 (默认: tcp://127.0.0.1:5555)
  --router ADDR        ROUTER 绑定地址 (默认: tcp://127.0.0.1:5556)
  --input-device DEV   音频输入设备
  --output-device DEV  音频输出设备
  --log-level LEVEL    日志级别 (默认: INFO)
  --skip-audio-probe   跳过音频设备验证
  --local              使用本地 LLM
  --cloud              使用云端 LLM
```

### 4.6 `cade-order-test` — 全链路集成测试

连接已运行的 FSM 服务，走完一遍完整的点餐流程。

```bash
# 需要先启动 cade-order-fsm
cade-order-test

# 自定义地址
cade-order-test --pub tcp://192.168.1.100:5555 --router tcp://192.168.1.100:5556

# 自定义点餐文本
cade-order-test --order-text "two burgers and a coke" --confirm-text "yes that's right"
```

测试步骤：health → snapshot → serving_state(PAUSED_ORDERING) → 等待 LISTEN → 发送点餐文本 → 等待 CHECK → 发送确认文本 → 验证 order.confirmed。

### 4.7 `cade-order-e2e` — 真实设备端到端测试

使用真实麦克风和扬声器，要求用户实际说话完成点餐。

```bash
cade-order-e2e [选项]

选项：
  --pub ADDR           PUB 地址
  --router ADDR        ROUTER 地址
  --input-device DEV   输入设备
  --output-device DEV  输出设备
  --local / --cloud    LLM 模式
```

验证条件：状态经过 LISTEN/CHECK/FINISH、收到 order.confirmed、收到 tts.completed、最终回到 NOT_PERMITTED、order_group.json 文件一致性。

### 4.8 `cade-order-doctor` — 诊断工具

```bash
# 终端输出
cade-order-doctor

# JSON 输出
cade-order-doctor --json

# 写入报告文件
cade-order-doctor --json --write-report diagnostics/latest.json
```

检查项：
- 版本一致性（`cade.__version__` vs `pyproject.toml`）
- TTS / ASR / LLM 模块导入
- ZMQ socket 绑定
- 菜单文件完整性与 alias 去重
- 订单目录读写
- 长期未投递 outbox 条目（>1 小时）

### 4.9 `cade-fsm-graph` — FSM 状态图导出

```bash
# 基础 Mermaid 图
cade-fsm-graph

# 含可靠性子流程
cade-fsm-graph --full

# DOT 格式
cade-fsm-graph --format dot
```

---

## 5. ZeroMQ 协议详解

### 5.1 通信模式

| Socket 类型 | 端口 | 方向 | 用途 |
|-------------|------|------|------|
| PUB | 5555 | 服务端 → 客户端 | 广播事件（state change、order.confirmed 等） |
| ROUTER | 5556 | 双向 | 接收命令，返回 ACK |

### 5.2 消息格式

所有消息使用标准 JSON envelope：

```json
{
  "v": 1,
  "type": "命令或事件类型",
  "id": "唯一消息ID",
  "ts": 1700000000.123,
  "source": "来源标识",
  "session_id": 1,
  "client_id": "可选-客户端标识",
  "client_msg_id": "可选-客户端消息ID",
  "idempotency_key": "可选-幂等键",
  "payload": {}
}
```

### 5.3 命令列表

#### `serving_state.update` — 触发/结束点餐会话

```json
{
  "type": "serving_state.update",
  "id": "msg-001",
  "payload": {
    "state": "PAUSED_ORDERING",
    "customer_id": "customer_123",
    "customer_no": "5"
  }
}
```

`state` 取值：`IDLE`、`PAUSED_ORDERING`。发送 `PAUSED_ORDERING` 启动点餐，发送其他值重置 FSM。

#### `user_text.primary` / `user_text.secondary` — 发送文本输入

```json
{
  "type": "user_text.primary",
  "id": "msg-002",
  "payload": {
    "text": "I want a coke and two waters"
  }
}
```

#### `order_id.propose` — 建议订单号

```json
{
  "type": "order_id.propose",
  "id": "msg-003",
  "payload": {
    "order_id": "12345"
  }
}
```

5 位数字。必须在 FSM 创建订单前发送。

#### `order.confirmed.ack` — 确认订单已送达

```json
{
  "type": "order.confirmed.ack",
  "id": "msg-004",
  "payload": {
    "order_id": "12345",
    "status": "delivered"
  }
}
```

`status` 只能是 `"delivered"` 或 `"dead_letter"`。

#### `outbox.retry` — 手动触发重试

```json
{
  "type": "outbox.retry",
  "id": "msg-005",
  "payload": {
    "order_id": "12345"
  }
}
```

不指定 `order_id` 则重试所有未送达条目。

#### `outbox.undelivered` — 查询未送达条目

```json
{
  "type": "outbox.undelivered",
  "id": "msg-006",
  "payload": {}
}
```

#### `snapshot.get` — 获取当前快照

```json
{
  "type": "snapshot.get",
  "id": "msg-007",
  "payload": {}
}
```

返回：`state_event`、`order_snapshot`、`session_snapshot`、`metrics`、`outbox_pending_count`。

#### `events.get_since` — 重放历史事件

```json
{
  "type": "events.get_since",
  "id": "msg-008",
  "payload": {
    "last_event_seq": 100,
    "max_events": 500
  }
}
```

返回：`events`（事件数组）、`from_seq`、`to_seq`、`next_from_seq`、`has_more`、`count`。

#### `health.get` — 健康检查

```json
{
  "type": "health.get",
  "id": "msg-009",
  "payload": {}
}
```

#### `metrics.get` — 获取指标

```json
{
  "type": "metrics.get",
  "id": "msg-010",
  "payload": {}
}
```

#### `session.cancel` — 取消当前会话

```json
{
  "type": "session.cancel",
  "id": "msg-011",
  "payload": {
    "reason": "external_cancel"
  }
}
```

### 5.4 统一 ACK 格式

所有命令的 ACK 都包含以下字段：

```json
{
  "v": 1,
  "type": "命令类型.ack",
  "id": "服务端分配ID",
  "ts": 1700000000.456,
  "source": "voice-core",
  "session_id": 1,
  "last_event_seq": 481,
  "payload": {
    "ok": true,
    "accepted": true,
    "reason": null,
    "duplicate": false,
    "state": "LISTEN",
    "session_id": 1,
    "last_event_seq": 481
  }
}
```

| 字段 | 说明 |
|------|------|
| `ok` | 命令层操作是否成功（命令类使用） |
| `accepted` | 业务层是否接受（`user_text.*` 使用） |
| `reason` | 拒绝原因（如 `"invalid_state"`、`"duplicate_input"`） |
| `duplicate` | 是否为重复命令（幂等缓存命中） |
| `state` | 当前 FSM 状态 |
| `session_id` | 当前会话 ID |
| `last_event_seq` | 最新事件序号（用于断线重连） |

### 5.5 事件列表

FSM 通过 PUB socket 广播的事件：

| 事件 | 触发时机 |
|------|----------|
| `order.state` | 每次 FSM 状态转移 |
| `order.confirmed` | 订单确认提交 |
| `order.cancelled` | 订单被取消 |
| `order.metrics` | 会话结束时发布指标 |
| `order.error` | 处理异常 |
| `order.warning` | 重试超限等警告 |
| `order.invalid_transition` | 非法状态转移尝试 |
| `order.llm_candidate` | LLM 与规则解析不一致时发出 |
| `order.heartbeat` | 心跳（默认每 2 秒） |
| `tts.request` | 请求 TTS 播放 |
| `tts.completed` | TTS 播放完成 |
| `tts.failed` | TTS 播放失败 |
| `tts.cache_hit` / `tts.cache_miss` | TTS 缓存命中/未命中 |
| `tts.fallback_used` | TTS 使用了 fallback 后端 |

---

## 6. 点餐 FSM 工作流

### 6.1 状态转移图

```
NOT_PERMITTED ──[serving_state=PAUSED_ORDERING]──▶ PERMITTED
PERMITTED ──[ask.begin]──▶ ASK
ASK ──[ask.completed]──▶ LISTEN
LISTEN ──[order.extracted]──▶ REPEAT
REPEAT ──[repeat.completed]──▶ CHECK
CHECK ──[repeat.retry]──▶ REPEAT
CHECK ──[order.fixed]──▶ REPEAT
CHECK ──[wrong.without_fix]──▶ LISTEN
CHECK ──[check.correct]──▶ FINISH ──[session.reset]──▶ NOT_PERMITTED
任意状态 ──[session.reset]──▶ NOT_PERMITTED
```

### 6.2 典型交互流程

```
外部系统                       FSM                        TTS
   │                           │                          │
   │─ serving_state.update ──▶│                          │
   │  (PAUSED_ORDERING)        │                          │
   │                           │──── tts: "What would ──▶│
   │                           │     you like to order?"  │
   │                           │◀── LISTEN                │
   │─ user_text.primary ─────▶│                          │
   │  "one coke"               │                          │
   │                           │──── tts: "Let me ──────▶│
   │                           │     confirm. You ordered │
   │                           │     coke. Is that right?"│
   │                           │◀── CHECK                 │
   │─ user_text.primary ─────▶│                          │
   │  "yes"                    │                          │
   │                           │─── order.confirmed ─────▶│ (PUB)
   │                           │──── tts: "OK I'll ────▶│
   │                           │     get coke for you"    │
   │                           │◀── NOT_PERMITTED         │
```

### 6.3 FINISH 阶段提交子步骤

订单确认时，`check.correct` after-hook 按以下顺序原子提交：

1. **save_order_group** — 写入 `order_group.json`（stage=confirmed）
2. **append_event** — 追加 `order_confirmed` 到 `events.jsonl`
3. **outbox_pending** — 写 `pending` 状态到 `outbox.jsonl`
4. **publish_confirm** — 通过 ZMQ PUB 广播 `order.confirmed`
5. **outbox_published** — 写 `published` 状态到 `outbox.jsonl`
6. **tts_finish** — 播放确认语音

每个步骤通过 `transition_journal.jsonl` 记录 `started/committed` 状态，崩溃后可据此跳过已完成的步骤。

### 6.4 Outbox 状态流

```
pending ──▶ published ──▶ delivered     （外部发 order.confirmed.ack）
pending ──▶ published ──▶ dead_letter    （超过 outbox_max_attempts）
published ──▶ published                  （retry 重发）
```

`OutboxRetryWorker` 后台线程自动扫描超时的 `pending/published` 条目并重发。

### 6.5 用户输入分类

在 `LISTEN` 和 `CHECK` 状态下，用户文本先经过 `InputPipeline` 分类：

| 分类 | 说明 | LISTEN 行为 | CHECK 行为 |
|------|------|-------------|------------|
| `valid_order` | 包含菜单项 | 解析订单 → REPEAT | — |
| `confirm_correct` | 确认 | — | → FINISH |
| `confirm_wrong` | 否认（含修改） | — | 解析修改 → REPEAT |
| `cancel_request` | 取消 | → NOT_PERMITTED | → NOT_PERMITTED |
| `repeat_request` | 要求重述 | 重复 prompt | → REPEAT |
| `menu_question` | 询问菜单 | 列出菜单项 | 列出菜单项 + 要求确认 |
| `out_of_menu_item` | 不在菜单 | 提示不可用 | — |
| `empty_or_noise` | 空输入/噪声 | 重试提示 | 重试提示 |
| `smalltalk` | 闲聊 | 礼貌回复 | 提示确认 |
| `unknown` | 未识别 | fallback LLM | fallback LLM |

**修改信号优先规则**："yes but add a coke"、"ok actually two waters" 等包含 `but/add/remove/change/instead` 等信号的输入，即使包含肯定词，也始终走 `confirm_wrong + fix_order`，不会直接确认订单。

---

## 7. 配置参考

所有配置通过环境变量设置，在 `.env` 文件或 shell 中定义。布尔值接受 `true/1/yes`。

### 7.1 LLM 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CADE_MODE` | `CLOUD` | `CLOUD` 或 `LOCAL` |
| `CADE_CLOUD_BASE_URL` | `https://api.deepseek.com` | 云端 API 地址 |
| `CADE_CLOUD_API_KEY` | `sk-placeholder` | 云端 API Key |
| `CADE_CLOUD_MODEL` | `deepseek-chat` | 模型名称 |
| `CADE_LOCAL_BASE_URL` | `http://127.0.0.1:8080/v1` | 本地 LLM 地址 |
| `CADE_LOCAL_API_KEY` | `not-needed` | 本地 API Key |
| `CADE_LOCAL_MODEL` | `qwen3.5-9b-q8-local` | 本地模型标识 |
| `CADE_TEMPERATURE` | `0.2` | 采样温度 |
| `CADE_MAX_TOKENS` | `256` | 最大输出 token |
| `CADE_TIMEOUT` | `60` | HTTP 超时（秒） |

### 7.2 音频配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CADE_INPUT_DEVICE` | `default` | 输入设备名或索引 |
| `CADE_OUTPUT_DEVICE` | `default` | 输出设备名或索引 |
| `CADE_INPUT_SOURCE` | 空 | PipeWire 源名（如 `nx_remapped_out`） |
| `CADE_ECHO_SUPPRESS_MS` | `300` | TTS 播放期间抑制麦克风（ms） |
| `CADE_ECHO_SIMILARITY_THRESHOLD` | `0.75` | 回声相似度阈值 |
| `CADE_ECHO_SIMILARITY_WINDOW_SEC` | `2.5` | 回声检测窗口（秒） |
| `CADE_BARGE_IN_ENABLED` | `false` | 允许语音打断 TTS |

### 7.3 ASR 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CADE_ASR_PROVIDER` | `cpu` | 推理后端 |
| `CADE_ASR_MODEL_DIR` | （模型路径） | 主模型目录 |
| `CADE_ASR_MODEL_TYPE` | `streaming_nemotron` | 主模型类型 |
| `CADE_VAD_MODEL` | （模型路径） | VAD 模型路径 |
| `CADE_ASR_FALLBACK_MODEL_TYPE` | `streaming_zipformer` | fallback 模型类型 |
| `CADE_ASR_FALLBACK_MODEL_DIR` | （模型路径） | fallback 模型目录 |
| `CADE_ASR_REPLACEMENTS` | 空 | ASR 输出文本替换 |

### 7.4 TTS 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CADE_TTS_PROVIDER` | `cpu` | 推理后端 |
| `CADE_TTS_MODEL_DIR` | （模型路径） | 默认模型目录 |
| `CADE_TTS_ROUTER_ENABLED` | `true` | 启用多后端路由 |
| `CADE_TTS_DEFAULT_BACKEND` | `kokoro` | 默认后端 |
| `CADE_TTS_FAST_BACKEND` | `piper` | 快速后端 |
| `CADE_TTS_FALLBACK_BACKEND` | `vits` | 兜底后端 |
| `CADE_TTS_SPEED` | `1.05` | 语速倍率 |
| `CADE_TTS_CACHE_ENABLED` | `true` | 启用 WAV 缓存 |
| `CADE_TTS_CACHE_DIR` | `~/.cache/cade/tts` | 缓存目录 |
| `CADE_TTS_CHUNKING_ENABLED` | `true` | 长文本分句合成 |

### 7.5 点餐 FSM 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CADE_ORDER_BASE_DIR` | `data/orders` | 订单数据目录 |
| `CADE_ORDER_MENU_FILE` | `menu.yml` | 菜单文件路径 |
| `CADE_ORDER_INPUT_MODE` | `both` | 接受的输入通道：`primary`/`secondary`/`both` |
| `CADE_ORDER_ASK_PROMPT` | `What would you like to order?` | 开场白 |
| `CADE_ORDER_FINISH_TEMPLATE` | `OK I'll get {foods} for you` | 确认语模板 |
| `CADE_ORDER_DEDUP_WINDOW_SEC` | `1.5` | 去重窗口（秒） |
| `CADE_ORDER_RULE_PARSE_ENABLED` | `true` | 启用规则解析（快速路径） |
| `CADE_ORDER_LISTEN_MAX_RETRIES` | `5` | LISTEN 最大重试 |
| `CADE_ORDER_CHECK_MAX_RETRIES` | `5` | CHECK 最大重试 |
| `CADE_ORDER_EMPTY_INPUT_MAX` | `3` | 最大连续空输入 |
| `CADE_ORDER_MAX_QTY_PER_ITEM` | `9` | 单项最大数量 |
| `CADE_ORDER_MAX_TOTAL_QTY` | `20` | 单笔最大总量 |
| `CADE_ORDER_OUTBOX_RETRY_SEC` | `30` | Outbox 重试间隔（秒） |
| `CADE_ORDER_OUTBOX_MAX_ATTEMPTS` | `10` | Outbox 最大重试次数 |

### 7.6 ZeroMQ 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CADE_ZMQ_PUB_BIND` | `tcp://0.0.0.0:5555` | PUB 绑定地址 |
| `CADE_ZMQ_ROUTER_BIND` | `tcp://0.0.0.0:5556` | ROUTER 绑定地址 |
| `CADE_ZMQ_HEARTBEAT_SEC` | `2.0` | 心跳间隔（秒） |

---

## 8. 菜单配置

菜单通过 `menu.yml` 定义（默认路径为项目根目录）。

### 8.1 格式

```yaml
items:
  - id: water              # 唯一标识（内部主键）
    name: water             # 显示名称
    aliases: [water, bottle of water]  # ASR 匹配别名
    category: drink         # 分类（可选）
    available: true         # 是否可点
    max_qty: 10             # 单笔最大数量
    price: 1.50             # 单价（可选）
    modifiers:              # 规格选项（可选，当前仅定义未消费）
      size: [small, medium, large]
```

### 8.2 别名解析优先级

1. `CADE_ORDER_FOOD_ALIASES`（环境变量内联 JSON）
2. `CADE_ORDER_FOOD_ALIASES_FILE`（JSON 文件）
3. `menu.yml` 中的 `id` + `aliases`
4. 代码内置默认别名

### 8.3 修改菜单

编辑 `menu.yml` 后重启服务即可生效。新增条目需要包含 `id` 和 `name`。

```yaml
items:
  - id: sandwich
    name: sandwich
    aliases: [sandwich, club sandwich]
    category: food
    available: true
    max_qty: 5
    price: 6.00
```

---

## 9. 持久化与可靠性

### 9.1 订单目录结构

每个订单在 `CADE_ORDER_BASE_DIR` 下创建一个目录：

```
data/orders/
  12345/
    order_group.json          # 订单数据（原子写）
    events.jsonl              # 业务事件日志（追加写）
    outbox.jsonl              # outbox 状态日志（追加写）
    session_snapshot.json     # 会话快照（原子写）
    transition_journal.jsonl  # 提交步骤日志（追加写）
```

### 9.2 Outbox 可靠性模型

```
┌──────────┐    ┌───────────┐    publish     ┌───────────┐
│  pending │───▶│ published │──────────────▶│ 外部系统  │
└──────────┘    └───────────┘                └───────────┘
                     │                            │
                     │ retry                      │ ack
                     ▼                            ▼
               ┌───────────┐              ┌───────────┐
               │ published │              │ delivered │
               └───────────┘              └───────────┘
                     │
                     │ 超过 max_attempts
                     ▼
               ┌───────────┐
               │dead_letter│
               └───────────┘
```

**关键语义**：
- `published` ≠ `delivered`：本地发布成功仅标记 `published`，收到外部 ACK 后才标记 `delivered`
- `OutboxRetryWorker` 每 `outbox_retry_sec` 秒扫描一次，自动重发超时条目
- 超过 `outbox_max_attempts` 次重试后标记 `dead_letter`
- `delivered` 和 `dead_letter` 状态的条目不再参与重试

### 9.3 崩溃恢复

服务重启时自动扫描所有订单目录的 `session_snapshot.json`：

| 快照状态 | 恢复行为 |
|----------|----------|
| `finish_confirmed/committed` | 跳过（已完成） |
| `finish_confirmed/pending` | 补全确认：确保 `order_group.json` stage=confirmed、outbox 有 pending 条目 |
| 其他非 NOT_PERMITTED | 重置为 `NOT_PERMITTED`（未完成会话取消） |

`transition_journal.jsonl` 记录每个提交子步骤的 `started/committed` 状态，恢复时跳过已 `committed` 的步骤。

---

## 10. 测试与诊断

### 10.1 运行测试

```bash
# 全部测试
pytest tests/ -q

# 仅 ZMQ/可靠性测试
pytest tests/test_zmq_runtime.py tests/test_order_reliability.py -v

# 含覆盖率
pytest tests/ --cov=cade --cov-report=term-missing
```

### 10.2 语义解析评测

```bash
# 运行 InputPipeline 语义回放评测
python scripts/eval_audio_replay.py

# JSON 输出
python scripts/eval_audio_replay.py --json

# 指定 manifest
python scripts/eval_audio_replay.py --manifest evals/audio/manifest.json
```

评测样本定义在 `evals/audio/manifest.json`，每条包含 `input_text`、`fsm_state`（LISTEN/CHECK）和 `expected` 语义结果。

### 10.3 LLM 评测

```bash
# Bootstrap JSONL 用例评测
python scripts/eval_llm.py
```

### 10.4 诊断命令

```bash
# 运行全部检查
cade-order-doctor --json

# 写入报告文件
cade-order-doctor --json --write-report diagnostics/$(date +%Y%m%d_%H%M%S).json
```

检查项：

| 检查项 | 说明 |
|--------|------|
| `version_consistency` | `cade.__version__` 与 `pyproject.toml` 版本一致 |
| `tts_import` | TTS 引擎可导入 |
| `asr_import` | ASR 引擎可导入 |
| `llm_import` | LLM 客户端可导入 |
| `zmq` | ZMQ socket 可绑定 |
| `menu_file` | `menu.yml` 存在且可解析 |
| `menu_integrity` | 菜单 alias 无重复 |
| `order_dir` | 订单目录可读写 |
| `stale_outbox` | 无超过 1 小时的 pending/published 条目 |

### 10.5 FSM 状态图

```bash
# 基础状态图
cade-fsm-graph

# 含 FINISH 可靠性子流程
cade-fsm-graph --full

# Graphviz DOT 格式
cade-fsm-graph --format dot > fsm.dot
```

---

## 11. 目录结构

```
cade-voice-core/
├── pyproject.toml              # 项目定义与 CLI 入口
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量示例
├── menu.yml                    # 菜单定义
├── PROJECT_DESCRIPTION.md      # 项目技术描述
├── USAGE_GUIDE.md              # 本文件
│
├── cade/                       # 主包
│   ├── __init__.py             # 版本号（从包元数据读取）
│   ├── config.py               # 全局配置
│   ├── cli.py                  # CLI 入口（text-chat/voice-chat/bench）
│   ├── controller.py           # 机器人控制器（普通对话）
│   ├── voice/session.py        # 语音会话（普通对话）
│   │
│   ├── asr/                    # 语音识别
│   │   ├── engine.py           # ASR 引擎
│   │   └── text_norm.py        # 文本标准化
│   │
│   ├── tts/                    # 语音合成
│   │   ├── engine.py           # TTS 引擎（facade）
│   │   ├── router.py           # 后端路由
│   │   ├── cache.py            # WAV 缓存
│   │   ├── normalizer.py       # 文本标准化
│   │   ├── chunker.py          # 长文本分句
│   │   ├── playback.py         # 音频播放
│   │   └── backends/           # 各 TTS 后端
│   │
│   ├── brain/                  # LLM 与决策
│   │   ├── llm_client.py       # OpenAI-compatible 客户端
│   │   ├── prompts.py          # 系统 prompt
│   │   ├── schemas.py          # 结构化输出 schema
│   │   ├── structured_backend.py
│   │   └── schema_export.py
│   │
│   ├── body/                   # 机器人身体控制
│   │   ├── robot_interface.py
│   │   ├── robot.py
│   │   ├── safety.py
│   │   └── world_state.py
│   │
│   ├── agent/                  # Agent 图
│   │   ├── state.py
│   │   └── graph.py
│   │
│   └── fsm/                    # 点餐 FSM
│       ├── order_fsm.py        # FSM 核心
│       ├── config.py           # FSM 配置
│       ├── events.py           # 事件/命令模型
│       ├── states.py           # 状态枚举
│       ├── cli.py              # FSM CLI 入口
│       ├── graph_export.py     # 状态图导出
│       ├── full_pipeline_test.py    # 集成测试脚本
│       ├── full_pipeline_standalone.py # E2E 测试脚本
│       │
│       ├── parsing/            # 输入解析管线
│       │   ├── input_classifier.py  # 快速分类器
│       │   ├── order_parser.py      # 订单/确认解析
│       │   ├── menu_context.py      # 菜单上下文
│       │   └── pipeline.py          # 统一管线
│       │
│       ├── runtime/            # 运行时适配
│       │   ├── zmq_runtime.py       # ZeroMQ 运行时
│       │   └── voice_runtime.py     # 真实音频运行时
│       │
│       ├── storage/            # 持久化
│       │   ├── outbox.py            # Outbox 管理 + 重试工作线程
│       │   └── journal.py           # Transition journal
│       │
│       └── diagnostics/        # 诊断
│           └── checks.py            # 诊断检查
│
├── scripts/                    # 工具脚本
│   ├── eval_audio_replay.py    # 语义回放评测
│   ├── eval_llm.py             # LLM 评测
│   ├── audio_monitor.py        # 音频监控
│   └── check_llm_structured_capabilities.py
│
├── evals/                      # 评测数据
│   ├── bootstrap/*.jsonl       # LLM 评测用例
│   └── audio/manifest.json     # 语义回放用例
│
├── tests/                      # 测试文件
│   ├── test_order_fsm.py
│   ├── test_order_reliability.py
│   ├── test_zmq_runtime.py
│   ├── test_input_pipeline.py
│   ├── test_speaking_gate_echo.py
│   └── ...                     # 共 22 个测试文件
│
└── data/orders/                # 订单数据（运行时生成）
```

---

## 12. 常见问题

### Q: 如何使用本地 LLM？

参考 [3.3 使用本地 LLM 测试](#33-使用本地-llm-测试三终端启动) 的三终端启动方式。简要步骤：

1. 终端 1：启动 llama-server
2. 终端 2：`CADE_MODE=LOCAL cade-order-fsm`
3. 终端 3：`CADE_MODE=LOCAL cade-order-test`

也可以在 `.env` 中持久化设置 `CADE_MODE=LOCAL`，这样就不用每次在命令前加前缀。

### Q: 如何只测试点餐流程不接音频？

```bash
# 启动 headless FSM
cade-order-fsm

# 另一个终端运行集成测试
cade-order-test
```

`cade-order-test` 通过 ZMQ 发送文本模拟完整点餐流程。

### Q: 如何查看订单数据？

订单存储在 `data/orders/` 目录下，每个订单一个子目录：
```bash
# 查看所有订单
ls data/orders/

# 查看某个订单详情
cat data/orders/00001/order_group.json | python -m json.tool

# 查看 outbox 状态
cat data/orders/00001/outbox.jsonl
```

### Q: 如何排查未送达的订单？

```bash
# 运行诊断
cade-order-doctor --json

# 或通过 ZMQ 命令查询
# 发送 outbox.undelivered 命令获取未送达列表
# 发送 outbox.retry 命令触发手动重试
```

### Q: 如何添加新菜单项？

编辑 `menu.yml`，添加新条目后重启服务：

```yaml
items:
  # ... 现有条目 ...
  - id: pizza
    name: pizza
    aliases: [pizza, pepperoni pizza]
    category: food
    available: true
    max_qty: 5
    price: 8.00
```

### Q: 如何切换 TTS 语音质量？

```bash
# 默认使用 kokoro（高质量）
# 切换到 piper（低延迟）
CADE_TTS_DEFAULT_BACKEND=piper

# 完全关闭 TTS 缓存（调试用）
CADE_TTS_CACHE_ENABLED=false
```

### Q: 如何调试 ASR 识别问题？

```bash
# 设置详细日志
CADE_LOG_LEVEL=DEBUG

# 查看实时 ASR 输出
CADE_LOG_LEVEL=DEBUG cade-order-voice 2>&1 | grep asr
```

### Q: 订单确认后没有收到 order.confirmed 事件？

1. 检查 SUB socket 是否在 PUB 绑定**之后**连接（ZeroMQ PUB/SUB 有慢启动问题，需要先连后发）
2. 检查 `outbox.jsonl` 中是否有 `published` 条目
3. 使用 `cade-order-doctor --json` 检查 stale outbox
4. 手动触发重试：发送 `outbox.retry` 命令

### Q: 如何清除测试订单数据？

```bash
rm -rf data/orders/[0-9]*
```

这不会影响运行中的服务。如果 FSM 正在运行，重启后新的订单目录会自动创建。
