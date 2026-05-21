# ROS-free 点餐 sub-FSM 实现计划

**Summary**
- 将旧 `sub-FSM` 中的 ROS bridge 还原为纯业务 FSM，保留点餐逻辑：`NOT_PERMITTED -> PERMITTED -> ASK -> LISTEN -> REPEAT -> CHECK -> FINISH -> NOT_PERMITTED`。
- 旧 ROS topic 全部替换为 ZeroMQ 跨机器通信：控制面用 `ROUTER/DEALER` 收命令并回 ack，事件面用 `PUB/SUB` 推送状态、TTS、订单确认。
- 当前 `cade.brain.schemas`、`LLMClient.get_order_action/get_order_repeat_speak/get_order_check_decision` 已具备点餐 schema 和 LLM helper，可直接复用；主要新增 FSM runtime、ZeroMQ adapter、配置、测试。

**Key Changes**
- 新增纯逻辑模块 `cade/fsm`：实现 `OrderSubFSM`、状态枚举、配置模型、事件模型。模块内禁止 `rospy/std_msgs/roslaunch`，只依赖注入的 `llm_client`、`robot`、`tts_sink`、`event_sink`、`order_storage`、`order_id_provider`。
- FSM 行为保持旧逻辑：
  - `serving_state=PAUSED_ORDERING` 开启会话；离开该状态立即 reset。
  - 仅 `LISTEN/CHECK` 接收用户文本；并发输入被忽略；双通道重复文本按时间窗口去重。
  - `LISTEN` 解析订单、合并别名和数量，失败则 TTS retry 并留在 `LISTEN`。
  - `REPEAT` 生成确认语；LLM 失败用确定性 fallback。
  - `CHECK` 确认正确则发布订单确认并结束；带修改则更新订单回 `REPEAT`；无有效修改则追问并回 `LISTEN`。
- 新增 ZeroMQ runtime：
  - `PUB tcp://0.0.0.0:5555` 发布 `order.state`、`tts.request`、`order.confirmed`、`order.metrics`、`order.error`。
  - `ROUTER tcp://0.0.0.0:5556` 接收 `serving_state.update`、`user_text.primary`、`user_text.secondary`、`order_id.propose`、`snapshot.get`、`health.get`、`session.cancel`。
  - 消息统一 JSON envelope：`v`、`type`、`id`、`ts`、`source`、`session_id`、`payload`；PUB 用 multipart `[topic, json]`。
- 状态回放：ZeroMQ 不自带 ROS latch，因此 voice core 内存保存最新 `OrderStateEvent` 和订单 snapshot；新客户端先发 `snapshot.get`，再订阅 PUB；运行时每 2s 发送 heartbeat。
- ROS topic 映射：
  - `/asr` -> `user_text.primary`
  - `/person_following/pause_reply_text` -> `user_text.secondary`
  - `/tts` -> `tts.request` 事件，同时本地可直接调用 `TTSEngine`
  - `/person_following/serving_customer_state` -> `serving_state.update`
  - `/person_following/order_confirm_json` -> `order.confirmed`
  - `/mhrc/order_subfsm_state` -> `order.state` + `snapshot.get`
  - `/mhrc/random_order_id` -> 可选 `order_id.propose`
- 配置新增 `CADE_ORDER_*` 与 `CADE_ZMQ_*`：订单目录默认 `data/orders`，food aliases 支持 JSON/env 或文件，order id 优先使用外部提供值，缺失时本地生成唯一 5 位 id，避免旧逻辑无限等待。
- 新增 CLI：`cade-order-fsm` 运行 headless ZeroMQ FSM；`cade-order-voice` 运行本地 ASR/TTS + ZeroMQ 集成。现有 `cade-voice-chat` 保持普通语音对话用途。

**Public Interfaces**
- Python API：`handle_serving_state(payload)`、`handle_user_text(text, source)`、`handle_order_id(candidate)`、`snapshot()`、`cancel(reason)`。
- `order.state` payload 保留旧字段：`timestamp`、`state`、`reason`、`serving_state`、`order`、`order_id`、`order_dir`、`session_id`。
- `order.confirmed` payload 对齐旧确认 JSON：`order`、`foods`、`foods_with_qty`、`recognized_text`、`check_text`、`order_id`、`order_dir`，并透传 `customer_id/customer_no/folder/customer_folder`。

**Test Plan**
- 纯 FSM pytest：happy path、wrong + fix_order、wrong 无修改回 LISTEN、双通道去重、非 `PAUSED_ORDERING` 忽略输入、会话中途 reset、LLM 失败 retry、TTS 失败事件、重复 order id、本地 order id fallback。
- ZeroMQ pytest：临时端口启动 runtime，验证 command ack、PUB 事件、`snapshot.get` 状态回放、多客户端订阅、断线重连后恢复。
- 回归检查：`rg "rospy|std_msgs|roslaunch" cade` 必须无生产引用；旧 `scripts/test_order_subfsm_stability.py` 场景迁移为无 ROS pytest。
- 保留现有 `tests/test_llm_client.py`、`tests/test_controller.py`、`tests/test_voice_session.py` 全量通过。

**Assumptions**
- 默认采用用户选择的“跨机器网络 + ZeroMQ”；新增依赖 `pyzmq`，不引入 ROS 或 ROS bridge。
- 每个 `cade-voice-core` 实例同一时刻只处理一个点餐会话。
- ZeroMQ 事件流是实时 best-effort；权威恢复靠 `snapshot.get`，若未来需要离线持久投递，再升级到 NATS/JetStream。
- voice core 只负责语音点餐、TTS 请求和订单确认，不直接控制 person tracker 或导航行为。
