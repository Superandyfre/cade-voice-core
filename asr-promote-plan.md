# ASR 升级 Plan：Zipformer 20M → Nemotron 0.6B

## 分析：提案 vs 项目现状

### 提案中合理的部分

1. **升级方向正确**：Nemotron 0.6B INT8 via sherpa-onnx 确实比 20M Zipformer 更适合作为主 ASR
2. **保留 Zipformer fallback**：完全同意
3. **文本规整**：Nemotron 输出带大小写和标点，当前 FSM 的 `_normalize_text_token()` 会做 lowercase + 去标点，但这一层来得太晚且粗糙
4. **endpoint 策略**：继续使用模型内置 endpoint，不引入外部 VAD — 和当前 streaming_zipformer 的逻辑一致
5. **分阶段实施**：先 wav 测试 → 接入麦克风 → 切默认 — 步骤合理

### 提案中过度设计的部分

| 提案建议 | 实际问题 |
|---|---|
| `StreamingRecognizerProtocol` + `NemotronStreamingRecognizer` + `ZipformerStreamingRecognizer` 三个独立 adapter 类 | Zipformer 和 Nemotron 在 sherpa-onnx 中都是 `OnlineRecognizer.from_transducer()`，API 完全一样。分开写两个类只是多了一层无用间接 |
| `StreamingASRConfig` pydantic model | 现有 `Config` 是平铺 env-var 风格，FSM config 才用 pydantic。硬塞一个 Pydantic ASR config 是风格不一致 |
| `ASREvent` 系统（partial/final/error + timestamp + confidence） | 当前整个管线（VoiceSession、E2E pipeline、FSM）只消费 final text，不处理 partial。加 ASREvent 需要同时改造 VoiceSession、E2E、FSM 的所有消费端，scope 远超"换模型" |
| `cade/asr/streaming_base.py`, `sherpa_nemotron.py`, `sherpa_zipformer.py`, `fallback.py`, `config.py` 五个新文件 | 加一个新 `model_type` 分支只需要在 `_create_recognizer()` 里加一个 `elif`，5 个文件是过早抽象 |
| 三线程 `AudioCapture → ASRWorker → FSM` | 当前代码已经是这个结构：ASR callback 在独立线程，FSM `handle_user_text` 在另一个线程 |

### 核心判断

**实际最小改动**：在 `ASREngine._create_recognizer()` 加一个 `streaming_nemotron` 分支（另一个 `from_transducer` 调用），加上 fallback 初始化逻辑和文本规整。代码改动量约 100 行。

**但为了后续可维护性**，值得适度做：fallback 机制、文本规整层、benchmark 脚本。不值得做：Protocol 抽象、ASREvent 系统、独立 adapter 类。

---

## 具体 Plan

### 阶段 0：验证 sherpa-onnx 对 Nemotron 的支持

> 在写任何代码之前，先确认 sherpa-onnx 版本和 Nemotron 模型的兼容性

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 0.1 | 检查当前 `sherpa-onnx` 版本，确认是否支持 `OnlineRecognizer.from_transducer` 加载 Nemotron 模型 | 终端命令 |
| 0.2 | 如果版本不够，升级 `sherpa-onnx` 到支持 Nemotron 的版本 | `requirements.txt` |
| 0.3 | 用 sherpa-onnx 官方 Python 示例脚本，手动加载 Nemotron INT8 模型，对测试 wav 做一次离线解码，确认 encoder/decoder/joiner/tokens 文件名和参数 | 终端命令，不改代码 |

### 阶段 1：下载模型 + wav 文件验证

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 1.1 | 在 `download_voice_models.sh` 中新增 Nemotron INT8 模型下载段（下载 `sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25` 或最新可用版本） | `scripts/download_voice_models.sh` |
| 1.2 | 运行下载脚本，确认模型文件完整 | 终端 |
| 1.3 | 写一个一次性 `scripts/test_nemotron_wav.py` 脚本，直接调用 sherpa-onnx API 加载 Nemotron 模型，对测试 wav 做流式解码，打印识别结果、耗时、RTF | 新建 `scripts/test_nemotron_wav.py`（临时脚本，不入库） |
| 1.4 | 对比同一 wav 在 Zipformer 20M 和 Nemotron 0.6B 下的识别质量、耗时 | 终端 |
| 1.5 | 验收标准：Nemotron 在 CPU 上 RTF < 0.5，识别质量肉眼可见更好 | — |

### 阶段 2：代码接入 — 在 ASREngine 中新增 `streaming_nemotron`

> 这是核心改动，改动量最小但完成主路径

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 2.1 | 在 `ASREngine._create_recognizer()` 中新增 `streaming_nemotron` 分支。和 `streaming_zipformer` 一样调用 `sherpa_onnx.OnlineRecognizer.from_transducer()`，区别是模型文件名 pattern 和 endpoint 参数 | `cade/asr/engine.py` L91-L177 |
| 2.2 | 在 `Config` 中新增 Nemotron 相关环境变量：`CADE_NEMOTRON_MODEL_DIR`、`CADE_NEMOTRON_NUM_THREADS`，并修改默认 `ASR_MODEL_TYPE` 和 `ASR_MODEL_DIR` 指向 Nemotron（但**不急着切默认**，先保持 `streaming_zipformer` 不变） | `cade/config.py` |
| 2.3 | 用 `cade-bench --asr-smoke --model-type streaming_nemotron --wav xxx.wav` 测试，确认走通新分支 | `cade/cli.py`（需给 `_bench_asr` 加 `--model-type` 参数） |
| 2.4 | 确认流式麦克风路径也能走通 Nemotron：`_start_listening_streaming()` 和 `_start_listening_pulse_streaming()` 不依赖模型类型，只依赖 `self._is_streaming`，所以天然兼容 | 无需改代码，手动测试 |

### 阶段 3：Fallback 机制

> 不搞复杂的三层架构，只做启动时 fallback

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 3.1 | 修改 `ASREngine.__init__`：加载主模型失败时，自动尝试加载 fallback 模型。在 Config 中新增 `CADE_ASR_FALLBACK_MODEL_TYPE` 和 `CADE_ASR_FALLBACK_MODEL_DIR` | `cade/asr/engine.py` L38-L86, `cade/config.py` |
| 3.2 | 记录当前实际使用的模型名称（`self.active_model_name`），在 logger.info 和 metrics 中体现 | `cade/asr/engine.py` |
| 3.3 | 测试：删除 Nemotron 模型文件，确认自动降级到 Zipformer 并正常工作 | 手动测试 |

### 阶段 4：文本规整层

> Nemotron 输出 "I would like a cheeseburger and a Coke."，FSM 需要规整后版本

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 4.1 | 新建 `cade/asr/text_norm.py`，实现 `normalize_asr_text(text: str) -> str`：lowercase、去末尾标点、常见 ASR 纠错替换（可配置） | 新建 `cade/asr/text_norm.py` |
| 4.2 | 在 `ASREngine` 的所有 callback 调用点（streaming offline/streaming/pulse offline/pulse streaming 四个路径），对输出文本做 normalize，同时保留 raw_text 供日志 | `cade/asr/engine.py` |
| 4.3 | 在 `Config` 中新增 `CADE_ASR_REPLACEMENTS` 环境变量支持（JSON string 或文件路径），加载到替换表 | `cade/config.py` |
| 4.4 | 测试：确认 "I would like a cheeseburger and a Coke." → "i would like a cheeseburger and a coke" | 单元测试 |

### 阶段 5：性能指标（轻量版）

> 不搞复杂的事件系统，只做 logging 级别的指标记录

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 5.1 | 在 streaming 监听循环中，记录每次 decode 耗时、endpoint 到 final 的延迟、RTF，以 structured log 输出 | `cade/asr/engine.py` |
| 5.2 | 统计空 final 次数、连续空 final 警告 | `cade/asr/engine.py` |
| 5.3 | 在 `ASREngine` 上暴露 `get_metrics() -> dict` 方法，供外部查询 | `cade/asr/engine.py` |

### 阶段 6：麦克风实时测试

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 6.1 | 安静环境测试：正常语速说点餐句子，记录 partial 稳定性、final 完整性、endpoint 延迟 | 手动 |
| 6.2 | 背景噪声测试：播放音乐 + 说话 | 手动 |
| 6.3 | 远场测试：距离麦克风 1-2 米说话 | 手动 |
| 6.4 | 快速说话 + 中途停顿：测试 endpoint 不会过早截断 | 手动 |
| 6.5 | 调参：根据测试结果调整 `rule1_min_trailing_silence`、`rule2_min_trailing_silence` 等参数 | `cade/asr/engine.py` |

### 阶段 7：切换默认模型

| 步骤 | 做什么 | 涉及文件 |
|---|---|---|
| 7.1 | 将 `Config.ASR_MODEL_TYPE` 默认值改为 `streaming_nemotron`，`Config.ASR_MODEL_DIR` 指向 Nemotron 目录 | `cade/config.py` |
| 7.2 | 运行全部现有测试，确保不回归 | 终端 |
| 7.3 | 运行 E2E 真机测试（`full_pipeline_standalone.py`），确认完整点餐流程通过 | 手动 |

### 阶段 8（可选，后续）：partial/final 分离

> 这不属于"换模型"，而是功能增强。留到 Nemotron 稳定跑起来后再做。

---

## 涉及文件总结

| 文件 | 改动类型 |
|---|---|
| `cade/asr/engine.py` | 新增 `streaming_nemotron` 分支 + fallback 逻辑 + 指标记录 + callback 处文本规整 |
| `cade/asr/text_norm.py` | **新建**，文本规整函数 |
| `cade/config.py` | 新增 Nemotron + fallback 环境变量，最终切换默认 |
| `scripts/download_voice_models.sh` | 新增 Nemotron 模型下载段 |
| `cade/cli.py` | `_bench_asr` 加 `--model-type` 参数 |
| `tests/test_asr_text_norm.py` | **新建**，文本规整单元测试 |

**不新建的文件**（提案建议但实际不需要）：`streaming_base.py`、`sherpa_nemotron.py`、`sherpa_zipformer.py`、`fallback.py`、`asr/config.py`。
