# vidsum

长视频转写 + 滚动摘要。输入一个本地视频或直链，输出带时间戳的完整转写稿和一份持续增长的滚动摘要。

核心约束是**送进模型的上下文大小与视频总时长无关**：视频按静音切段后逐段转写，
每段转写立即并入一份滚动摘要，因此三小时的视频和十分钟的视频单轮开销相同，
内存占用也基本不随时长增长。

## 安装

```powershell
# 音频处理二选一：装 PyAV（自带 FFmpeg）或装系统 ffmpeg
E:\anaconda\envs\testenv\python.exe -m pip install av

# 转写后端按需选择
E:\anaconda\envs\testenv\python.exe -m pip install sherpa-onnx   # 本地 paraformer
E:\anaconda\envs\testenv\python.exe -m pip install faster-whisper # 本地 whisper
```

总结一步只需要一个 OpenAI 兼容的 chat/completions 接口，用 `--llm-api-key` 或环境变量
`VIDSUM_LLM_API_KEY` 提供；不配置则只输出转写，不生成摘要。

`av` 装好后就不需要系统 ffmpeg 了（wheel 自带整套 FFmpeg 动态库）。两条音频路输出一致，
用 `--audio-backend ffmpeg|pyav|auto` 指定，默认 `auto`：有 ffmpeg 就用，否则退回 PyAV。

## 快速开始

```powershell
# 本地视频 + 本地模型
python -m vidsum lecture.mp4 --asr faster-whisper --model-size medium --llm-api-key sk-xxx

# 云端转写 + 云端总结（零本地依赖）
python -m vidsum lecture.mp4 --asr api --asr-model whisper-large-v3 --asr-api-key sk-xxx --llm-api-key sk-xxx

# 先验证流程与分段计划，不转写
python -m vidsum lecture.mp4 --dry-run
```

输出默认写在 `./vidsum-out/<视频名>/`：

```text
vidsum-out/<视频名>/
  audio.mp3            整段音频缓存（转写完默认删除，--keep-audio 保留）
  plan.json            分段计划
  chunks/NNN.json      每段转写结果（断点续跑的凭据）
  transcript.md        完整转写稿，边转写边增长
  transcript.txt       同上，纯文本
  frames/NNN/SSSS.SSS.jpg  每段关键帧，文件名是帧在视频中的秒数
  frames.json          抽帧清单（段号 -> 帧路径），断点续跑时复用
  summary.md           滚动摘要，边总结边增长
  summary_state.json   摘要进度与分段小标题
  final.md             收尾后的结构化终稿
```

任一步骤中断后重跑同一命令即可续上：已转写的段落、已完成的摘要段都会跳过。

## Web 控制台

不想记参数就用浏览器操作：

```powershell
E:\anaconda\envs\testenv\python.exe -m vidsum.webapp --port 8765
```

打开 http://127.0.0.1:8765/ 即可（只监听本机回环，换端口用 `--port`）。左侧填表单，
右侧实时观察：日志滚动输出，转写稿和摘要随处理进度一起增长，文件页列出全部产物。
任何时候只跑一个任务；运行中点"停止"立即生效——每个任务都跑在独立子进程里，
停止就是直接终止那个进程，即便卡在模型加载这类不吐日志的阶段也能杀掉，不会把服务
本身拖住。已完成的段落都在磁盘上，重跑同一任务接着来。控制台启动时还会顺手清理
上一次遗留的子进程，避免旧任务偷偷占着显存。

控制台就是同一套 `pipeline.run` 的壳（只是包在子进程里跑）：表单项与命令行参数一一
对应，产物结构、断点续跑行为完全一致，输出默认落在项目根目录的 `vidsum-out/`
（表单里可改）。子进程的 stdout 按 UTF-8 逐行流回界面当实时日志，中文日志不受系统
locale 影响。转写方式在顶部做成「本地 / 云端 API」两条平行轨道，点一下即切换，各自
只显示自己的配置：本地轨道选 sherpa 或 faster-whisper（并带模型目录、推理线程、
Whisper 模型档位），云端轨道填接口地址 / Key / 模型。两条轨道互不干扰，来回切换时
对方已填的内容都会保留。输入框填 http(s)/m3u8 直链时会多出 User-Agent / Referer /
Cookie 请求头。表单选择和最近输入存在浏览器本地，深浅主题跟随系统、也可手动切换。

## 打包成 exe

用 PyInstaller 打成 onedir 形态的 `dist\vidsum\vidsum.exe`：

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

默认用 `E:\anaconda\envs\testenv\python.exe`（装了全部可选依赖），换解释器加
`-Python 路径`。产物是 `dist\vidsum\` 整个目录：`vidsum.exe` 加一个 `_internal`
依赖目录，一起拷到没装 Python 的机器上就能跑，双击即开 Web 控制台，命令行参数
也全盘支持。分发时把这个目录压成 zip 即可。

两个构建决策值得说明：

- **onedir 而不是 onefile**。onefile 的 bootloader 启动时要把整个包自解压到
  `%TEMP%\_MEIxxxx`，火绒这类行为防护会拦掉“未知 exe 往磁盘释放 DLL”的动作，
  exe 根本起不来；onedir 的 DLL 构建期就落在 exe 旁边，运行期一个字节都不往
  temp 写，启动也省掉每次数十秒的自解压延迟。
- **console=True 不能关**：实时日志走 stdout，关了只剩空窗口。

模型（1.5GB 起）不进 bundle，放 exe 旁边即可被识别：

```powershell
# 把项目根的 models\whisper 硬链接到 exe 旁边：秒完成、不占第二份空间
# （必须是真目录——huggingface_hub 的缓存扫描不认 junction/符号链接，
# 挂了会被当成“本地没缓存”转而去连 Hub）
powershell -ExecutionPolicy Bypass -File packaging\link_models.ps1
```

PyInstaller 每次构建都会清空 `dist\vidsum`，所以重建 exe 后跑一次上面的脚本
就能恢复模型目录，不必手动拷 1.5GB。输出默认落在 exe 旁的 `vidsum-out\`。

### 端到端验证

`packaging\e2e_check.py` 驱动真实 exe 跑全链路（先起服务，再跑脚本）：

```powershell
.\dist\vidsum\vidsum.exe --web --port 8799
E:\anaconda\envs\testenv\python.exe packaging\e2e_check.py
```

覆盖：前端页面从 bundle 里出来、冻结态子进程 spawn（exe 冻结后自调用
`--job-runner`）、中文日志经管道不变乱码、真实音频转写（GPU 命中、模型从
exe 相邻目录加载、产出 transcript.md）。脚本会顺手确认 exe 旁边有
`models\whisper`，没有就自动硬链接过去。

## 工作方式

1. **抽音频**：从文件或 URL 解复用出 16kHz 单声道音轨。视频文件本身从不落盘，
   网络流（mp4/m3u8）也由音频后端自行拉取。
2. **静音切分**：扫描静音区间，分段时把切点吸附到最接近目标位置的静音中点，
   避免把一句话拦腰截断。段长按目标时长均分，让每段送给模型的文本量大致相等。
3. **抽关键帧**：按分段计划在每段内均匀取几个时间点（避开首尾各 1 秒）抽 JPEG 帧，
   宽默认 480px，落在 `frames/NNN/<秒数>.jpg`，清单记在 `frames.json`。帧文件名即时间戳，
   续跑时清单命中就直接复用。只有 ffmpeg 音频后端能抽帧，PyAV 后端自动跳过。
4. **并行转写**：各段独立送 ASR，`--workers` 控制并发。每段结果立刻落盘，因此中途崩溃不丢进度。
5. **滚动总结**：维护一份滚动摘要，每来一段就 `新摘要 = LLM(旧摘要 + 本段文本)`。
   摘要过长时先压缩再更新（层级式合并），上下文因此有上界。
   加 `--vision` 时把该段关键帧一并喂给多模态 LLM（需要模型支持图片输入），
   幻灯片标题、板书、图表这些只存在于画面里的信息也能进摘要。
6. **收尾**：再用一轮调用把滚动摘要整理成结构化终稿，预算是 `max(--llm-max-tokens, 3072)`。
   推理模型（step-5、o1、deepseek-r1 等）会把 token 先花在"思考"上，偶尔一个 token 都不剩：
   返回空正文且 `finish_reason=length`。这时 vidsum 会自动把预算加倍重试（最多两次，上限
   32768）；仍然空就报错退出，错误信息里会写明用过的预算。调大 `--llm-max-tokens`
   （网页控制台里是「token 上限」）重跑即可——转写和滚动摘要都有缓存，只会重跑收尾这一轮。

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--chunk-minutes N` | 每段目标时长，默认 10 分钟 |
| `--workers N` | 并发转写数，默认 `min(4, CPU/2)` |
| `--asr` | `faster-whisper`（本地）/ `sherpa`（本地 paraformer）/ `api`（OpenAI 兼容转写接口）/ `dummy`（联调） |
| `--model-size` | faster-whisper 模型大小，默认 `medium` |
| `--whisper-download-root` | faster-whisper 模型的下载/缓存目录，默认项目目录下的 `models/whisper`；想放别处就指过来，如 `E:\whisper-models` |
| `--audio-backend` | `auto`（默认）/ `ffmpeg` / `pyav` |
| `--sherpa-model-dir` | sherpa 模型目录，含 `asr/` `vad/` `punc/` 子目录；也可用环境变量 `VIDSUM_SHERPA_MODEL_DIR` |
| `--sherpa-threads` | sherpa 推理线程数，默认 2 |
| `--no-punctuation` | 关闭 sherpa 的标点恢复 |
| `--language` | 音频语言，默认 `zh` |
| `--no-silence-split` | 关闭静音对齐，按固定时长硬切 |
| `--force` | 忽略缓存从头重跑 |
| `--dry-run` | 只抽音频并打印分段计划 |
| `--keep-audio` / `--keep-chunks` | 保留中间音频文件 |
| `--no-frames` | 关闭抽帧（默认每段抽关键帧并插图进终稿附录） |
| `--frames-per-segment N` | 每段关键帧数，默认 2 |
| `--frame-width N` | 关键帧宽度（像素，等比缩放），默认 480 |
| `--vision` | 总结时把关键帧喂给多模态 LLM（需模型支持图片输入） |
| `--llm-max-tokens N` | 单次总结输出的 token 上限，默认 2048（收尾一轮至少 3072） |

网络源需要鉴权时，用 `--user-agent` / `--referer` / `--cookie` 传请求头，音频后端会带上它们。

直接贴 B 站视频页链接（`bilibili.com/video/BV...` 或 `.../av...`）也可以：工具会自动把
它解析成 DASH 音频流直链并带上 `Referer`，不必自己去找播放地址。这一步要能访问
bilibili.com；要高音质或遇到需要登录/大会员的片子时，把 Cookie 填进 `--cookie`
（网页控制台里输入链接后会出现"网络源请求头"一栏）。

## sherpa 后端

sherpa-onnx 一条路走的是「解码 → VAD 切段 → 逐段 paraformer → 恢复标点」：

- **VAD 切段不能省**。paraformer 一次喂太长的音频输出会明显劣化，而 VAD 切出来的段正好落在
  语句之间。喂给 VAD 的块也要小：实测 1 秒以上的粒度会把整段语音并成一句，默认用 1024 样本
  （64ms）。
- **起流预热**。silero 的 LSTM 状态从零开始，音频最开头约 1 秒常被判成非语音，视频第一句就
  这么整句丢失。喂 VAD 前先垫约 0.5 秒静音让模型进入稳态，检出后再把这段从时间戳里减掉，
  首句就能从接近 0 的位置回来（`SherpaBackend(warmup_seconds=...)` 可调，设 0 关闭）。
- **模型可以复用**。`--sherpa-model-dir` 不指定时按下面顺序搜索：环境变量
  `VIDSUM_SHERPA_MODEL_DIR`、项目内 `models/sherpa`、`%LOCALAPPDATA%\vidsum\models\sherpa`，
  以及 VideoToNo 的缓存目录。目录里只要有 `asr/model.int8.onnx` 和 `asr/tokens.txt` 就算命中，
  `vad/silero_vad.onnx` 必需，`punc/model.onnx` 可选（缺了就跳过标点）。
- **串行处理**。VAD 对象带内部状态，多线程共用会把音视频错切到一起，所以后端内部用一把锁把
  `transcribe` 串起来。单段推理本身已由 `--sherpa-threads` 做线程级并行，`--workers` 对它无效。
- **标点模型 294MB**，加载和推理都比转写模型慢；只想要转写稿时加 `--no-punctuation`。

## 测试

```powershell
python -m unittest discover -s tests -t .
```

275 个用例，全部不依赖 ffmpeg、模型和网络：

- `tests/test_commands.py` 直接断言 ffmpeg 命令行的构造（参数顺序、取值、`-headers` 位置）；
- `tests/test_pipeline.py::RealMediaLayerTest` 只替换最底层的 `media._run`，
  上面的探测、抽音频、静音扫描、切片全部跑真实代码路径；
- `tests/test_audio_av.py` 用假 av 模块 + 真 numpy 覆盖 PyAV 的抽取、切片与静音检测，
  重点守住重采样器要排空尾部、`encode(None)` 只能调一次这两条；
- `tests/test_asr_sherpa.py` 的假 VAD 故意在 `pop()` 时清空 `front` 的 samples，
  顺序写反（先 pop 再读）立刻会暴露；
- `tests/test_webapp.py` 用裸 socket 驱动控制台的 HTTP 接口（含目录穿越防护），
  mock 子进程层验证任务生命周期、单任务保护与"停止即终止"；另有一个真起子进程的
  编码回归用例，守住中文日志在 GBK locale 下也不变乱码；
- `tests/test_frames.py` 覆盖抽帧规划（均分取点、避首尾、短段不取）、帧路径与
  时间戳往返、`frames.json` 清单读写与 base64 data URI；
- 其余用例覆盖分段规划、边界去重、LLM 重试、摘要解析与断点续跑；
- `tests/test_bili.py` 打桩 `urlopen` 覆盖 B 站解析（URL 判定、view 接口主路径与页面回退、WBI 签名、
  `__INITIAL_STATE__` 提取、音频流挑选、整条编排），全程不联网。

## 设计取舍

- **Python 而不是 C++**：抽音频是 ffmpeg 子进程，转写是 ctranslate2（C++）或云接口，
  总结是 HTTP 调用，瓶颈都不在编排代码上。换语言省不下时间，却要付构建和生态的代价。
- **共享模型实例**：faster-whisper 的底层 ctranslate2 推理会释放 GIL 且模型可多线程共用，
  所以并发转写不会让权重内存翻倍。若某个后端不是线程安全的，用 `--workers 1` 退化。
- **流式不等于省内存**：转写文本三小时也只有几十 KB。分块真正的收益是 LLM 上下文有上界、
  ASR 长音频不漂移、以及崩溃后能续跑。
- **任务跑在子进程里**：Web 控制台的每个任务都是独立的 `python -m vidsum._job_runner`
  子进程（exe 冻结后没有解释器可用，改为自调用 `vidsum.exe --job-runner`，
  由 `vidsum.entry` 统一分发）。停止即终止进程，卡在 CUDA 初始化里的任务也杀得掉，
  任务崩溃也不会带垮服务；
  日志经 stdout 管道流回界面（强制 UTF-8，中文不乱码），断点续跑照旧依赖磁盘缓存。
