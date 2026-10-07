# 前期准备报告

状态：**准备完成，尚未开展正式训练。** 执行日期：2026-09-14（Australia/Sydney）。目录：项目根目录。

## 实际完成

已建立独立项目、Python venv、集中配置、100 条虚构投诉数据、组级拆分、数据检查、真实官方 Tokenizer 预处理、LoRA 模块/参数检查、严格评测、20 项 CPU 测试以及 GPU 最小前后向验证。项目没有正式训练入口，训练参数仅作为下一阶段的预览。

原模型只对 train 的 c02-1 做了一条推理接口检查。没有 optimizer.step、正式训练循环、训练后权重保存、完整测试集推理或前后效果比较。没有调用付费 API、使用云 GPU、修改系统驱动/CUDA/Python、上传文件或创建远程仓库。

主要文件和作用、可复制 PowerShell 命令见 [README](../README.md)；教学解释见 [learning_guide.md](learning_guide.md)。

## 实际环境与版本

| 项目 | 实测结果 |
| --- | --- |
| 当前执行位置 | 本机 Windows 11，10.0.26200，AMD64；PowerShell；不是 WSL、Linux 容器或远程 GPU |
| 起始工作区 / Git | 项目父目录，检查时为空；git status/rev-parse 提示不是 Git 仓库。未初始化 Git，因此不存在本项目 commit |
| 项目 Python | .venv\Scripts\python.exe，Python 3.12.9，64 位；原系统 Python 未安装 torch |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，可由当前 Agent 启动的本机 Python 实际访问 |
| 总显存 / 空闲显存 | nvidia-smi 总量 8151 MiB；首次空闲 6752 MiB，环境记录时 6788 MiB；空闲值随桌面程序变化 |
| 驱动 / 计算能力 | 591.91 / (12, 0)；wheel 包含 sm_120 |
| CUDA | torch 自带 runtime 12.8；torch.cuda.is_available()=True；FP32、BF16 的 4×4 CUDA 矩阵乘法均实际通过 |
| 系统 CUDA 编译器 | PATH 上未找到 nvcc，WinError 2；这不等于全面证明系统没有 toolkit。本项目预编译 wheel 与 GPU smoke 不依赖 nvcc |
| WSL | Ubuntu-24.04 与 docker-desktop，WSL2，检查时均 Stopped。本阶段没有启动、安装或验证 WSL 内环境 |
| 磁盘 | 起始 D 盘空闲 290,452,447,232 bytes（约 270.5 GiB）；安装依赖后环境记录为 282,370,383,872 bytes（约 263.0 GiB），模型下载前采样 |
| 官方来源网络 | GitHub API、raw.githubusercontent.com、Hugging Face API/固定 revision 下载、PyTorch 官方 wheel 源和 PyPI 均实际成功 |

已安装并验证的直接依赖：

| 依赖 | 版本 |
| --- | --- |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.57.6 |
| PEFT | 0.18.1 |
| Datasets | 4.5.0 |
| Accelerate | 1.12.0 |
| PyYAML | 6.0.3 |
| Tokenizers | 0.22.2 |
| Hugging Face Hub | 0.36.2 |
| Safetensors | 0.8.0 |
| NumPy | 2.5.3 |

完整实际版本见 [requirements-lock.txt](../requirements-lock.txt)，原始环境证据见 [environment.json](../artifacts/environment.json)。pip check 返回 No broken requirements found。安装成功不等于所有接口已验证；本报告只覆盖以下明确执行的测试。

## 来源固定与模型下载

- 参考仓库：datawhalechina/self-llm。
- 实际章节：models/Qwen3/08-Qwen3_0_6B的小模型有什么用.md，路径未变。
- 参考 commit：e0fe14a35123f5cab6d32cb9e716b571bf0994cf，通过 GitHub 官方 API 查询 master 获取。
- 目标模型：Qwen/Qwen3-0.6B。
- 实际 revision：c1899de289a04d12100db370d81485cdf75e47ca，通过官方 API 核对并按此下载。
- 模型许可：Apache-2.0；原许可证随本地模型保留。
- model.safetensors：1,503,300,328 bytes；SHA256 为 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b，与官方 LFS SHA256 比较通过。
- 全部文件摘要见 [model_manifest.json](../artifacts/model_manifest.json)。加载只使用本地文件，trust_remote_code=False，不执行模型仓库 Python 代码。
- 官方资料、访问日期和用途见 [sources.md](sources.md)；教程原文及许可见 docs/reference/。

## 数据与标签

100 条数据完全虚构，25 个不同源模板各 4 个变体，seed=42；train/val/test 为 60/20/20，对应 15/5/5 组。全部缺失、第三方信息、身份不确定及无投诉样本也保留本组模板，避免借用跨组的通用模板。

覆盖四字段齐全、多个缺失、顺序变化、干扰人物、第三方地址/邮箱、长短句及无有效投诉。question 是原文连续片段并保留标点，姓名仅认投诉人。标注规则及当前合成生成器适用的模板归一化检查见 [annotation_rules.md](annotation_rules.md)。

训练集 SHA256：a90d3d67c6844027f97c66780d3ae15c3885fb33d48606044bff79520c14e06b。最终 smoke、单条推理和当前 train 文件一致。Tokenizer 实际编码 train 为 178–251 tokens，val 为 177–233 tokens，均包含前缀+目标+EOS，没有截断或过滤。未对测试集做模型推理、Prompt 选择或参数选择；格式与组隔离检查不用于模型效果决策。

c02-1：169 个 prompt token 的 labels=-100，后 50 个 token 仅为完整 JSON 与 EOS，共 219 tokens。c02-2 为 212 tokens，双样本 padding 检查右补 7 个 PAD。EOS=151645，PAD=151643，二者没有混用。详见 [token_labels_c02-1.json](../artifacts/token_labels_c02-1.json) 和 [smoke_batch.json](../artifacts/smoke_batch.json)。

## 已执行的检查

| 检查 | 最终结果 | 证据 |
| --- | --- | --- |
| 项目依赖安装及 pip check | 通过 | artifacts/install_torch.log、install_dependencies_retry.log、requirements-lock.txt |
| 本机 GPU FP32/BF16 小运算 | 通过；两个结果均为 3920.0 | artifacts/environment.json |
| 合成数据生成、字段/类型/重复/分组/归属检查 | 通过；100 条，60/20/20 | artifacts/check_data.log、data/manifest.json |
| CPU unittest | 20/20 通过，无 skip | artifacts/unittest.log；数据 8、评测 7、预处理 5 |
| 真实 Tokenizer、非思考前缀、JSON/EOS、padding、超长拒绝 | 通过 | tests/test_preprocessing.py、artifacts/smoke_dry_run.json |
| Datasets train/val 映射与总 token 长度 | 通过 | artifacts/smoke_dry_run.json |
| LoRA 实际模块及参数冻结 | 通过 | artifacts/lora_audit.json |
| 最终样本一次前向 | 通过；loss=0.2881852686405182，有限 | artifacts/smoke_backward.json |
| 最终样本一次反向 | 通过；全部 392 个适配器参数张量梯度存在且有限，196 个非零，基础参数无梯度 | 同上 |
| 原模型一条推理 | 接口通过，c02-1 的四字段与 gold 一致；不能据此声称业务准确率 | artifacts/base_one_example.json，保留 raw/parsed/token IDs/config |
| 评测入口默认 dry-run | 通过；无 --run 不加载推理模型 | scripts/evaluate.py |

最终 GPU smoke 为 batch=1、BF16、SDPA、r=8，最大长度 512。实际基础参数 596,049,920，加入 LoRA 后总参数 601,096,192，可训练参数 5,046,272，占 0.8395116%。七类真实线性层共 196 个模块接入 LoRA；逐层名称、维度、requires_grad 状态在审计 JSON 中。

最终峰值 allocated 为 2,539,966,464 bytes（约 2.37 GiB），reserved 为 2,573,205,504 bytes（约 2.40 GiB）。这是此单样本、无优化器状态的观测，不保证正式训练的显存占用。

标注修订前另执行过一次同规模 smoke，结果留在 smoke_before_data_revision.json，仅作历史记录；最终数据修订后重新执行以上检查，未沿用旧 loss。两次均无参数更新、无权重保存。

## 已发生的问题及处理

| 问题 | 具体错误 / 原因 | 当前状态 |
| --- | --- | --- |
| 工具默认 Windows 沙箱无法启动 | helper_sandbox_lock_failed；SetNamedSecurityInfoW sandbox dir failed: 5 | 使用工具允许的审批路径完成已授权的项目命令；未修改沙箱设置或系统权限。基础设施问题本身未修复 |
| 初次依赖安装失败 | pip 按 cp936/GBK 读取 requirements 中文注释，UnicodeDecodeError | 注释改为 ASCII，并使用 python -X utf8；重试安装通过。保留初次错误日志 |
| 初稿合成数据不够严谨 | 相同布局跨组、部分 null 与残留文本矛盾、question 末尾标点不一致 | 改为 25 个源布局直接渲染，统一归属与标点，并加回归检查 |
| 数据检查中间版本失败 | NameError: NAMES is not defined | 补齐导入，畸形行及实体归属检查均通过 |
| 评测中间版本失败 | null_failure_rate 重复加误填，出现 2.0；前导空白被误拒 | 每格最多计一次，合法空白接受；非法/缺字段保持主指标分母 |
| 非有限数字边界 | JSON 数字 1e999 可被 Python 解成 inf，保留 parsed 会导致后续严格序列化失败 | 用有限浮点解析器明确拒绝，raw 保留，parsed=None；正负溢出测试通过 |

当前没有阻止本阶段完成的项目问题。负例测试中的非法 JSON、模板异常和超长报错属于预期拒绝路径，不是最终测试失败。

## 与教程相比的修改

| 教程做法 / 核对点 | 本项目处理及原因 |
| --- | --- |
| Notebook 的 !pip/!wget，下载时 --no-check-certificate | 原始 Markdown 中 pip/wget 拼写正常；网页 p ip/w get 属渲染现象。本机改用 PowerShell+venv 和正常 TLS 官方下载 |
| 不完整环境安装、固定硬件/精度假设 | 核对官方 CUDA wheel，安装精简依赖，用实际 kernel、BF16 与模型反传验证 |
| pandas/SwanLab/外部跟踪例子 | 直接 JSONL 与 Datasets，不启用跟踪服务、不读取 API Key |
| 手工拼接 <s>/角色/think 标记 | 官方模板处理两端，并验证完整训练序列与推理前缀的 token 级一致性 |
| 用 pad_token_id 作为训练结束目标 | 使用真实 EOS，仅 JSON+EOS 参与 loss，padding=-100 |
| 超过 MAX_LENGTH 静默截断 | 明确报错，不让半截目标 JSON 进入训练 |
| 没有显式 train/val/test 隔离 | 按源模板组固定拆分并检查重叠；测试集保留到下一阶段 |
| 列出 target_modules 后直接接入 | 先检查真实模块是 Linear，再审计所有参数状态和梯度 |
| 直接 trainer.train()，批量训练 | 没有训练入口，默认 dry-run，最小前后向不更新 |
| 单例展示、采样式推理，无严格指标 | 共用固定贪心配置、严格 JSON/Schema/精确匹配/null 指标，人工预测先验证评测程序 |

## 未执行与下一步

正式训练、学习率/epoch 调优、完整测试集推理、适配器保存与重载一致性测试、原模型与微调模型效果比较均**未执行，按本阶段边界留待下一阶段**。重建一个全新 venv 并从完整 lock 文件恢复也未执行，不能把版本记录说成环境重建已验证。

建议按顺序阅读 README → annotation_rules.md → learning_guide.md → preprocessing.py → modeling.py。在 PowerShell 执行以下已验证的检查：

~~~powershell
Set-Location .\qwen3-lora-extraction
.\.venv\Scripts\python.exe -X utf8 scripts/check_env.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_data.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 scripts/smoke_test.py --samples 2
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate.py
~~~

最后两条默认不做参数更新；evaluate 默认只显示计划。需要亲手复查一次前后向时，可显式追加 smoke_test 的 --backward，仍不会更新或保存权重。未来适配器加载代码已准备，但没有训练后适配器可加载，详见学习指南中的“未执行”说明。

这批 100 条合成数据只证明流程可以检查和运行，不能证明真实业务泛化，也不存在可报告的“微调提升”。


## 2026-10-04 追加修正：生成参数必须核对实际生效值

用户运行原模型单条推理时，出现 generation_config default values have been modified 警告。实查已安装 Transformers 4.57.6 源码发现：对于较新版本保存的模型，默认参数合并会把用户 GenerationConfig 内与全局默认值相同的 do_sample=False 替换为模型的 True。本模型 generation_config.json 标记 4.51.0 并默认采样，因此旧调用实际采用采样，而旧 provenance.generation 仍记录请求值 False。

本报告此前关于“共用固定贪心配置已验证”的表述需要按此修正：数据、梯度和旧单条输出的检查结果仍成立，但旧 base_one_example.json/base_preview.json 不能作为实际贪心解码的证据，不能直接与未来固定贪心的微调模型进行效果比较。旧文件已保留，内容哈希在本次修复前后未改变。

变更只涉及 scripts/evaluate.py 和新增 tests/test_generation.py。通过公开 generate 接口显式传递生成参数及 use_model_defaults=False，同时预先补齐特殊 token ID，保存 requested/effective/model-defaults、生成模式和框架版本。没有修改 Transformers 安装包、模型权重、Prompt、数据或依赖版本；没有训练参数更新。

已实际执行：

- 针对性 CPU 回归：2/2 通过。用极小随机 Qwen3 模型，通过真实 generate 对照旧调用的采样行为；修复后的调用验证 multinomial 次数为零，逐 token 选择与 argmax 一致，内部合并结果符合传入配置。
- 全部 CPU unittest：22/22 通过，无失败。旧 artifacts/unittest.log 是之前 20 项测试的历史记录；本次测试的退出状态见本次执行输出。
- 一次真实 train c02-1 推理：成功，生成 62 个 token（含 EOS），无生成错误。新文件 artifacts/base_preview_verified.json 确认 generation_mode=greedy_search、do_sample=False、num_beams=1、max_new_tokens=256、use_model_defaults=False，使用 CUDA/BF16。
- 新输出的 JSON、Schema 和四字段全匹配指标均为 1.0，标准答案为 null 的邮箱未误填。本次仍只有 1 条接口检查样本。
- 独立只读审查未发现可操作问题；审查者没有重复运行测试，不把静态审查称为独立执行验证。

本次官方接口核验见 docs/sources.md 的 2026-10-04 追加记录。正式训练、完整测试集推理和微调前后效果比较仍未执行。

## 2026-10-07 下一阶段已完成

本报告记录前期准备与历史问题。用户随后授权完成首轮复现，现已正式训练、保存 LoRA、新进程重载并完成完整 20 条测试比较。最新状态以 reproduction_report.md、README 和 outputs/reproduction-20261007/reproduction_summary.json 为准；前期“未执行”记录是当时的阶段状态。