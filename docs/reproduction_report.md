# 首轮闭环复现报告

**完成日期：2026-10-07，Australia/Sydney。状态：已实际训练、保存、重载并完成固定测试比较。**

项目位置：克隆后的仓库根目录（qwen3-lora-extraction）。原准备阶段的“不开展正式训练”边界已在用户要求完成复现的下一阶段中推进；本轮仅执行固定的一轮小规模训练，没有长时间调参、付费 API、云 GPU、上传或系统级更改。

## 本轮实际方案

| 设置 | 固定值 |
| --- | --- |
| 模型 | Qwen/Qwen3-0.6B |
| revision | c1899de289a04d12100db370d81485cdf75e47ca |
| 参考仓库 commit | e0fe14a35123f5cab6d32cb9e716b571bf0994cf |
| 参考章节 | models/Qwen3/08-Qwen3_0_6B的小模型有什么用.md |
| 数据 | 100 条虚构数据、25 个源模板组；train/val/test=60/20/20 |
| 随机种子 | 42 |
| 最大总序列长度 | 512，不静默截断目标 |
| LoRA | r=8、alpha=32、dropout=0.1；七类真实 Linear 模块，共 196 个 |
| 精度 / 注意力 | 基础 BF16、LoRA FP32，单卡 CUDA，SDPA |
| batch / 累积 / epoch | 1 / 4 / 1 |
| 优化器 | AdamW，学习率固定 1e-4，weight_decay=0，梯度裁剪 1 |
| loss 目标 | 每条样本的被监督 token 均值，再在累积组内对样本等权平均；最后不足组按实际条数缩放 |
| 生成 | 官方非思考模板、greedy_search、do_sample=false、num_beams=1、max_new_tokens=256、use_model_defaults=false |

冻结配置：configs/reproduction-frozen.yaml。训练前 artifact/代码/数据摘要见 artifacts/experiment_plan_20261007.json。完成汇总脚本核对数据、配置、训练/推理源文件 SHA256 与冻结记录一致。只有 train 更新参数；val 的训练前后 loss 仅作诊断，没有选择超参或检查点。训练结束后才执行两次完整 test，没有依据测试结果改方案。

当前环境再次实测为 Windows 11 10.0.26200、Python 3.12.9、驱动 591.91、RTX 5060 Laptop GPU 8151 MiB。torch 2.10.0+cu128、Transformers 4.57.6、PEFT 0.18.1、Datasets 4.5.0、Accelerate 1.12.0、PyYAML 6.0.3；CUDA FP32/BF16 小运算与 pip check 通过。详见 artifacts/environment_reproduction.json 和 requirements-lock.txt。未改变全局环境；PATH 上未找到 nvcc，已使用官方 wheel 完成计算。首轮实验执行时尚未初始化 Git；发布后的提交记录可用 git log 查看。上表 commit 是参考仓库版本。

## 执行与验证

| 检查 | 实际结果 |
| --- | --- |
| 环境、数据、依赖检查 | 通过 |
| train.py 默认 dry-run | 通过；60 个微批、15 次预计更新，不加载模型或创建优化器/输出 |
| 全部 CPU unittest | 32/32 通过，无 skip；日志 artifacts/unittest_reproduction.log |
| 正式训练 | 60 个不同训练样本、60 次前后向、15 次 AdamW 更新，固定 1 epoch |
| 梯度 | 首次 392 个 LoRA 梯度张量都存在且有限；196 个非零，符合默认 B=0 初始化 |
| LoRA 变化 | 392/392 个适配器张量数值发生变化且有限，基础参数无梯度 |
| 基础参数 | 310 个基础参数张量训练前后 _version 一致；优化器只包含 LoRA，不复制基础权重进行逐值比对 |
| 保存 | safetensors + adapter_config + Tokenizer 保存成功；官方基础模型 ID/revision 已记录 |
| 新进程重载 | 全部 392 个适配器张量与文件精确匹配，Tokenizer/模板一致，全部参数冻结 |
| 保存前后行为 | train c02-1 的 prompt 编码、greedy 生成 token 和 raw 完全一致；行为检查覆盖这一参考样本 |
| 完整固定测试 | 原模型和 LoRA 各 20 条，均无 generation_error |
| 可比性检查 | 相同实际 test 行/id/gold/input、数据 SHA、模型/Prompt/seed/BF16/device、完整有效生成配置 |
| 独立复核 | 只读重新计算全部评测指标、核对训练/重载/时间/配置证据，无未解决问题；未重复训练 |

总参数 601,096,192，可训练参数 5,046,272（约 0.8395%）。基础冻结与适配器变化证据在 lora_audit.json；训练逐微批 loss、每次 update 和梯度范数在 training_history.json。

本轮实际耗时：

- 训练循环：14.0213 秒。
- 训练入口总过程：28.2476 秒，包含模型加载、训练前后验证、保存和参考生成。
- 新进程重载验证：10.1668 秒。
- 原模型完整 20 条测试：39.6378 秒；LoRA 完整测试：40.3251 秒，均包含各自进程启动与模型加载。
- 训练全过程峰值 allocated：2,706,083,328 bytes，约 2.52 GiB；reserved：3,619,684,352 bytes，约 3.37 GiB。
- adapter_model.safetensors：20,236,472 bytes，约 19.30 MiB。实际文件 SHA256 与 training_summary 中的摘要一致。

这些是该硬件、短合成数据和此配置下的观测，不是通用运行时间或显存保证。

## 固定测试集上的结果

测试集为 20 条；gold 为 null 的字段单元格总数为 35。非法 JSON/schema 行仍以四个字段错误计入全部样本分母；不清洗围栏或人工挑答案。

| 指标 | 原模型 | LoRA 后 |
| --- | --- | --- |
| 严格 JSON 可解析率 | 18/20，90% | 20/20，100% |
| Schema 合法率 | 18/20，90% | 20/20，100% |
| name 精确匹配 | 12/20，60% | 20/20，100% |
| address 精确匹配 | 18/20，90% | 20/20，100% |
| email 精确匹配 | 18/20，90% | 20/20，100% |
| question 精确匹配 | 9/20，45% | 19/20，95% |
| 四字段全部正确 | 6/20，30% | 19/20，95% |
| null 误填 | 10/35，28.57% | 1/35，2.86% |
| null 失败（含非法/缺字段） | 15/35，42.86% | 1/35，2.86% |

val 平均每样本 loss 从 0.4859383 变为 0.0437659。该诊断与实际生成评测分别保留，不用 loss 代替字段指标。

本轮在这 20 条合成数据上观察到更高的格式和字段匹配率。测试只有 5 个合成源模板组，数据模式规则化，不能由此推定真实投诉业务的泛化表现，也不宣称复现了教程的原始业务指标。这里完成的是沿用指定教程任务、固定模型和 LoRA 路线的学习性流程复现。

## 仍有的错误

唯一 LoRA 错误 c08-1 的原文是一份没有投诉人的会议通知，gold 的四字段都应为 null。模型正确留空了姓名、地址和邮箱，却输出：

~~~json
{"name":null,"address":null,"email":null,"question":"会议定在云岚市明日举行，附件为空。"}
~~~

这说明“判断有没有有效投诉”仍可能出错。我们记录该错误，没有根据它继续调整 Prompt 或训练参数。

原模型 c04-3、c09-4 输出 Markdown 代码围栏，严格 JSON 解析失败。这两行没有被丢弃，四字段均在 20 个样本的分母中记错。

## 新增工程与修复

- scripts/train.py / src/training.py：默认 dry-run，显式 --run 训练；固定首轮设置、尾组累积正确缩放、有限值/冻结/变化审计、失败时保留日志、拒绝覆盖已有训练目录。
- scripts/verify_reload.py：保存 Tokenizer 与基础模型对照、全部 LoRA 张量精确核对、内存参考和新进程生成一致性。
- scripts/compare_runs.py：重算原始结果、严格检查可比性、给出差值与每条错误，不宣称必然提升。
- scripts/evaluate.py：保留已经验证的生成参数修复；新增模型加载前拒绝覆盖旧结果。
- 独立审查发现比较器的词法路径检查可能允许 ../；已改成 resolve 后检查项目范围，并加入回归。
- 测试新增梯度累积/尾组、优化器只能更新 LoRA、dry-run 无写输出/不可读 test、结果不可比拒绝和输出保护。未扩大依赖。

默认 Windows 工具沙箱仍有 helper_sandbox_lock_failed / SetNamedSecurityInfoW error 5 的基础设施问题；使用工具批准的项目命令完成操作，未修改沙箱或系统设置。没有其它阻碍本轮完成的环境问题。

## 文件位置与亲手复查

所有训练结果在 outputs/reproduction-20261007：

- adapter/：LoRA 权重、配置和 Tokenizer。基础模型继续使用 models/Qwen3-0.6B 的固定官方快照。
- training_summary.json、training_history.json、lora_audit.json：训练与更新证据。
- reload_reference.json、reload_verification.json：保存前后的精确验证。
- base_test.json、lora_test.json：全部原始答案、解析结果和配置。
- comparison.json、reproduction_summary.json：公平比较与整体完成状态。
- config.yaml：训练时配置副本；evaluation_timing.json：实际评测时间。

training_summary 中 pending_separate_process 是训练脚本结束时的阶段性字段；随后重载检查已经 passed，整体状态见 reproduction_summary.json。

以下是已经实际验证、可安全重复的 PowerShell 检查：

~~~powershell
Set-Location .\qwen3-lora-extraction
.\.venv\Scripts\python.exe -X utf8 scripts/inspect_one.py
.\.venv\Scripts\python.exe -X utf8 scripts/train.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
~~~

本轮正式训练、重载、原模型/LoRA 测试与比较命令已实际执行，日志在 artifacts/training_run_20261007.log、reload_run_20261007.log、base_test_20261007.log、lora_test_20261007.log。已有结果默认拒覆盖，阅读它们不需要重复训练。

后续学习建议：先在 src/training.py 找到 loss.backward、optimizer.step、zero_grad，结合 60 个微批和 15 次更新日志，再阅读训练脚本如何保存与 verify_reload 如何验证。新增完整实验时使用新的配置和输出目录；若根据本次错误改进方案，应另留未用于开发的新测试数据，不把当前 test 分数反复用来挑参数。

官方资料与本轮访问日期见 docs/sources.md。首轮复现完成，不自动进入额外 epoch 或调参。
