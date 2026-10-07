# Qwen3-0.6B 投诉信息抽取 LoRA 复现项目

**已完成首轮闭环复现（2026-10-07）：数据处理 → 正式训练 → 保存 LoRA → 新进程重载 → 固定测试集前后比较。**

本机 Windows 11、RTX 5060 Laptop GPU、Python 3.12 项目 venv。使用 100 条完全虚构投诉，按源模板组拆分为 train/val/test=60/20/20。本轮固定 1 epoch、batch=1、梯度累积=4、学习率=1e-4，共 60 个微批和 15 次参数更新。

可分享的精简结果见 [results/](results/README.md)。完整记录见 [复现报告](docs/reproduction_report.md)，概念与真实样本解释见 [学习指南](docs/learning_guide.md)。[前期报告](docs/preparation_report.md) 保留准备阶段及兼容问题的历史记录。

## 已观察的结果

同一组 20 条测试样本、相同模型 revision/Prompt/BF16/贪心解码：

| 指标 | 原模型 | LoRA 后 |
| --- | --- | --- |
| 严格 JSON 可解析率 | 18/20，90% | 20/20，100% |
| Schema 合法率 | 18/20，90% | 20/20，100% |
| 四字段全部精确匹配 | 6/20，30% | 19/20，95% |
| gold=null 时错误填入内容 | 10/35，28.57% | 1/35，2.86% |

这只是小规模合成测试集上的观察，不证明真实业务泛化能力。唯一 LoRA 错误 c08-1：把会议通知误填为投诉内容。本轮没有根据测试结果改 Prompt、参数或选择检查点。

32 项 CPU 测试通过。训练循环约 14.02 秒，训练入口总过程约 28.25 秒（含加载、验证、保存及参考生成）。新进程重载验证约 10.17 秒；原模型/LoRA 完整测试各约 40 秒。

## 文件入口

| 文件 | 作用 |
| --- | --- |
| configs/lora.yaml | 当前集中配置；训练入口默认 dry-run |
| configs/reproduction-frozen.yaml | 本轮实际使用的冻结配置 |
| scripts/inspect_one.py | 入门：看一条原文、标准答案、token 和 labels |
| scripts/check_env.py / check_data.py | 本机 CUDA 实际运算、依赖与数据检查 |
| src/preprocessing.py | 官方非思考模板、JSON+EOS 标签、超长拒绝及 padding |
| src/modeling.py | 本地基础模型、LoRA 注入、冻结与重载 |
| scripts/train.py / src/training.py | 显式 --run 才训练；累积尾组、有限值、参数变化与日志检查 |
| scripts/verify_reload.py | saved tokenizer、全部适配器张量、参考生成一致性验证 |
| scripts/evaluate.py | 原模型和 LoRA 共用的固定生成与严格评测 |
| scripts/compare_runs.py | 核对可比性，重算原始预测指标，输出差值与逐行错误 |
| outputs/reproduction-20261007/ | 已完成的训练、权重、重载与完整测试结果 |
| docs/annotation_rules.md / sources.md | 标注与指标规则、官方资料及访问日期 |
| tests/ | 数据、评测、生成配置、训练累积和输出保护的 CPU 回归 |

## 现在亲手运行（PowerShell）

在已配置的本机，无需重新安装。下面第一行从仓库父目录执行；若已在仓库根目录，直接运行 Python 命令。以下检查不会重新训练：

~~~powershell
Set-Location .\qwen3-lora-extraction
.\.venv\Scripts\python.exe -X utf8 scripts/inspect_one.py
.\.venv\Scripts\python.exe -X utf8 scripts/train.py
~~~

第二条显示配置、60 个微批和 15 次预计更新；不创建模型、优化器或训练输出。

查看已训练模型的第一条测试回答：

~~~powershell
$result = Get-Content .\outputs\reproduction-20261007\lora_test.json -Raw | ConvertFrom-Json
$result.rows[0].raw
~~~

本轮实际执行过的训练命令：

~~~powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train.py --config configs/reproduction-frozen.yaml --run
~~~

已有 outputs/reproduction-20261007，重复执行会拒绝覆盖。需要新一轮时，先复制配置并设置全新的 training.output_dir；本训练入口限定首轮 1 epoch、batch=1。修改实验方案属于下一轮，本报告不会根据当前 test 分数继续调参。

## 结果位置

- adapter/adapter_model.safetensors：20,236,472 bytes，约 19.30 MiB 的 LoRA 权重；加载仍需要本地固定 revision 基础模型。
- training_summary.json / training_history.json / lora_audit.json：训练、梯度、基础冻结及 LoRA 真正变化的证据。
- reload_reference.json / reload_verification.json：保存前内存模型参考及新进程检查结果。
- base_test.json / lora_test.json：全部 20 条原始输出、解析结果及逐字段判定。
- comparison.json / reproduction_summary.json：公平比较和整体完成状态。

上述文件是本机产物，不随源码仓库上传；克隆仓库后可查看 results/ 的精简结果，或按训练命令自行生成。上述文件均在 outputs/reproduction-20261007；训练摘要里的 pending_separate_process 是训练脚本结束当时的状态，随后验证已经 passed，整体状态以 reproduction_summary.json 为准。

.gitignore 排除 venv、缓存、基础模型、训练输出和权重。源码与精简结果通过 Git 管理；基础模型、LoRA 权重及原始环境日志保留在本地。使用 PyTorch + Transformers + PEFT；Datasets 用于前期 smoke 数据映射。本轮训练采用便于学习的直接 PyTorch 循环，不依赖云服务、QLoRA、多卡或外部跟踪平台。

## 新克隆目录的安装说明

本仓库上传源码、配置、虚构数据和 results/ 精简摘要；不包含现成 venv、模型或训练输出。以下完整新环境重建命令为**待验证说明**；当前本机对应依赖已安装验证，详见复现报告。

~~~powershell
git clone https://github.com/shuxiachai/qwen3-lora-extraction.git
Set-Location .\qwen3-lora-extraction
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip --isolated install --cache-dir .cache/pip --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
.\.venv\Scripts\python.exe -X utf8 -m pip --isolated install --cache-dir .cache/pip --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 scripts/download_model.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_env.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_data.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
~~~

使用已有 100 条虚构 JSONL，prepare_data.py 可以确定性重生成。显式 --run 训练需要 CUDA；默认 dry-run 可先检查输入。完整环境版本在 requirements-lock.txt，按其重建全新环境尚未复验。

## 参考与范围

本学习项目参考 [Datawhale self-llm 的指定 Qwen3 章节](https://github.com/datawhalechina/self-llm/blob/e0fe14a35123f5cab6d32cb9e716b571bf0994cf/models/Qwen3/08-Qwen3_0_6B的小模型有什么用.md)。原文、来源与 Apache-2.0 许可保留在 docs/reference/。本项目另行编写数据、检查、训练和比较代码，修正了模板/标签/数据划分与生成参数兼容性问题。复现成果是学习性流程验证，不是生产服务或真实业务效果承诺。

## 许可

本项目自行编写的代码、文档和完全虚构的示例数据采用 [Apache-2.0 许可](LICENSE)。Copyright 2026 shuxiachai。

`docs/reference/tutorial.md` 是 Datawhale self-llm 教程的原样副本；来源、commit 和归属说明见 [NOTICE](docs/reference/NOTICE.md)，上游 Apache-2.0 许可保留在 [docs/reference/LICENSE](docs/reference/LICENSE)。

基础模型和各项依赖遵循各自的许可；本仓库不分发模型权重。
