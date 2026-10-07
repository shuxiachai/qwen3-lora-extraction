# 已使用资料

访问日期均为 **2026-09-14（Australia/Sydney）**。先阅读教程和下列官方资料，再实现处理及模型代码。文档推荐、安装成功和实际计算通过是不同证据层次；本机结果见 `artifacts/`。

| 资料 | 用途与核对结论 |
| --- | --- |
| [Datawhale 指定章节，固定 commit](https://github.com/datawhalechina/self-llm/blob/e0fe14a35123f5cab6d32cb9e716b571bf0994cf/models/Qwen3/08-Qwen3_0_6B的小模型有什么用.md) | 保留原项目与四字段任务。实际路径未变。原文及 Apache-2.0 许可留在 `docs/reference/`。 |
| [Qwen 官方模型卡，固定 revision](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md) | Transformers 低于 4.51.0 不识别 qwen3；非思考通过模板参数关闭。模型卡不保证本机效果或吞吐。 |
| [官方 tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json) | 实际 Jinja 模板、非思考前缀、EOS=`<\|im_end\|>`、PAD=`<\|endoftext\|>`；本地下载副本用于运行。 |
| [PyTorch 官方安装](https://pytorch.org/get-started/locally/) / [固定版本命令](https://pytorch.org/get-started/previous-versions/) | 选择官方 Windows CUDA 12.8 wheel，固定 torch 2.10.0。只安装 torch，没有添加不需要的 torchvision/torchaudio。 |
| [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/) | Blackwell 支持与 CUDA 12.8 wheel 的官方说明；本机仍必须执行真实 kernel 验证。 |
| [Transformers 4.57.1 chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating) | messages、add_generation_prompt、训练完整对话关闭额外生成前缀；避免重复 special tokens。查阅 4.57.1 文档，实际安装并测试 4.57.6。 |
| [Transformers Qwen3 forward 接口](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3) | labels=-100 被忽略、attention mask、CausalLM loss 与 use_cache。 |
| [PEFT LoRA 文档](https://huggingface.co/docs/peft/v0.18.0/developer_guides/lora) | 实际模块选择、rank/alpha/dropout、A/B 初始化；实际安装 0.18.1。 |
| [PEFT checkpoint 格式](https://huggingface.co/docs/peft/v0.18.0/developer_guides/checkpoint) | 适配器保存内容与基础模型依赖；本阶段仅说明保存方法，未保存任何训练后权重。 |

仓库 commit 通过 GitHub 官方 API 获取；模型 revision 通过 Hugging Face 官方 API 获取。未把网页缓存中的“最新版”当作本机已验证版本。所有依赖版本以 `requirements-lock.txt` 和 `artifacts/environment.json` 为准。


## 2026-10-04：生成参数兼容性复核

追加查阅 [Transformers 4.57.1 GenerationMixin.generate 官方接口](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)，并只读核对本项目实际安装的 Transformers 4.57.6 generation/utils.py。官方接口提供 use_model_defaults 参数；未设置时，较新格式模型会启用模型默认值合并。

本项目的旧调用中，GenerationConfig 内 do_sample=False 与框架全局默认值相同，会被 Qwen 模型的 do_sample=True 覆盖。旧结果的 provenance.generation 记录的是请求配置，不能作为当时实际使用贪心解码的证据。修复与重新验证的结果见 preparation_report.md 的追加记录。


## 2026-10-07：首轮正式训练与重载

本轮实现前查阅：

- [PyTorch 2.10 optimizer 官方说明](https://docs.pytorch.org/docs/2.10/optim.html)：优化器接收指定参数集合，backward 计算梯度后 step 更新参数，zero_grad 清空梯度。
- [PEFT 0.18 checkpoint 官方说明](https://huggingface.co/docs/peft/v0.18.0/developer_guides/checkpoint)：保存的是适配器参数/配置，重新使用需要基础模型。
- [PEFT integration functions](https://huggingface.co/docs/peft/v0.18.0/package_reference/functional)：使用 get_peft_model_state_dict 与保存的 safetensors 核对全部适配器参数。实际安装 PEFT 0.18.1，已做真实新进程重载验证。

模型、教程与先前官方模板/生成资料版本保持原记录。实际环境及本轮完整结果见 reproduction_report.md；未按网页最新版升级任何依赖。
