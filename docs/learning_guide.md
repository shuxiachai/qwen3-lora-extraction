# 从一条投诉理解微调

项目已完成一轮真实 LoRA 微调与重载验证，已保存模型在 outputs/reproduction-20261007/adapter。完整结果见 reproduction_report.md。先阅读 `annotation_rules.md`，再结合 `src/preprocessing.py` 和 `artifacts/token_labels_c02-1.json` 看下面的例子。

## 1. 文本、标准答案、Tokenizer

训练样本 `c02-1` 的投诉人是周明远，地址是虚构的云岚市星桥区幻月街2号雾港苑2室，投诉内容是“新装的星网宽带连续掉线，报修后仍未处理。”。它没有投诉人的邮箱，因此标准答案为：

```json
{"name":"周明远","address":"云岚市星桥区幻月街2号雾港苑2室","email":null,"question":"新装的星网宽带连续掉线，报修后仍未处理。"}
```

`input` 的完整原文以当前 `data/train.jsonl` 和 token 明细中的 `input` 为准。`build_messages` 创建 system 指令和 user 原文两条消息。system 说明四字段、归属和不猜测的规则；训练时再加入 assistant 的标准 JSON。模型每次看到这些消息最后仍是一串 token。

Tokenizer 根据官方词表把文字转换成整数 ID。一个汉字不一定对应一个 token，英文、标点和空格也会改变分词；不能按字符数推断长度。我们把完整消息交给 `apply_chat_template(tokenize=True)`，由模型自带模板处理角色和控制符，不手写 `<s>` 或另加 BOS。

训练的完整对话使用 `add_generation_prompt=False`；推理只给 system+user，使用 `add_generation_prompt=True`。二者都显式 `enable_thinking=False`。这个 revision 会在 assistant 开头放入空的 `<think>\n\n</think>\n\n` 前缀，它是已给定上下文，本项目不监督这一段。

`encode_example` 分别取得推理前缀 token 和完整训练对话 token，并断言前者是后者的精确前缀。它再确认目标部分解码后恰好等于完整标准 JSON，后面只有一个官方 EOS。任何不一致都会报错，避免把错误边界写进 labels。

## 2. input_ids、attention_mask 和 labels

| 对话区域 | input_ids | attention_mask | labels | 是否作为预测目标 |
| --- | --- | --- | --- | --- |
| system、user、assistant 非思考前缀 | 官方 token ID | 1 | -100 | 否，但模型仍读取它们 |
| assistant JSON | JSON token ID | 1 | 同位置 token ID | 是 |
| 回复结束 `<\|im_end\|>` | 151645 | 1 | 151645 | 是 |
| 批次右侧 padding | 151643 | 0 | -100 | 否 |

`-100` 是 PyTorch 交叉熵默认忽略的标签值，不是词表里的 token，也不会被模型输出。它表示“这个位置没有需要学习的目标”。attention mask 解决的是注意力是否使用 padding，labels 解决的是哪些位置参与 loss，两者不能互相代替。

CausalLM 在内部做 next-token shift：位置 t 的 logits 用来预测位置 t+1 的 label。因此第一个 JSON token 由前缀末尾位置预测，不要手工再次平移 labels。前缀虽然 labels=-100，仍通过注意力影响答案。

打开 `artifacts/token_labels_c02-1.json`，逐行看 `position/token_id/token/label/contributes_to_loss`。中文 token 的词表表示有时看似乱码，这是字节级分词的展示；完整 `supervised_text` 给出可靠解码。`artifacts/smoke_batch.json` 展示两条不同长度样本如何补齐。

最大长度 512 指前缀+JSON+EOS 的总 token 数。程序完全不截断：超过上限就抛出 `SampleTooLong`，不会留下半截 JSON 继续训练。未来扩大数据集时，应记录过滤比例或有依据地调整上限，不能悄悄丢掉答案。

## 3. LoRA 接到了哪里

加载模型后先遍历 `named_modules()`，确认七类目标都是实际存在的 Linear：q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj。`artifacts/lora_audit.json` 列出每层实际名称、维度、参数总量和所有可训练参数，不能把教程打印的数字直接当作本机结果。

对权重 W，LoRA 增加 `(alpha/r) × B × A`。rank=8 表示 A/B 的中间维度为 8；alpha=32 对应缩放 4。原来的 W 保持冻结，只有名称中 `lora_A` / `lora_B` 的参数允许训练。本例不训练 embedding、lm_head 或原始线性层权重。`requires_grad` 审计和 backward 后的梯度审计分别验证这一点。

默认初始化 A 为随机矩阵、B 为零。因此第一次 backward 的 A 梯度可以恰好为零；这不意味着反传失败。检查要求所有适配器梯度存在且有限，至少一部分非零，基础参数梯度全部为空。

LoRA 省下的是待训练权重、梯度和优化器状态的一大部分，基础模型及激活仍占显存。单次 smoke 的显存占用不能当作完整训练的显存保证。

## 4. 四个常见训练参数

- **batch size**：一次前向看的样本数。本项目模型 smoke 固定为 1；处理两条用于查看 padding，并不把它们一起训练。
- **梯度累积**：多次前后向先累加梯度，再更新一次。单卡 batch=1、累积=4，通常对应一次更新汇总约 4 个样本；末尾不足一组及 loss 归一化需要下一阶段实现时核对。smoke 检查不累积、不更新；正式 train.py 本轮采用累积 4 次再更新。
- **学习率**：优化器更新参数的步长尺度。本轮固定使用 1e-4，是未调优的首轮起点。
- **epoch**：遍历训练集一轮。本轮正式训练为 1，共处理 60 条训练样本。

前向产生 loss；反向把它对可训练参数的导数放入 `.grad`；真正改变参数还需要优化器更新。`smoke_test.py` 没有创建优化器或调用更新，也没有保存适配器。一次有限 loss 只证明这份输入可以计算，不证明模型已经学会抽取。

## 5. 保存与重新加载（现已实际验证）

PEFT 的 `model.save_pretrained(adapter_dir, safe_serialization=True)` 保存适配器权重和配置，`tokenizer.save_pretrained(adapter_dir)` 可一起保存模板/词表信息。这不是完整基础模型。必须同时记录基础模型 ID、revision、训练配置、数据版本以及评测配置。

本项目已准备 `load_adapter_for_inference(config, adapter_path)`：核对适配器记录的 revision，先加载相同基础模型，再用 `PeftModel.from_pretrained(..., is_trainable=False)` 接回适配器。之后同样使用 `eval()` 与共用推理流程。本轮已保存并完成新进程重载验证：全部适配器权重精确一致，参考样本生成 token 和原始回答也完全一致。

## 6. 为什么 loss 下降不等于效果提升

模型可能记住训练模板；也可能更善于输出常见 JSON，却在缺失字段时编造信息。训练 loss 是已知标准答案条件下的 token 预测误差，实际生成还涉及逐步累积的错误。必须用独立数据和固定生成配置衡量完整输出。

训练集用于更新适配器；验证集用于开发与参数选择；测试集留到最后一次方案冻结后的比较。来自同一源模板的改写属于同一 group，不能跨集合。数据格式检查可以读测试集，不能根据测试模型表现反复改 prompt。

评测先要求整个输出是严格 JSON，再检查四字段 schema，然后比较每个字段。格式失败不能被排除。`evaluation.py` 保留原始输出、解析结果和逐字段判断；所有比例的分母见 `annotation_rules.md`。真实空值误填率还应结合空值失败率一起看：完全不会输出 JSON 的模型不能因为“没有可解析的乱填值”而显得很好。

本项目第一版选择固定的贪心解码以便比较，未按效果调参，也不声称这是模型最优生成配置。未来原模型和微调模型必须保持测试样本、prompt、revision、精度与推理设置可比较。本阶段的一条原模型检查即使全对，也不是可靠准确率结论。

## 7. 微调、Prompt 和 RAG 的边界

这里选择微调是为了学习：用少量可训练参数让一个固定模型适应稳定字段和标注规则。在真实项目中，先验证清晰 Prompt 或少量示例能否达到要求，再判断数据量、成本和维护投入是否值得微调。本批 100 条合成数据只适合验证流程。

经常变化的产品说明、最新政策、企业知识和可追溯出处通常更适合 RAG；单纯改输出要求常可以先改 Prompt。微调不适合当成可靠的知识库更新机制，也不能保证消除幻觉。此任务只从给定投诉里抽取，不需要先增加检索、工具调用或 Agent 框架。

官方资料与访问日期见 [sources.md](sources.md)。


## 实际 token 边界补充（2026-09-14 最终数据）

c02-1 的完整原文：

> 邮件主题【服务申诉】。投诉人：周明远。投诉人邮箱未登记。投诉人地址：云岚市星桥区幻月街2号雾港苑2室。正文：新装的星网宽带连续掉线，报修后仍未处理。

实测总长 **219 tokens**，前 169 个屏蔽，后 50 个监督 JSON 与 EOS。以下位置从 0 开始，来自 artifacts/token_labels_c02-1.json：

| 位置 | 内容 | token ID | label | 参与 loss |
| --- | --- | --- | --- | --- |
| 167 | 结束 think 标签 | 151668 | -100 | 否 |
| 168 | 两个换行 | 271 | -100 | 否 |
| 169 | JSON 起始 {" | 4913 | 4913 | 是 |
| 170 | name | 606 | 606 | 是 |
| 171 | ":" | 3252 | 3252 | 是 |
| 217 | "} | 9207 | 9207 | 是 |
| 218 | 回复 EOS | 151645 | 151645 | 是 |

第二条样本长 212 tokens，在两条样本的检查 batch 中右补 7 个 PAD；补齐位置 attention_mask=0、labels=-100。模型前后向只使用第一条，batch size 为 1。最终 loss=0.2881852686405182，仅代表未更新参数时的可计算性检查。


## 2026-10-04 实操补充：配置与实际运行行为

本次运行揭示了一个实际工程问题：配置写了“不随机采样”，框架却可能补入模型默认的随机采样设置。看到生成参数被修改的警告，应检查实际生效值。

旧版本的 provenance.generation 只记录请求配置；本指南此前对固定贪心配置的说明是设计意图，实际验证不足。现已修复公开 generate 调用，并补上真实生成回归测试。新文件 artifacts/base_preview_verified.json 的 provenance.generation_mode 是 greedy_search，generation_effective.do_sample 为 false，use_model_defaults 为 false。

这条投诉在修复后的原模型生成中四字段全部正确。它说明原模型有能力处理这道题，也说明单条生成和评测接口能工作；它不能证明整体准确率为 100%，也不能证明微调一定有必要。未来比较要使用相同数据、Prompt 和可核对的生成设置。


## 2026-10-07：你已经完成一次真实微调

此前 smoke 只算梯度；这次 scripts/train.py --run 真正执行了优化器更新。60 条训练样本每条处理一次，四条累积一组，共有 15 次更新。检查确认所有 392 个 LoRA 参数张量都改变了，基础模型保持冻结。

阅读 src/training.py 的 train_epoch，按顺序找：

1. loss = model(**batch).loss：模型前向，得到被监督 token 的平均误差。
2. (loss / group_size).backward()：按本累积组实际条数缩放，再把梯度累加。
3. clip_grad_norm_：限制梯度幅度，检查数值有限。
4. optimizer.step()：真正调整 LoRA 参数。
5. optimizer.zero_grad(set_to_none=True)：这一组更新结束，清空梯度准备下一组。

本轮每组四条；实现也正确处理最后不足四条的情形。loss 采用每条样本 token mean 的等权平均，并不等同于把不同长度样本的全部 token 混在一起统一求均值。

scripts/train.py 保存 adapter 后，用内存模型生成 train c02-1 的参考答案；scripts/verify_reload.py 在新进程加载权重与 Tokenizer，逐张量核对，生成的 token 也逐个一致。这比“目录里有个权重文件”提供了更直接的保存/重载证据。

固定 20 条测试上，原模型四字段全对 6 条，LoRA 全对 19 条。剩下 c08-1 错在把会议通知当成投诉。这是一个学习案例：格式正确、很多字段正确，也可能有业务判断错误。20 条合成数据的结果只用于本轮流程观察。

已保存模型位置：outputs/reproduction-20261007/adapter。完整运行与结果见 reproduction_report.md，整体完成状态见 outputs/reproduction-20261007/reproduction_summary.json。
