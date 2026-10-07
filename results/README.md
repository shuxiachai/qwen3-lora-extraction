# 可分享的复现实验结果

本目录是 2026-10-07 合成 LoRA 复现实验的精简摘要，不是权重备份。适配器仍只保存在 `../outputs/reproduction-20261007/adapter`，这里不复制权重或完整运行产物。

证据来源为本地 `../outputs/reproduction-20261007/` 中的 `reproduction_summary.json`、`training_summary.json`、`comparison.json`、`base_test.json`、`lora_test.json` 与 `reload_verification.json`。方法与准备背景见 [复现报告](../docs/reproduction_report.md)、[准备报告](../docs/preparation_report.md) 和 [标注规则](../docs/annotation_rules.md)。原始 outputs 仅保留在本地，未作为本摘要的一部分。

- `summary.json`：冻结配置及数据哈希、模型 revision、训练规模和时长、适配器大小、软件版本/GPU 信息、32 项 CPU 测试与重载状态。路径均为仓库相对路径。
- `comparison.json`：固定 20 个合成测试样本上的精选聚合指标。`base` 为未加载适配器的模型，`lora` 为加载适配器的模型。
- `error_cases.json`：三个真实测试案例：`c08-1` 的 LoRA 剩余错误（同时保留 base 对照），以及 `c04-3`、`c09-4` 的 base 严格 JSON 围栏错误。原始 `raw` 未被清洗或改写。

空值误填指标的分母是 35 个 gold 为 `null` 的字段。结果来自很小的合成测试集，不保证对业务数据泛化；当前错误分析不用于本轮调参。
