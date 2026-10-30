# 最新分析数值记录

这里保存40个独立LSTM训练评估、2026-10-30低高速实机及残差补偿实验的CSV、NPZ和JSONL。内容包括成功记录、被保护或预检中止的记录、静止验证和诊断，未丢弃失败记录。

4个分卷解压后保留原始 `log/joint_lstm_*`、`log/joint_residual_*` 路径。[manifest.json](manifest.json)列出全部文件和SHA256。

在仓库根目录校验：

```bash
python3 scripts/verify_w3_work_results.py results/20261030/analysis_recordings
```

与[主体成果归档](../README.md)一起解压后，可获得报告脚本及其最新分析输入。原始H5、bag、CAN不在本补充包内；不能将这些派生记录当成完整的原始机器人数据备份。
