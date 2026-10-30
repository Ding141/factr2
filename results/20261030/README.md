# 截至 2026-10-30 的 NEXT / W3 工作成果

本次保存代码、模型权重、训练与实验配置、补偿模型、报告、波形、审计及源码快照。当前实现为七个独立关节 LSTM，每个网络读取全部关节的50帧状态；保留旧共享 LSTM 及其备份。

## 最新报告

|工作|报告|要点|
|---|---|---|
|完整重训与旧模型保留|[独立 LSTM 训练报告](reports/joint_lstm_retrain_20261028_v1/REPORT.md)|逐关节独立选择验证集最佳轮次；旧模型不删除|
|低速与高速新旧模型实机对比|[完整对比报告](reports/joint_lstm_live_comparison_20261030_v2/COMBINED_REPORT.md)|高速同滤波RMS 0.56429→0.13724 N·m；低速部分关节退步|
|同模型残差补偿实机对比|[补偿完整报告](reports/joint_residual_comparison_20261030_v2/REPORT.md)|同帧RMS 0.137249→0.136665 N·m，仅改善约0.43%；J6略有恶化|
|上一轮与本轮波形|[三路叠加图](reports/joint_residual_comparison_20261030_v2/prior_vs_current_waveforms.png)|区分跨次变化与本次同帧补偿效果|
|当前补偿窗口|[运动中截图](reports/joint_residual_comparison_20261030_v2/compensated_window_during_motion.png)|8082为当前独立LSTM未补偿，8083为同一模型补偿后|

更多阶段包括采集协议、受控消融、冻结方差头、速度/加速度补偿、平滑轨迹、CAN及数据质量审计。见[全部报告索引](REPORT_INDEX.md)。报告中的缺失原始文件链接会标明原始路径，不伪装成已上传的数据。

## 模型及当前观察配置

- 当前独立模型：[joint_lstm_observer_20261028_v1](../../models/w3/right/joint_lstm_observer_20261028_v1)。
- 旧共享模型：[wide_highspeed_observer_20261026_v1](../../models/w3/right/wide_highspeed_observer_20261026_v1)。
- 旧模型备份：[comparison_baseline_20261028_v1](../../models/w3/right/comparison_baseline_20261028_v1)。
- 活动配置：[observer_active.json](../../config/w3/right/observer_active.json)。残差启用J1/J6/J7，J2–J5保持未补偿；模型权重未改变。
- 补偿文件在分卷归档内的 `log/joint_residual_comparison_20261030_v1/compensation/`，恢复后即可得到原始路径。
- [模型文件SHA256清单](model_files.json)记录全部模型文件；每个模型目录另外保留自身原有清单。

## 归档内容与恢复

`experiment-results.tar.gz.part001` 等分卷合计约771 MB，保存8,477个文件，保留原来的 `data/`、`log/`、`runs/` 相对路径。包括历史训练权重与恢复检查点、归一化文件、实验参数、报告、图、补偿文件、源码快照以及小型审计记录。

[manifest.json](manifest.json)记录每个分卷及每个归档文件的SHA256、字节数，并列出没有上传的原始流或可重算缓存。约19.1 GB的原始H5、bag、CAN、逐帧CSV/JSONL和大数组缓存未包含在这些分卷中，仍在原机器；这不是完整的原始采集数据备份。

先校验（只读取，不解压、不运行机器人）：

```bash
python3 scripts/verify_w3_work_results.py results/20261030
```

在仓库根目录恢复归档时，可先查看已有同名文件并选择独立目录，避免覆盖自己的实验记录：

```bash
mkdir -p /tmp/factr2-results-20261030
cat results/20261030/experiment-results.tar.gz.part* | tar -xzf - -C /tmp/factr2-results-20261030
```

可浏览的 `reports/` 是发布副本，Markdown链接经过转换；归档内保留原始报告内容及其哈希。归档的进程状态仅为历史记录，不应据此恢复或重复机械臂运动。

配置保留原机器 `/home/venom/factr2` 的绝对路径，以维持原模型清单一致性。异机迁移应通过导出/激活脚本重新生成配置及清单；不要批量修改被哈希绑定的模型文件后继续使用旧清单。原始数据没有恢复时，不具备从头重训全部历史实验的完整输入。

## 配套控制仓库

控制侧反馈新鲜度改动与测试保存在 [Ding141/dual_arm_robot](https://github.com/Ding141/dual_arm_robot/tree/liyq-dev)，本仓库不复制其Git历史。当前成果不等同于已通过真实接触测力标定，具体有效范围与不足见各报告。
