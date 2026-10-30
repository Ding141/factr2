# 新旧 LSTM 的保留、训练与对比

2026-10-28。本轮在独立关节 LSTM 上完整训练，并关闭新旧模型的附加残差补偿。
训练、导出、激活和在线状态以 `log/joint_lstm_retrain_20261028_v1/pipeline_state.json`
为准；本说明中的命令不会启动机械臂运动。

## 保留旧模型

原目录 `models/w3/right/wide_highspeed_observer_20261026_v1` 保持完整，另存核验副本
`models/w3/right/comparison_baseline_20261028_v1`。14 个文件的 SHA256 记录在
`log/joint_lstm_retrain_20261028_v1/preserved_old_model.json`，部署前后均需核验。
旧均值权重的 SHA256 为 `2cdd28518d828e234ae10d5f313b37af516969b77afad76661aaf0918b8a8e59`。
较早日期的模型和补偿实验也保留；比较时通过推理路径关闭附加补偿。

新模型使用独立版本目录 `models/w3/right/joint_lstm_observer_20261028_v1`，
不会覆盖旧模型。默认观察入口通过 `config/w3/right/observer_active.json` 选择新目录。
激活记录包含旧配置备份、模型哈希和回退信息。

## 两个窗口

两个估计器订阅同一份右臂 100 Hz 数据，保留相同的输入新鲜度、适用范围和暖机检查。
只有满足有效性检查的输出才进入波形。8082 是旧共享 LSTM，8083 是新独立 LSTM。

在 `/home/venom/factr2` 的两个终端分别执行：

```bash
ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 \
FASTRTPS_DEFAULT_PROFILES_FILE=/home/venom/factr2/config/w3/dds_loopback.xml \
bash scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/observe_w3_shared_baseline.py --port 8082
```

```bash
ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 \
FASTRTPS_DEFAULT_PROFILES_FILE=/home/venom/factr2/config/w3/dds_loopback.xml \
bash scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/observe_w3_joint_lstm.py --port 8083
```

执行前检查是否已有同类窗口运行，避免重复估计器和端口占用。实际机器人驱动、健康状态
和只读适配器须已经提供数据；此处命令不启动驱动或使能电机。

原始残差均为 `反馈关节力矩 - 模型空载关节力矩预测`。显示曲线均经过 12 Hz 再 5 Hz
的因果低通，页面可展开原始残差。显示保留两位小数，计算与原始记录保持全精度。
旧 sigma 头不参与这次对比；两边没有学习的残差校正、死区或强制归零。

## 训练与重启恢复

每支 LSTM 读取全部 21 维状态的 50 帧历史，只预测一个关节，分支之间不共享权重。
使用完整 154,779 个训练窗口，单次实验种子 0，各分支独立优化器、验证最优权重和早停。
原验证及测试记录的划分保持不变。

主机重启时，J2/J4/J5/J6/J7 已完整结束；恢复任务直接复用这些分支。
当时 J1/J3 只保存了最佳模型，缺少 AdamW 状态，因此从相同初始化重跑，核对前 39 轮
训练/验证损失，并在原最佳轮次严格比较权重；没有以重置优化器冒充精确续训。
新加入的 `resume.pt` 每轮保存模型、优化器、早停状态、随机状态和数据加载顺序。
恢复测试要求中断恢复后的最终权重与连续训练严格一致。

恢复过程的来源、文件哈希及旧版源码保存在 `reboot_recovery.json`、各运行目录的
`recovery_provenance.json` 和 `source_snapshot_pre_reboot/`。

## 如何理解对比结果

训练冻结后才运行独立验证/测试记录上的比较。每个 episode 重置滤波，并统一排除开头
一秒；同时报告原始误差及同滤波后的 RMS，避免把平滑效果当成预测精度改善。
`old` 分组是沿用记录（低速与此前较小范围高速），`high` 是扩大范围高速记录。

这次还改变了初始化、完整数据量、训练目标和轮数，因此对比反映新完整训练方案的效果，
不能只归因于独立 LSTM。测试记录历史上已查看过，属于独立记录的回顾评估，不能称为新盲测。
空载残差降低也不等于真实接触力矩准确，真实接触仍需独立标定或已知外力验证。

最终逐关节指标、训练曲线、测试波形和在线数据检查保存在
`log/joint_lstm_retrain_20261028_v1/REPORT.md`、`training_curves.png`、
`heldout_waveforms.png`、`live_verification.json`。
