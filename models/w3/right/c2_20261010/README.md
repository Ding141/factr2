# 2026-10-10 右臂 C2 采集模型

实际真机训练checkpoint副本，五个必需文件按原始字节保存，通过metadata中的artifact_sha256验证。

输入为[q,qdot,q_cmd-q]，右臂7关节，100Hz，历史50帧。模型SHA及原始数据来源见ARCHIVE_MANIFEST.json；TRIAL_REPORT.md与trial_summary.json包含指标。

训练80轮，选第78轮，重复动作测试总体RMSE约0.054053 N·m。未验证独立接触精度；组合姿态已有明显空载偏置，不能作为已校准传感器或控制闭环输入。零接触诊断见BASELINE_DIAGNOSTIC.md。

启动手册：[右臂连接与模型启动](../../../../docs/operator_guide/RIGHT_ARM_STARTUP_20261010.md)。

原始H5、bag、逐帧评估CSV未提交Git；原始训练配置中的绝对路径仅记录训练来源，推理加载不需要该H5存在。重新训练仍需恢复原始录制。标定文件不在该checkpoint中；迁移机器要恢复本机标定并核验工具/负载/控制条件。
