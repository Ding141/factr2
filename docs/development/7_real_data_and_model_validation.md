# 7. 真机自由运动采集、训练与独立模型验收

## 目标、依赖与执行条件

在 W3 真机上采集可靠的无接触运动数据，为左右臂分别训练并评估自由空间力矩模型。依赖 [2](2_w3_command_state.md)、[3](3_adapter_and_health.md)、[4](4_recording_and_data_quality.md)、[5](5_training_and_offline_evaluation.md) 已验收。第 6 部分在线功能可以独立推进。

这是实验阶段，需要现场操作者和正常的 W3 真机。当前请求只是拆分开发任务，不授权未来 agent 自主使能、发轨迹或接触机械臂。开发 agent 先完成采集脚本/配置/覆盖清单与离线 dry-run，真机操作按用户后续指令、既有 W3 流程与现场确认执行。缺硬件时，交付这些准备工作，并将真机条目标记 BLOCKED。

## 先读与可修改范围

- 整体规划的数据采集、训练、风险章节；第 4/5 部分验收报告与工具文档。
- W3 `src/README.md`、`src/docs/CONTROL_PIPELINE.md`、`src/ieir_controllers/scripts/teach_replay.py`、`src/ieir_controllers/config/dual_arm_controllers.yaml` 和实际 launch 产生的运行配置。
- W3 `src/ieir_controllers/scripts/teleop_force_observer.py` 可作 raw torque/重力/摩擦残差诊断，其结果不是外力真值或 NEXT 训练标签。
- 修改范围：FACTR2 采集方案/配置、元数据、数据清单、训练 run、评估报告。W3 启动与轨迹用已有工具；若需新轨迹格式转换器，单独实现并 dry-run，不混入 NEXT 只读节点。不调整控制器增益、摩擦、标定来“让模型误差好看”。

## 实施步骤

1. 建立 `docs/experiments/free_motion_protocol.md`、`config/w3/collection_matrix.yaml` 和离线轨迹生成/校验工具（例如 `scripts/generate_collection_trajectories.py`）。默认只生成文件/预览，不发布命令。若使用 teach_replay，核对实际 YAML 格式与 time_scale/ramp 行为；NEXT H5 不是 W3 回放轨迹。
2. 每侧覆盖多个姿态区域、单关节/多关节、正反方向、低/中/计划工作速度、加減速/保持段。幅度和速度从真实 URDF/现场可用工作范围确定；运动前检查自碰/互碰/线缆，不把线性插值当碰撞规划。
3. 先单侧小幅、低速、无接触 pilot，确认 q/qdot/q_cmd/tau 的方向、单位和时间差；实测 command_state 与控制器采用目标一致。位置控制器必须 active，gravity 反馈只作诊断；不以原始稀疏 JointTrajectory 点代替 command_state。
4. 启动分工：W3 shell 负责既有 bridge/control 和独立健康检查/rosbag；NEXT shell 负责 real profile adapter 和 recorder；训练可离线。保持 profile 工具/负载/标定/增益/前馈版本一致，记录实际有效值及 SHA256，不仅引用仓库默认。
5. 所有训练/验证/测试 episode 为无外部接触的自由运动。手推、碰撞、拖线、异常 motor 健康、切控制模式/工具期间的段标记并排除，保留原始审计，不悄悄删几行拼回窗口。
6. 最小起点：每侧至少 3 个不同采集 session 分配 train/val/test，每个至少覆盖上述运动类别；跨冷机/热机条件，记录温度来源。若没有温度接口，以冷机/预热时长等可观测条件记录“温度未知”，不伪造温度值。
7. 采集时长由覆盖与误差收敛决定，不能凭“采够几分钟”宣告模型合格。建议先 pilot 再补不足区域，每批运行 strict checker，只有 accepted 数据进入 manifest。最终报告逐项列出覆盖数量、时长和未覆盖区域。
8. 在最终 test 前冻结 `config/w3/{side}/acceptance.yaml`、split-manifest、训练超参数/种子集合、模型选择规则。训练/调参/阈值拟合用 train+val；最终 test session 保留。反复查看 test 后再调参，应视为它已变成 val，并重采新的独立 test。
9. 用默认 stateless LSTM 做正式训练，至少评估 3 个声明的随机种子，报告 validation 差异，以事先声明的 val 规则选择部署 run。没有证据时不切换模型类型或把两臂合成一个网络。
10. 在完整独立 test 上执行第 5 部分评估，保存每关节指标、各速度/姿态/session 分组、残差频谱与原始时序图，标记分布外运动/低覆盖条件。记录 peak memory / device / 总时间与最终 artifact 哈希。

## 如何冻结模型门槛

无 F/T 传感器时不存在可凭空填写的绝对外力真值。本阶段指标评估的是**无接触自由力矩预测误差**，不是接触力精度。

验收配置必须包含 7 个关节各自的 `max_rmse_nm` 和 `max_abs_bias_nm`。根据工作需求、pilot 信号波动和协议量化/噪声确定可接受预算，写明确定方法，再冻结。pilot 本身只可进入 train/val，不作为最终 test。空预算、全部无穷或看完 test 后放宽预算都不能算通过。

建议第一版相对门槛（是工程起点，不是论文给定精度）：

- 对相同 test 的训练均值常数预测 baseline，整体 RMSE 至少降低 20%（模型/baseline <=0.8）。
- 每个有效动态关节 RMSE 不差于 baseline（比值<=1.0），并同时满足冻结的绝对 RMSE/bias 预算。
- 近常量关节的 baseline 可能接近 0；由 pilot 量化/噪声确定并冻结 `sigma_floor_nm`，低于此 floor 的关节按绝对预算验收，不能除以零或用巨大 floor 掩盖失败。
- 所有预声明关键速度/姿态/session 分组满足预算。平均误差合格不能掩盖某个关键区域系统性超标。

若这些起点确实不符合实际任务，应在最终 test 前写出替代的有限、量化门槛及依据。门槛文件、时间、哈希与最终测试报告共同归档。未满足门槛时报告失败和需要补数据的区域，不把“训练能运行”标成可部署模型。

## 验收标准

| 编号 | 验收项 | 必须证据 |
|---|---|---|
| DATA-01 | 真机接口 pilot | command_state/四流 joint names、单位、方向、source skew、频率；raw health、controller active 与运行配置记录 |
| DATA-02 | 数据合法 | accepted H5 全部 strict PASS，四流[N,7]、>=45 Hz、无重复/回退/NaN，连续段不跨>40 ms缺口 |
| DATA-03 | 来源与污染 | 每episode 工具/负载/标定/温升条件/轨迹/接触标记完整，排除异常段的理由与原始审计关联 |
| DATA-04 | 分割隔离 | 每侧至少3 session，train/val/test来源哈希无交集；验证/测试运动和条件覆盖清单齐全 |
| MODEL-01 | 训练可复现 | 至少3种子、resolved config、actual device、best_epoch、完整5项 run artifact 和 metadata |
| MODEL-02 | 独立误差 | 每关节/关键分组达到事先冻结预算；baseline比较达到声明门槛，报告sample counts与单位 |
| MODEL-03 | 部署身份 | 左右各一个合格run，side/order/H/rate/contract/工具/标定完全对应；权重及metadata哈希可核验 |
| MODEL-04 | 限制透明 | 明确无力传感器绝对真值、未覆盖区域/负载/温升条件、失败组与部署适用范围 |

左右可以逐侧通过：`PASS_HARDWARE(left)` 不等于双臂完成。若只能得到一个合格侧，交给第 8 部分做该侧试验，另一侧仍是待完成项；项目最终验收要求两侧都有模型和证据。

## 交付与交接

提交实验协议、覆盖清单、已冻结 acceptance 配置、精简 split-manifest/结果摘要和 `docs/acceptance/7_real_data_and_model_validation.md`。数据/权重/完整图表留在忽略目录，报告记录绝对路径、SHA256 与恢复步骤。为第 8 部分提供每侧 checkpoint、适用工具/负载/标定版本、无接触残差分布、禁止外推的区域，以及尚未用于模型选择的接触实验方案。
