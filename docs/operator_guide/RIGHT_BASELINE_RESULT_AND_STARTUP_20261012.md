# 2026-10-12 右臂采集、训练及只读诊断结果

## 保存内容与当前结论

本次新增完整空载协议规划、分组补采、独立会话数据准备、候选训练与空载/卸载验收。修复 100 Hz 录制断流分段、重复反馈诊断误报警、实时参数读取与运动审计写入阻塞。训练损失可以按物理力矩 MSE 计算；归一化仍只来自训练集。

候选保存在 `/home/venom/factr2/models/w3/right/baseline_candidate_20261012/`，约 1 MB，包含权重、归一化、冻结训练配置、metadata、metrics、冻结验收配置、失败报告、控制快照及数据独立性/协议覆盖审计。原始 H5、ROSbag、本机标定和运行日志继续保留在本机，不作为本次 Git 文件提交。旧模型 `models/w3/right/c2_20261010` 保留。

训练于北京时间 01:37 完成：22 轮早停，以验证损失选第 2 轮，训练耗时约 32.4 分钟。验证总体 RMSE 0.199203 N·m；J1/J2/J4 为 0.286377/0.279488/0.208575 N·m，超过每关节 0.2 N·m 门限。某静止姿态 J1 平均残差达 1.274124 N·m；从不同方向返回同一目标姿态的 J1 基线差达 0.780952 N·m，超过 0.1 N·m。

**独立空载验证失败，deployment_ready=false。** 依照冻结门禁，最终测试集未评估，人工手推卸载验收未执行。退出码 1 是验收拒绝，训练本身正常完成。选定末端任务几何有效率为 100%，不表示所有姿态都无奇异点；部分静止末端换算仍因残差不符合末端集中载荷假设而无效。没有更改验收阈值或自动扣零。

用户随后明确要求部署此候选用于观察与排查，已启动只读观测。100 Hz、50 帧历史和健康/范围/奇异/拟合门禁均保留。运行状态 valid 只表示输入与计算门禁通过，不能代替精度验收。

## 数据协议与恢复

详细流程见 `/home/venom/factr2/docs/operator_guide/RIGHT_BASELINE_PROTOCOL_20261011.md`。train/val/test 分别由独立原始会话构成；完整逐关节慢/快扫动、三组条件良好末端任务、双向回位与静止段均检查覆盖，控制参数一致。历史窗口不跨原始 episode、来源会话或数据角色。

采集中断时只复用质量合格、完整连续的协议组；失败块不冒充完成。验证续采按照原先冻结的路径、随机种子和完整协议进行，不依据模型分数挑选。200 ms 调度、250 ms 反馈新鲜度、3°跟踪限制保留。

三份已关闭旧 ROSbag 使用 `.db3.zst` 无损压缩，恢复命令和原始/压缩 SHA256 保存在对应 bag 目录的 `COMPRESSED_STORAGE.json`。读取旧 bag 前先恢复 SQLite；H5 准备/训练无需恢复 bag。不得压缩或读取正在写入的 bag。

## 启动右臂连接与模型

电脑与机器人连接、CAN1 配置、支撑和 A/B 启动流程沿用 `/home/venom/factr2/docs/operator_guide/RIGHT_ARM_STARTUP_20261010.txt`。本候选仅适用于右臂、无夹爪、空载和本机对应标定/控制条件。已经存在 A/B 时保留它们；启动观测无需重启控制或回零。

先检查是否已有 C/D/E/F 观测进程和端口，已有相同服务时直接使用，不能重复启动。以下命令不发送运动目标，也不使能或切换控制模式。

终端 C，系统 ROS 只读健康器：

```bash
cd /home/venom/dual_arm_robot
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
/usr/bin/python3.10 /home/venom/factr2/factr2_w3_adapter/tools/w3_health_monitor.py
```

终端 D，100 Hz real 适配器：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh ros2 launch factr2_w3_adapter adapter.launch.py side:=right profile:=real publish_hz:=100
```

终端 E，显式加载未经验收的候选模型：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh ros2 launch factr2_next readonly.launch.py \
  side:=right \
  inference_config:=/home/venom/factr2/config/w3/right/inference_baseline_candidate_20261012.yaml \
  web_config:=/home/venom/factr2/config/w3/right/visualize.yaml
```

终端 F，末端等效力与力矩：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh python scripts/w3_endpoint_wrench.py \
  --config config/w3/right/endpoint_wrench_baseline_candidate_20261012.yaml
```

打开 `http://127.0.0.1:8081/` 关节力矩和 `http://127.0.0.1:8082/` 末端窗口。末端上图 Mx/My/Mz 单位 N·m，下图 Fx/Fy/Fz 单位 N，横轴秒；作用点及坐标轴为 right_attachment_point。末端负载假设不适用于任意连杆/多点接触。无效区间留空，不能读作零外力。

停止观测只在 C/D/E/F 对应终端 Ctrl+C，保持 A/B；结束重力补偿或重新连接前需要可靠支撑。默认旧推理配置 `inference_smooth_100hz.yaml` 未改为本候选，需用上面的显式配置才加载本版。

checkpoint metadata 和冻结配置中的绝对路径保留真实来源，不修改它们伪造标定或数据条件。目录变化时调整启动/末端配置路径，核验实际模型、标定、URDF 和控制快照。原始训练数据不必存在才能推理；本机标定必须另外恢复并核验。

## 本次运动观察

用户确认净空并要求大幅快速测试后，执行 J2/J4 各 ±20°往返，再做 ±15°联动。使用 sin³ 平滑曲线、14 秒周期和连续绝对时间窗口，含前后静止总计约 52.8 秒；规划峰值速度 10.3646°/秒、加速度 12.0854°/秒²。离散 URDF 碰撞检查通过，但支架、线缆及现场姿态不属于该模型认证范围。

5 个阶段均完成，最大实测跟踪误差 2.2651°，最大调度间隔 5.0081 ms；未触发 3°/200 ms/250 ms 保护。回到 `[0,50,0,-55,0,0,0]°` 控制目标，七电机正常、位置保持/重力补偿保持 active、观测保持 100 Hz。这是操作者观察测试，没有加入训练数据，也不构成外力精度验收。

运动规划/完成审计本机保存在 `/home/venom/factr2/log/fast_motion_observation_20261012_015810/`。候选训练、部署和最终验收原始记录在 `/home/venom/factr2/log/baseline_campaign_20261012_000958_resume4/`。

## 提交前验证

相关离线自动测试 117 项通过，耗时 160.86 秒，覆盖独立数据协议与续采、空载/卸载评估、异步审计、平滑运动与调度保护、适配器健康语义、录制质量、训练契约与末端换算。未初始化真实运动节点。归档 checkpoint 与新推理配置通过加载/配置检查；权重 SHA256 与训练输出相同，失败验收结论保留。

旧的手工资源基准脚本 `scripts/test_training_resources.py` 依赖已经不存在的 `reports/generated/4/left/continuous/continuous.json`，不属于此次可运行回归集合，未重建历史 fixture 或伪称该资源基准通过。
