# 空载误报补偿改进（2026-10-27）

本轮更新只影响检测输出。没有启动机器人驱动、使能电机、切换控制模式或发送运动命令。当前主机的机器人数据源和检测服务均未运行，新模型尚未完成新实机运动复测。

七关节仍全部启用补偿。以训练/验证会话分开拟合，加入姿态、姿态与速度/加速度耦合特征、小型冻结MLP和连续速度权重。验证集只选择J1/J2更新；J3–J7保留原补偿。新模型超出支持范围时回退到原模型。输出不设死区、不自动归零，12Hz输入滤波和5Hz输出滤波保持不变；补偿不会写入电机控制指令。

上一轮33,294帧实机数据的离线回放，在正式段中间140秒的有效帧上：整体RMS 0.43328→0.37891N·m，减少12.55%；J1为0.74668→0.58736，J2为0.57420→0.48286，J3–J7逐点差小于3e-13N·m。回放数据没有用于本轮拟合或逐关节选择。

历史保留数据：低速原补偿0.09630→0.08394N·m，高速0.39094→0.36263N·m。低速仍高于仅滤波基线0.07725N·m，原低速MSE回归门槛仍未通过，不放宽门槛或宣称已经消除误报。默认仅为用户要求的只读研究观察，deployment_ready=false；旧测试集已经多次观察过，不是全新盲测。

36项数值与回归测试、2项隔离域ROS传输测试通过。固定运动学输入叠加持续0.3N·m阶跃，最终保留完整增量、90%响应约80ms；这不等于真实接触精度标定。在线/离线数学路径差约4e-14N·m。

默认配置在 `config/w3/right/compensation_active.json`，固定到模型目录 `data/right/validated_joint_compensation_20261026_v1` 及其SHA256。J1/J2采用新非线性模型，其余采用原补偿；原模型和未通过选择的实验模型均保留。

仅检查配置，不启动ROS节点：

```bash
cd /home/venom/factr2
bash scripts/next.sh python scripts/observe_w3_kinematic_compensation.py --check-only
```

在按原流程准备好真实机器人数据、适配器及冻结NEXT基础服务之后，启动只读补偿窗口：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/venom/factr2/config/w3/dds_loopback.xml
bash scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/observe_w3_kinematic_compensation.py
```

入口不传 `--experiment` 时使用新默认配置。显式指定原模型目录仍可作对照；避免同时启动多个同话题检测源。检查文件成功不代表机器人正在发布数据。

完整报告、七关节波形及证据：`log/compensation_improvement_20261027_v1/REPORT.md`。训练入口为 `scripts/fit_w3_pose_dynamic_compensation.py`、`scripts/fit_w3_nonlinear_compensation.py`；逐关节选择为 `scripts/select_w3_compensation_joints.py`；回放核查为 `scripts/replay_w3_compensation.py`。原NEXT均值、sigma及机械臂控制器均未修改。
