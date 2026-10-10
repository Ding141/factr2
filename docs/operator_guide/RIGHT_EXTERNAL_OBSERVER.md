# 当前右臂只读外力矩与末端观测

更新：2026-10-10。当前模型为 `models/w3/right/c2_20261010`，100 Hz、右臂、50帧历史、ROS域74。

完整连接、控制、模型和网页的命令，以 [详细启动手册](RIGHT_ARM_STARTUP_20261010.md) 为准；本文保留接口和换算说明。旧小范围模型未覆盖的J4零位范围限制不再作为新模型范围说明，但近奇异几何限制仍存在。

本次重复动作测试关节力矩RMSE约0.054 N·m；未进行独立接触精度认证，某些组合姿态已有明显空载残差。估计在电脑运行，不发送运动命令。七关节页面8081，末端页面8082。

## ROS 输出

| Topic | 内容 |
| --- | --- |
| `/next/right/external_joint_torque` | 滤波外力矩估计 |
| `/next/right/external_joint_torque/raw` | 未滤波外力矩估计 |
| `/next/right/free_joint_torque_pred` | 无接触关节力矩预测 |
| `/next/right/status` | 推理状态、范围提示、频率、延迟和丢帧计数 |

三个力矩 Topic 使用 `sensor_msgs/JointState`，七个力矩数值写在 **position** 字段，单位仍为 **N·m**。这是现有 NEXT 接口的约定，不能当作角度数据订阅。

## 末端力与力矩页面

页面：<http://127.0.0.1:8082/>，终端 F；代码 `scripts/w3_endpoint_wrench.py`，配置 `config/w3/right/endpoint_wrench.yaml`。

上方为末端 `Mx、My、Mz` 曲线，单位 N·m；下方为 `Fx、Fy、Fz`，单位 N；横轴均为节点启动后的时间（秒）。力矩的作用点和六个分量的坐标轴都使用 `right_attachment_point`，即目前空载右臂的末端安装点。不是绕机器人底座原点的力矩。

本功能由关节残差换算等效末端负载：`tau_NEXT ≈ J(q).T @ [F; M]`。使用七关节/六分量的阻尼最小二乘；先按 0.4 m 特征长度对线速度雅可比缩放，再进行奇异值分解，默认阻尼为 0.005。坐标转换仅旋转力和力矩，保留末端作用点。符号沿用 NEXT 残差，尚未通过已知方向加载实验验证；不要直接宣称是环境作用于机器人方向的传感器测量值。

前提是外部载荷作用在末端。若接触发生在臂杆、多点或其他位置，计算出的等效末端量不一定代表真实末端接触。拟合残差检查不能定位接触点，也不能证明单点假设成立。

只有以下检查全部通过才显示数值并发布末端估计：实时机器人描述与部署配置的 SHA256 相符，关节位置和关节残差时间戳精确匹配且新鲜，NEXT 状态为 `valid`，当前位置位于配置采集范围内，缩放雅可比条件数不超过 100，末端接触拟合相对残差不超过 0.35。这些是诊断门限，不是接触精度认证。不可用区间画成空白，数值显示“—”，不伪装成零外力。

近全伸直零位仍可能因雅可比接近奇异而不可用。最新逐关节范围见启动手册；范围通过不等于组合姿态充分覆盖，空载偏置仍待解决。

在已有 C、D、E 正常运行时，另一个终端执行：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1
scripts/next.sh python scripts/w3_endpoint_wrench.py --config config/w3/right/endpoint_wrench.yaml
```

末端 Topic `/next/right/endpoint_wrench` 使用标准 `geometry_msgs/WrenchStamped`，`force` 为 N，`torque` 为 N·m，`header.frame_id=right_attachment_point`。不可用时不发布数值，下游必须同时检查 `/next/right/endpoint_wrench/status` 和消息新鲜度。

验证包括：已知力/力矩的恢复、特征长度单位一致性、真实机器人几何的线性/角速度雅可比差分、虚功一致性、奇异点与非末端拟合残差拒绝、坐标轴旋转，以及隔离 ROS 域 172 的流测试。隔离测试输入 `[2, -1, 0.5] N` 和 `[0.2, -0.3, 0.1] N·m`，发布 450 帧后验证停止输入会清空数值并留空曲线；仿真数据没有进入真机域 74。该测试只验证换算和数据链路，不代表真实机器人末端估计精度。
