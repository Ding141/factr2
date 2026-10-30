# 最新空载运动结果对应的论文与改进路线

检索日期：2026-10-16，Asia/Shanghai。用户在本轮明确确认此次运动为空载。本文件补充此前综述，结合新模型真机结果讨论方案；本轮没有修改在线估计或控制器。

## 本机证据及解释边界

最新实验：`log/targeted_force_observation_20261016/REPORT.md`、`analysis.json`。20 个实验块完成，包含七关节探测、多关节连续运动及首末各 10 秒静置，原生目标速度峰值约 5.25/6.78°/s，不能将 6/10°/s 档位上限当作实测速度。静止统计采用各静置段最后 5 秒。

| 条件 | 有效末端输出比例 | 力模长均值 N | P95 N | 峰值 N |
| --- | ---: | ---: | ---: | ---: |
| 运动前静止 | 98.6% | 0.191 | 0.305 | 0.421 |
| 运动 | 70.3% | 1.117 | 3.507 | 9.833 |
| 回位后静止 | 100.0% | 0.796 | 1.015 | 1.260 |

力统计仅覆盖有效末端输出。运动中关节推理持续有效，末端拒绝原因均为 `inconsistent_with_tool_contact`，本次没有记录到 `near_singular` 拒绝。目标回到原值，但实际静止姿态的最大均值变化为 1.232°，因此这不是严格同一实际姿态下的摩擦迟滞单因素实验。

空载、无接触时，真实接触外力为零，驱动电机仍承担重力、惯性和摩擦。估计残差可写作：

`tau_residual = tau_external + free_torque_prediction_error + measurement_error`。

当前记录支持持续偏差、动态预测误差及波动同时存在；不能将这些残差全部认作随机白噪声，也不能从空载日志单独证明其全部来自摩擦、温度或采样延迟。

此前 `docs/research/SIGNAL_AUDIT_20261016.md` 已核对单位、原生目标及训练侧冻结相位参数，并在另一日志验证：没有发现度/弧度倍数错误；大固定时间平移缺乏支持。零附近速度编码有约 0.28、0.63、0.70°/s 量级的半步偏差，不能直接等同于真实微动。当前 EMA alpha=0.2、100 Hz 的低频群延迟约 40 ms，直流增益为 1，持续偏差不会因 EMA 消失。此前离线居中 Savitzky–Golay 仅用于诊断，不是可直接上线的因果滤波器。

## 1. 停止后的偏置：运动历史和摩擦内部状态

Shilin Shan, Quang-Cuong Pham. **Sensorless Estimation of Contact Using Deep-Learning for Human-Robot Interaction**, ICRA 2024；核对作者全文 v2。

[全文](https://arxiv.org/html/2309.16219v2) · [接收信息](https://arxiv.org/abs/2309.16219)

论文针对动静切换及零速摩擦迟滞，提出 Motion Discriminator（MD）：速度超过相应阈值时更新运动记忆，低于阈值时保留历史，并使用 HRDL 学习。重点读 III-B、IV-A2、V；其消融中简单延长 LSTM 窗口并非持续改善。

本机候选：在保留原版 NEXT 基线的同时，训练 NEXT+MD 或 HRDL-MD 对照。NEXT 当前 50 帧/100 Hz 约为 0.5 秒，静置几秒后窗口不再直接包含此前运动；加入长时运动状态能否解释返回差，需要验证。

迁移要点：阈值按本机训练侧静止分布、位置变化及量化水平选择。论文中的低速度阈值不能直接复制，否则零速量化偏差也可能不断更新 MD。MD 应先在完整父会话上因果计算，再切训练片段；训练、验证、测试各自维护状态，不能跨会话继承。历史缺失、断流或重启需显式处理未知状态。该方案是 FACTR2 的扩展。

## 2. 输入噪声：从量化位置估计速度

P. R. Bélanger, P. Dobrovolny, A. Helmy, X. Zhang. **Estimation of Angular Velocity and Acceleration from Shaft-Encoder Measurements**, IJRR 17(11), 1998. DOI: 10.1177/027836499801701107。

[出版社摘要](https://journals.sagepub.com/doi/10.1177/027836499801701107)

研究以 Kalman 方法从量化编码器测量估计速度和加速度，讨论高采样率下有限差分的退化；本次核对摘要，未取得完整正文。可作为因果速度观测器的依据，不能据此保证本机性能。

本机候选：比较原始速度反馈与因果状态估计后的速度。估计器参数仅在训练会话拟合/冻结，离线与在线使用相同递推和初始化；若改变网络输入，必须重建特征、归一化并重新训练。禁止仅在部署端临时平滑速度；禁止把控制目标速度当作实际速度。保留原始流供审计，并单独验证引入的动态滞后。

## 3. 温度及摩擦变化

Maged Iskandar, Sebastian Wolf. **Dynamic friction model with thermal and load dependency: modeling, compensation, and external force estimation**, ICRA 2019, 7367–7373. DOI: 10.1109/ICRA.2019.8794406。

[DLR 作者记录及全文入口](https://elib.dlr.de/128180/)

模型包含温度、速度和载荷依赖，可与 LuGre/GMS 动态摩擦状态结合，在谐波减速器试验台验证。

本机候选：先用同路径、同控制条件进行专门冷热对照。已有 CAN 名义温度可作为待验证特征，但不等于经校准的关节摩擦温度；目前数据的 thermal 标签为 unclassified，不能宣称已完成冷热机辨识。建立热状态输入或动态摩擦分支需新增独立验证。

## 4. 保留基础模型，学习条件相关补偿

Philipp Scholl et al. **Learning-based adaption of robotic friction models**, Robotics and Computer-Integrated Manufacturing 89 (2024), 102780. DOI: 10.1016/j.rcim.2024.102780。

[DLR 作者记录、全文及勘误入口](https://elib.dlr.de/204073/)

论文使用基础摩擦网络加残差网络，适应非对称摩擦等新工况。本机可借鉴离线条件补偿：自由力矩预测为 `NEXT + correction(q, v, q_cmd-q, motion_memory[, thermal_state])`。该补偿学习的是综合预测误差，不能预先认定为纯摩擦。只用明确无接触训练数据拟合，保留独立验证/测试；不在真实接触过程中在线追踪零点。

## 5. 近零速摩擦不确定性与末端换算

Magnus Linderoth, Andreas Stolt, Anders Robertsson, Rolf Johansson. **Robotic Force Estimation Using Motor Torques and Modeling of Low Velocity Friction Disturbances**, IROS 2013, 3550–3556。

[作者所在大学记录及全文入口](https://portal.research.lu.se/en/publications/robotic-force-estimation-using-motor-torques-and-modeling-of-low-/)

论文显式考虑库仑摩擦的速度相关不确定性，以实时凸优化融合多关节信息，并用力传感器验证。

本机候选：在自由力矩均值预测改善后，研究按训练侧关节不确定性加权的末端求解；某关节近零速不可靠时降低其权重，并输出置信度/有效性。这是借鉴其不确定性处理思想，不是当前阻尼最小二乘的原样复现。正则化不能恢复不可观测分量，也不能替代动力学误差补偿。保留现有拒绝状态，不能以放宽门限提高表面可用率。

## 6. 残差学习与自适应观测器

**K-VARK: Kernelized Variance-Aware Residual Kalman Filter for Sensorless Force Estimation in Collaborative Robots**, arXiv:2512.13009v2，2026-06-16；按预印本列示。

[作者全文](https://arxiv.org/html/2512.13009v2)

方法联合学习残差均值及状态相关方差，并设计自适应扰动 Kalman 滤波。作者明确列出近零速、静摩擦和温漂方面的不足。其接触消融还表明，单独增加测量噪声适应可能把接触突变当噪声而造成更大滞后。因此不能将本方法简化成对现有末端数值增加一个 Kalman 滤波器。

## 原版复现边界与下一步

[FACTR2 作者全文，当前可访问 v1，附录 A.1–A.3](https://arxiv.org/html/2606.12406v1) 使用 stateless 滑窗 LSTM、50 帧位置/速度/跟踪误差。MD、温度、条件补偿和概率观测器均应单列为扩展实验。

推荐按阶段隔离变量：原版 NEXT → NEXT+因果速度估计 → NEXT+MD → 两者组合。再考虑热状态/残差补偿及不确定性观测器，每一步均保留前一步对照。已有定向数据可支持部分对照，不应假定采集过启停数据就已具备记忆状态。

模型训练仍预测实际无接触驱动力矩，不能把其标签改成零。可以在训练侧按静止、换向、单关节、联动及速度组审查/平衡样本；这是本机提出的实验设计，不是上述论文都规定的步骤。

选择模型时同时比较每关节偏置、动态 RMSE/P95、相同实际姿态下的返回差、末端有效率及拒绝原因；最终另采已知载荷/真实接触与完全卸载数据验证持续力保真和恢复。当前空载日志已用于诊断，不能反复据此选滤波参数后仍把它当成未触及的独立测试集。所有延迟/尺度拟合继续限定训练会话，独立日志只验证冻结设置。

数值目标应是偏置和噪声落入预先规定的零附近误差预算，并保持小外力、持续外力的响应。固定单姿态扣零不能补偿跨姿态/温度/历史的误差；连续自动扣零或显示死区可能掩盖真实持续力。低通滤波可降低部分波动，但不能消除直流偏置，且应报告接触响应延迟。
