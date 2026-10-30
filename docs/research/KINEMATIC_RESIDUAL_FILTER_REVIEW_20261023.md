# 运动残差补偿与因果滤波：文献依据和本机实现

检索日期：2026-10-23。以下区分论文结论与本机工程选择；没有把其他机器人的指标当作本机性能。

|研究|已核实内容与借鉴|本次实现的边界|
|---|---|---|
|Wang 等，Robotica，2026-10-01，SNGMO|增强 Stribeck 模型、随速度调整观测器带宽/阻尼，处理低速与换向残差；说明残差抑制和碰撞响应需要同时权衡。|核实出版社摘要，全文访问受限。未逐式复现 SNGMO；本次使用固定、可量化延迟的因果滤波，没有直接照搬自适应阈值。|
|Zhou 等，Control Engineering Practice，2026，107140；2026-07-11 提前上线|将动力学残差补偿与剩余力/力矩残差学习分阶段，分别处理建模与电流到关节力矩映射偏差。|核实作者机构摘要及上线日期，纸本卷期为 2026 年 11 月。本次只学习空载关节残差，缺少接触真值，不复现接触 wrench 校准阶段。|
|Sorrentino 等，RA-L，2025，PINN＋UKF|全文方法部分使用位置、速度、加速度状态的卡尔曼滤波平滑导数，再结合摩擦学习和力矩估计；指出速度/编码器分辨率的影响。|本机无独立电机/关节双编码器及其传感器组合，只借鉴位置观测的常加速度 KF，不声称完整 UKF 或 PINN 复现。|
|Scholl 等，RCIM，2024；arXiv 修订 2025-05-09|冻结基础摩擦模型，再以少量数据学习剩余误差，适应不对称摩擦。|本机基础模型为 NEXT，新增可审计的 ridge 残差回归；数据有限，保留局部适用范围。|

原始来源：

- [Robotica：Robotic collision detection based on a speed-adaptive nonlinear gain momentum observer](https://www.cambridge.org/core/journals/robotica/article/abs/robotic-collision-detection-based-on-a-speedadaptive-nonlinear-gain-momentum-observer/03667EA2B71EB0762140153A3F384766)
- [HKUST：Sensorless contact forces estimation for robotic manipulators using dual-stage temporal-residual neural network compensation](https://researchportal.hkust.edu.hk/en/publications/sensorless-contact-forces-estimation-for-robotic-manipulators-usi/)
- [Sorrentino 等全文，尤其 III-A 中 KF 方程](https://arxiv.org/html/2507.10105v1)
- [Scholl 等全文](https://arxiv.org/html/2310.16688v2)
- [FACTR 2 官方方法说明](https://jasonjzliu.com/factr2/)

## 对当前数据的判断

此前空载 J1 往复残差与方向有强关联，EMA 对 RMS 改善不足 1%；不能仅靠更强低通解决。速度/加速度相关项可以学习，但相关性并不唯一指向惯性：摩擦、跟踪、输入量化、姿态与历史相互耦合，NEXT 已经学到部分动力学，不能再机械地减去完整 M(q)q̈ 而重复补偿。

本次采用 `c(filtered_motion)` 修正原 NEXT 剩余误差，包含因果估计的实际/指令速度、平滑方向函数、速度绝对值和可选加速度，不使用当前实测力矩或残差作特征。它是自由运动误差模型，系数不是已辨识的真实惯量或电流标定参数。

输入预处理只作用于新增分支：q 与 q_cmd 使用位置观测的常加速度 KF；量化速度反馈另做 8 Hz 低通诊断。原 LSTM 保持训练时的 21 维输入口径，避免未经重新训练就改变输入分布。

实测力矩与原 LSTM 自由预测先经过相同的 12 Hz 因果低通，再减去离线残差预测，最终输出通过 5 Hz 因果低通。两条力矩支路必须具有相同滤波时序，否则会人为制造加减速残差。全部按源时间推进，超过 40 ms 断流、时间回退或无效数据重置；无未来帧、居中差分或零相位离线滤波。

补偿在训练数据不足的特征列上固定零权重，逐关节按验证改善和历史回退条件启用；超出所学局部姿态/特征范围时回退。无常数截距、无在线拟合、无静止自动归零、不裁剪外力输出。原 sigma 没有针对新残差重新校准，不移作新输出的置信带。

## 验证要求与目前结论

训练/选择/评估按完整记录与完整往复周期隔开，不随机拆相邻帧；比较原始、只滤波、仅速度、速度＋加速度四类结果。第 3 次往复此前已经查看，仍称回顾性留出，不称新盲测。旧姿态的补偿回退并不证明跨姿态补偿有效。

固定运动状态的合成阶跃用于检查补偿不随力矩输入自适应衰减，并量化 80 ms 左右的 90% 上升时间。它不替代真实接触实验：真实接触还会改变运动状态，平滑也会削弱短脉冲。因而新分支仅作观测，不接入碰撞保护或力控。

本机选择与逐关节结果见 `data/right/kinematic_compensation_20261023_v2/REPORT_ZH.md`。主要改善来自速度/方向，加速度额外收益较小；不能因为输入加了加速度就声称已经解决惯性误差。
