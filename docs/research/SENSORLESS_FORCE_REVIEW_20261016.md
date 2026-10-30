# 空载偏置与运动假外力：相关研究及改进建议

检索日期：2026-10-16（Asia/Shanghai）。本文是文献调研与实验建议；没有修改运行模型、门限或控制器，也没有执行机器人运动。

后续已完成信号审计、定向数据采集及原版 NEXT 初步重训。结合新模型空载运动结果的更新判断、速度量化处理文献与下一步路线见 [最新空载问题调研](UNLOADED_FORCE_REMEDIES_20261016.md)。下文保留首次调研时的实验状态，不代表后续工作尚未执行。

## 对当前问题的判断

现有记录优先支持自由运动力矩预测存在静态偏置和动态误差，再通过末端雅可比换算放大。摩擦迟滞、温度和物理采样时延仍是待验证假设，不能仅凭空载残差逐项确定。

本机依据：[2026-10-15 空载随机运动报告](/home/venom/factr2/log/random_free_test_20261015/attempt_03/REPORT.md)。运动时 J4 关节滤波残差 RMSE 为 0.204 Nm；有效末端 Fz 绝对值 P95 为 5.644 N；末端发布有效率为 69.88%。运动前后同一指令姿态的 J4 残差均值变化 0.161 Nm，但实测 J4 姿态均值也变化了 0.787°，因此这不是摩擦迟滞的单因素证明。无独立力传感器，零外力依据操作者确认的空载无接触条件。

当前试验模型的数据未通过项目自己的快速双向运动和部分静态姿态覆盖检查，且不是跨独立采集会话验收。这些项目门限不是下面论文统一规定的标准。见 [coverage audit](/home/venom/factr2/data/right/pose_trial_20261015/production_coverage_audit.json)。

## 优先阅读的论文

### 1. 静止后残留力：长期运动记忆

Shilin Shan, Quang-Cuong Pham. **Sensorless Estimation of Contact Using Deep-Learning for Human-Robot Interaction**. ICRA 2024；预印本初稿 2023，核对版本 v2，2024-03-06。

[论文全文](https://arxiv.org/html/2309.16219v2)

论文专门讨论零速摩擦迟滞和运动到静止的切换。Motion Discriminator（MD）按速度阈值更新历史特征，停止后保留先前运动信息；配合 HRDL 网络及含启停的数据改善静态估计。它的消融也表明，延长 LSTM 输入并非总能改善精度。重点读 III-B、IV-A2、V。

**本机建议（需要实验验证）**：保留 NEXT 基线，另训练加入 MD 的模型；用正反方向到达相同姿态、不同停留时间和多关节换向进行对照。训练、验证、部署必须一致地维护 MD 状态；断电重启或数据段截断时，应标记历史未知。增加 MD 属于对 FACTR2 的扩展，不能称为原版复现。

### 2. 热态与换向误差：带温度、负载的动态摩擦模型

Maged Iskandar, Sebastian Wolf. **Dynamic friction model with thermal and load dependency: modeling, compensation, and external force estimation**. ICRA 2019, 7367–7373. DOI: 10.1109/ICRA.2019.8794406。

[DLR 官方记录与全文](https://elib.dlr.de/128180/)

研究把温度、速度和负载依赖引入动态摩擦模型，并与 LuGre/GMS 内部状态结合，在谐波减速器试验台验证，包括外力矩估计。它说明只用瞬时速度的摩擦曲线可能不足；试验台结果不能直接作为整臂精度保证。

**本机建议**：先在相同路径和控制参数下比较冷机、热机、反向到位后的残差。如存在可重复热态关系，再考虑温度输入或经标定的热状态估计；单纯“运行了多久”只能是待验证代理量。负载相关补偿也需验证，不能把未知接触自动当成摩擦消除。

### 3. 保留现有模型：学习补偿残差

Philipp Scholl et al. **Learning-based adaption of robotic friction models**. Robotics and Computer-Integrated Manufacturing 89 (2024), 102780. DOI: 10.1016/j.rcim.2024.102780。

[DLR 官方记录与开放全文](https://elib.dlr.de/204073/)

作者保留基础摩擦估计器，再训练小网络学习新工况下的残差，检验非对称摩擦、换向和负载变化。实验中的少量数据适配结果只适用于其设置，不能据此保证本机几十秒即可完成标定。官方记录同时提供勘误入口。

**本机建议**：可对冻结的 NEXT 预测增加离线训练的偏差补偿分支，输入考虑姿态、跟踪误差和运动历史。应把它称为综合预测误差补偿；目前无法将本机残差确认为纯摩擦。只用明确无接触的数据拟合，并保留独立接触/卸载测试，避免将持续真外力学入零点。

### 4. 动态误差与噪声：辨识、学习和观测器结合

Sichao Liu, Lihui Wang, Xi Vincent Wang. **Sensorless force estimation for industrial robots using disturbance observer and neural learning of friction approximation**. Robotics and Computer-Integrated Manufacturing 71 (2021), 102168. DOI: 10.1016/j.rcim.2021.102168。

[出版社论文页](https://www.sciencedirect.com/science/article/pii/S0736584521000521)

出版社摘要描述了动力学参数辨识、神经网络摩擦近似与扰动卡尔曼观测器的组合，并在六自由度工业机器人数据上验证。本次核对了摘要，未取得完整正文，因此不报告具体误差或滤波参数。

**本机建议**：作为第二阶段对照方案，先改善动力学/摩擦预测，再设计观测器噪声模型。不能把在现有残差上简单添加一个卡尔曼滤波器等同于复现该方法，也不能预先保证优于 NEXT。

### 5. 用学习模型校正动量观测器

Daegyu Lim et al. **MOB-Net: Limb-modularized Uncertainty Torque Learning of Humanoids for Sensorless External Torque Estimation**. IJRR 2025, 44(1), 96–128. DOI: 10.1177/02783649241260428。

[论文全文 v2](https://arxiv.org/html/2402.11221v2)

MOB-Net 学习模型不确定力矩并校正动量观测器，同时利用输出均值和标准差处理碰撞检测。其对象为浮动基人形机器人；论文的腿部训练数据需要足底力传感器，不能概括为训练和部署都不需额外传感器。

**本机建议**：借鉴“物理观测器 + 误差学习”的结构作为候选基线；固定基机械臂需要重新推导数据目标和输入。碰撞检测改进与连续力数值精度应分别评估；置信度输出也必须独立校准。

### 6. 较新的不确定性方法：K-VARK

Oğuzhan Akbıyık, Naseem Alhousani, Fares J. Abu-Dakka. **K-VARK: Kernelized Variance-Aware Residual Kalman Filter for Sensorless Force Estimation in Collaborative Robots**. arXiv:2512.13009；初稿 2025-12，核对最新 v2，2026-06-16。此处按预印本列示。

[论文全文 v2](https://arxiv.org/html/2512.13009v2)

方法学习残差均值与输入相关方差，结合自适应扰动卡尔曼滤波。在六自由度机械臂上做了负载和接触实验。v2 明确承认近零速分析有限、温漂和静摩擦仍待改进，且接触任务的末端力精度仍有限，不能作为已经解决全部问题的证据。重点读 IV、V-D、VI、VII。

**本机建议**：可研究关节残差的状态相关分布及带置信度输出，而非只发布单一力值。自适应协方差和模型必须通过已知力阶跃、持续接触及卸载测试，防止以减弱响应换取空载曲线平滑。完整实现需重建其动量状态及噪声模型。

### 7. 奇异位形：测量可观测性的边界

Maged Iskandar et al. **Collision Detection, Identification, and Localization on the DLR SARA Robot with Sensing Redundancy**. ICRA 2021。

[DLR 作者全文](https://elib.dlr.de/144160/1/Iskandar_ICRA2021.pdf)

论文解释关节外力矩到末端六维力的映射在雅可比降秩时不能唯一恢复所有分量。其无奇异估计使用额外力/力矩测量的冗余信息，不能直接当作纯软件无传感器方案。

**本机建议**：软件上按可观测性选姿态，并研究带噪声权重的正则化估计；降秩时只报告可观测的力/力矩组合及不可用状态。阻尼不能补回丢失的信息。force-only 假设末端外力矩为零，不能不经接触模型验证就切换模式提升有效率。本次大量无效结果的诊断是残差不符合末端接触模型，应与奇异单独统计。

## 继续深入时可读的两篇

- Victor Vantilborgh et al., **Probabilistic Latent Variable Modeling for Dynamic Friction Identification and Estimation**, arXiv:2412.15756v1，2024。[全文](https://arxiv.org/html/2412.15756v1)。把不可观测摩擦历史表示为潜在动态状态，在 KUKA 平台检验动力学预测；不应将其开环预测验证直接等同于接触力估计验收。
- Ines Sorrentino et al., **Physics-Informed Neural Networks with Unscented Kalman Filter for Sensorless Joint Torque Estimation in Humanoid Robots**, arXiv:2507.10105v1，2025。[全文](https://arxiv.org/html/2507.10105v1)、[论文关联代码仓库](https://github.com/ami-iit/paper_sorrentino_ral2024_balancing_torque)。针对谐波传动，用电机侧及关节侧信息估计摩擦和关节输出力矩；任务是力矩控制，不能直接视为末端外力检测的完整替代方案，迁移前需核对本机信号是否具备。

## 与原版 FACTR2 的关系

[FACTR2 / NEXT，2026，附录 A.1–A.3](https://arxiv.org/html/2606.12406v2) 使用无持久隐藏状态的滑动窗口 LSTM，最终输入为位置、速度和指令跟踪误差历史；不要求显式估计加速度。数据采集强调独立关节、耦合运动和不同速度。

本机可先保留相同网络、损失与输入，完善覆盖并重新训练一个可比较的基线。MD、温度、残差分支、动量观测器以及概率输出都应作为独立扩展实验，记录新增输入与训练目标，不能悄悄替换后仍声称严格复现。原版窗口的设计旨在避免长期隐藏状态漂移；它是否丢失本机关键摩擦历史，需要消融验证。

## 建议的验证顺序（本调研提出，尚未执行）

1. **先核对测量链。** 使用已有日志分析力矩、位置、速度及原生跟踪目标的真实延迟、噪声和单位。参数文件一致不能证明实际采样无时延；延迟只能在训练数据上估计并冻结，用独立日志验证。
2. **补充有针对性的无接触数据。** 连续平滑运动与启停实验分开设计：同姿态正反方向到位并停留、不同速度/加速度、单关节与多关节组合，以及冷/热机。启停实验服务于摩擦辨识，不能把故意的启停混入需要连续平滑的轨迹而不加标记。速度提升仍须在实际跟踪和硬件限制内逐级验证。
3. **先比较 NEXT 与 NEXT+MD。** 训练/验证/测试按独立会话隔离，保证历史状态无泄漏；MD 应从每个会话的实际历史生成，并保持离线和在线更新一致。加入历史特征是改善静止迟滞的第一候选，不预先宣称摩擦已被辨识。
4. **再比较残差补偿与概率观测器。** 在同一冻结测试集上分别评估，不同时改变所有组件。若加入刚体动力学或 LuGre/GMS，需辨识其参数；仅用不准确 URDF 得到的模型可能引入新的误差。
5. **关节与末端分层验收。** 先查看各关节偏置、RMSE、P95、返回前后差异和动态条件分组；再查看六维末端数值、有效率、无效原因及观测条件。碰撞标志单独统计。无效输出不能填零，不能只在有效样本上展示较低误差而忽略可用率。
6. **验证真实力响应。** 静态可使用有可追溯方向与大小的已知载荷；动态接触最好临时使用同步、经标定的力/力矩传感器，并补偿附加工具的重力和惯性。空载测试不能单独证明接触幅值、持续接触和卸载恢复正确。

预期目标是得到可量化的零附近误差、可接受动态残差和可靠有效率。数值严格恒零既不是这些论文的普遍保证，也不能作为对真实接触仍然敏感的充分证据。持续自动扣零可能吸收真实慢变外力；提高门限只能改变报警，不能修复连续力值偏差。
