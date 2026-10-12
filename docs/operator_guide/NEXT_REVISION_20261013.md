# 右臂 NEXT 重新采集与训练（2026-10-13）

适用当前 `/home/venom/dual_arm_robot` 控制仓库与 `/home/venom/factr2` 学习仓库。此版本已修改采集规划、训练前审计、训练诊断和可选末端点力换算；尚未采集新版真机数据，不能声称三个现场问题已经消除。旧数据、权重与正在运行的进程没有替换。

## 为什么重采，而不是继续增加 epoch

现有 `baseline_campaign_20261012_000958_resume4` 的训练窗口中，单关节运动占 87.89%，组合运动占 4.72%，静止占 3.91%。逐关节运动几乎都围绕同一个 home；val 的两个组合任务姿态距最近训练姿态的七关节欧氏距离约 21～26°。一维关节范围很宽，并不代表七维组合姿态有足够覆盖。

旧候选 `baseline_candidate_20261012` 的验证最优在第 2 轮，之后训练损失下降而验证损失上升；某静止块 J1 均值残差 1.274 N·m，同姿态双向返回的基线差最高 0.781 N·m。它的独立空载验收失败。这些现象支持先修补数据分布和部署范围，无法证明只增大网络或继续训练能解决问题。

理论空载外力为零，实际估计以独立空载残差、卸载后恢复和噪声容限验收。不能通过把电机力矩训练标签改成零、启动时扣一个常数，或在疑似无接触时不断扣零，来证明模型正确。

## 与论文的对应关系

依据 [论文第 4 节及附录 A.1–A.3](https://arxiv.org/html/2606.12406v2) 与 [作者代码库](https://github.com/philiphan0109/factr2_next)。保留以下 NEXT 契约：

- 输入：最近 50 帧 `[q, qdot, q_cmd-q]`，100 Hz，每次窗口重新初始化 LSTM 隐状态。
- 标签：已确认无外部接触时的实测电机力矩；输出 7 个关节的自由空间力矩。
- 推理：`tau_external = tau_measured - tau_free_predicted`，保留未经扣零的原始残差。
- 网络：2 层 LSTM、hidden 128；2 层 MLP、hidden 256；dropout 0.1。
- AdamW、学习率 0.001、weight decay 1e-6、梯度裁剪 1；早停 patience 20、warmup 10、min_delta 1e-5。按物理 N·m 的 MSE 选验证最优；标准化只拟合训练集。batch 2048 沿用参考代码设置，训练时间取决于本机设备。
- 采集顺序保留逐关节覆盖、组合/笛卡尔式联动、慢速和快速运动。

共享任务姿态、静止/返回探测、覆盖门禁、分组诊断、控制参数冻结和末端换算属于 W3 工程扩展。没有把 FIRST 的接触数据重采样用于 NEXT 训练，也没有把接触/卸载数据加入自由空间训练集。

## 新采集配方

文件：[right_next_protocol_v2.yaml](../../config/w3/motions/right_next_protocol_v2.yaml)。生成器：[w3_baseline_protocol.py](../../scripts/w3_baseline_protocol.py)。

1. 接近 home `[0,50,0,-55,0,0,0]°`，静止 12 s。
2. 七关节分别在新工作范围内做慢、快完整双向正弦运动。速度上限仍为 6/12°/s，加速度 24/36°/s²，jerk 120°/s³。
3. 在 8 个共享组合任务姿态采集。每个姿态先从两个方向返回并各静止 12 s，再做两种七关节同步正弦的慢/快轨迹、运动后静止，以及 XYZ 往复与 XY 圆形的慢/快轨迹。
4. 回到 home 并静止。各任务姿态、返回探测、连接与联动路径的每个命令采样点都须满足缩放雅可比条件数 ≤20；不提高在线的 100 门限。

共享姿态的种子为 6131。train、val、test 必须分别真实录制，角色种子 1701/2903/4307 改变起行方向、速度和联动相位。共享部署姿态允许验证同一工作区的跨会话重复性；它不是一次录制的随机窗口划分，也不代表已验证全部未见姿态的泛化。

| 关节 | 新工作范围（°） |
| --- | --- |
| J1 | -40～40 |
| J2 | 20～90 |
| J3/J5/J7 | -30～30 |
| J4 | -85～-25 |
| J6 | -23～23 |

这是当前任务区域的定向采集，模型不能因此被宣称覆盖旧的大范围。新的检查点冻结工作范围，生成对应 `inference.yaml`；在线节点拒绝使用与该检查点不符的范围。范围内的稀疏组合仍可能预测不准，范围门禁不能代替精度验收。

在已归档的本机 URDF 上，train 计划约 46.8 分钟、257 段；时间约为单关节 25.6%、组合运动 46.6%、静止 18.4%、连接 9.5%。这是当前保守速度下加强静止/组合覆盖的工程协议，不是论文示例的 10 分钟采集。可先做较少姿态的试采；若修改配方，三个角色须重新生成同一配方，不能混合不同版本、范围或锚点。按新配方完成的独立 val/test 也需要相近时长。

训练前按实际有效窗口计数（不是命令时间比例）检查 train 和 val：组合运动 ≥25%、静止 ≥10%、单关节 ≤60%；val 静止姿态距训练实测姿态的最大单关节距离 ≤5°。它们是可审查的工程门限。旧数据不满足这些条件，不能直接拿旧 H5 当作新版数据。

## 生成、采集与训练入口

生成计划只写文件。URDF 和初始角度须来自当次运行的机器人描述和控制目标；归档 URDF 的离线结果不能直接作为真机运动计划。先按现有 [启动手册](RIGHT_ARM_STARTUP_20261010.md) 建立当前状态。

```bash
cd /home/venom/factr2
scripts/next.sh python scripts/w3_baseline_protocol.py \
  --config config/w3/motions/right_next_protocol_v2.yaml \
  --urdf /ABSOLUTE/PATH/current_right.urdf \
  --initial-deg J1 J2 J3 J4 J5 J6 J7 \
  --role train --output /ABSOLUTE/PATH/train.yaml
```

分别生成 val/test。同样使用现有 `w3_collision_check.py` 审查每个计划和当次左臂姿态，保留匹配哈希的 `.collision.json`。几何条件良好不代表无碰撞，离线检查也不包含未建模的现场障碍。实际录制仍用 `quick_capture.py --frequency 100 --motion ...` 和 `w3_coverage_motion.py run --file ... --send`；具体启动、结束、磁盘检查和 C2 发送步骤见 [既有完整操作说明](RIGHT_BASELINE_PROTOCOL_20261011.md)。`--send` 会实际运动；本次代码修改没有执行它。

末端、夹爪、负载、标定、增益、重力和摩擦配置须在所有会话中一致；保持相近预热条件。训练只接受确认无接触的完整有效动作，断流后分段，不跨会话或事件边界构造窗口。三角色会话目录填写到 `baseline_campaign.example.yaml` 的副本中，然后：

```bash
scripts/next.sh python scripts/prepare_w3_baseline_campaign.py \
  --sessions /ABSOLUTE/PATH/campaign.yaml \
  --output /home/venom/factr2/data/right/NEXT_V2_CAMPAIGN_ID

scripts/next.sh python scripts/audit_w3_next_data.py \
  --config data/right/NEXT_V2_CAMPAIGN_ID/train.yaml \
  --output data/right/NEXT_V2_CAMPAIGN_ID/coverage_audit.json

scripts/next.sh python scripts/train_w3_baseline.py \
  --config data/right/NEXT_V2_CAMPAIGN_ID/train.yaml \
  --acceptance config/w3/right/baseline_acceptance.yaml
```

v2 整理器自动选用 [train_next_v2.yaml](../../config/w3/right/train_next_v2.yaml)。覆盖不足时在优化之前报错；若想先定位问题，独立运行审计命令会保存完整原因。只有 train/val 的数值进入覆盖诊断，test 只作结构/独立性检查，精度留到候选模型固定之后。

每轮报告静止、组合、单关节等组的各关节物理 RMSE 和偏差；`metrics.json` 还保留每个静止块的诊断。MSE 选模型的规则不变，额外诊断不会改变训练标签、扣基线或选择测试集。新模块源码哈希写入元数据。

候选目录包括模型、归一化、冻结配置、元数据、指标、`baseline_acceptance.json` 和待审查的 `inference.yaml`。没有卸载验证时，`deployment_ready` 仍是 false，训练脚本返回 1 表示整套验收未完成或失败，不表示训练一定报错。

## 卸载恢复和奇异点

真正的接触/卸载只用于验证：至少 3 个不同 case，记录空载静止、施力、完全松手后静止；标注真实时段和 H5 哈希，使用 `quick_capture.py --validation-only` 与 `evaluate_w3_baseline.py --release-h5 ... --annotations ...`。现有阈值保持：静止均值 ≤0.1 N·m、P95 ≤0.25 N·m、返回/卸载前后基线差 ≤0.1 N·m，卸载后 2 s 稳定再检查至少 5 s。详见 [卸载验收步骤](RIGHT_BASELINE_PROTOCOL_20261011.md#手推与卸载验证)。未通过时保留失败，不重新扣零获得 PASS。

stateless LSTM 在 50 帧输入完全被替换后不会携带前一次施力的隐状态。因此长时间卸载残差应继续检查实测姿态、命令误差、摩擦/回差、热态和预测误差。审计同时报告“静止输入近似相同但电机力矩不同”的潜在歧义；这是诊断提示，不是数学证明，也不会修改标签。旧数据这项诊断没有发现符合当前严格容差的矛盾对，不能把所有残留直接归因于不可观测摩擦。

NEXT 关节外力矩不需要求逆雅可比，不因末端奇异而停止。六维末端力/力矩仍须在完整雅可比可观测时才可输出。需要只观察末端点力时，可对现有只读脚本选择：

```bash
scripts/next.sh python scripts/w3_endpoint_wrench.py \
  --config /ABSOLUTE/PATH/endpoint_profile.yaml --mode force_only
```

`force_only` 解三维平移子问题，明确假设接触在指定末端原点、外加力矩为零；它可避免单纯角向退化造成的拒绝，但仍拒绝平移奇异或与点力假设不相容的关节残差。页面的 Mx/My/Mz 留空，诊断标注 `moments_observed=false`；ROS WrenchStamped 的 moment=0 表示模型假设，不是力矩测量结果。纯力假设也可能无法区分某些可投影的分布载荷，不能拿它测拧转力矩。默认 `full_wrench` 模式不变。

新版协议通过选择任务姿态减少末端奇异，点力模式只在假设成立时适用；不存在通过重新训练让真正的机械奇异点消失的办法。
