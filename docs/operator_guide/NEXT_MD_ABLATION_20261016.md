# 原版 NEXT 与 NEXT+MD 的离线对照

本次按用户要求实现 Shan & Pham, ICRA 2024 的 Motion Discriminator（MD）特征，并与现有 NEXT 对照。离线实验不发送运动、不自动部署；后续按用户要求新增的独立在线观测见下节，原版检测继续运行。

## 论文和源码核对

- [论文 v2，Algorithm 1、III-B、Table I](https://arxiv.org/html/2309.16219v2)
- [作者主页](https://shanshilin.github.io/)，论文条目仅提供 Paper/Video。
- 2026-10-16 核对作者主页、arXiv 及 GitHub 定向检索，未找到可确认属于该论文的官方公开代码；不据此断言不存在。本实现由论文伪代码重写，没有引入搜索结果中其他领域同名 Motion Discriminator 的代码。

论文 Algorithm 1 在 `abs(velocity[j]) >= threshold[i,j]` 时保存该速度，否则保留上次值。本实现保存最后越阈值速度，不夹到阈值、不只保留符号。这里将 MD 拼接到 NEXT 的 LSTM 输入；没有实现论文完整 HRDL 网络。

## 冻结的对照条件

正式本次实验目录：`data/right/md_ablation_20261016_v2`。

| 项目 | 原版 NEXT | NEXT+MD |
| --- | --- | --- |
| 基础输入 | q、qdot、qcmd−q，21维 | 同样21维，加105维MD |
| 历史 | 50帧，100 Hz，每窗口重置LSTM | 相同；MD另按原始会话维护 |
| 网络 | 两层128维LSTM，256维回归头 | 相同隐层/头，输入投影增大 |
| 标签与损失 | 实测无接触力矩；物理Nm² MSE | 相同 |
| 训练设置 | AdamW，lr=.001，weight_decay=1e-6，clip=1 | 相同 |
| 批量/上限/早停 | 2048，120轮，patience=20、warmup=10、delta=1e-5 | 相同 |
| 随机性 | seed=0的已完成冻结训练 | 共享初始参数与基线相同；新增MD列另初始化；恢复基线初始化后的RNG，使小批次和dropout随机序列配对 |

原版 NEXT 复用 `runs/targeted_next_targeted_training_20261016_v1_20261016_191157_629792` 最佳第12轮。复用前核对原manifest、模型哈希及共享模型/数据/损失/诊断源码，并核对MD组的前21列输入、标签及相应归一化与基线逐项一致。

原冻结划分保持5个训练、2个验证、2个测试会话。父录制ID和原始H5哈希都不允许重复或跨集合出现。模型按最低验证物理MSE选择，测试不参与早停或阈值选择。

## MD状态与阈值

入口 `scripts/run_w3_md_ablation.py prepare`：

1. 只用训练集明确静止段的后半段（至少到位后1秒），单关节实测位置跨度≤0.1°时纳入其静止反馈分布。
2. 计算每关节绝对速度P99.5作为静止反馈底值，再加论文Table I的15个阈值。Table I含跨层重复值，这里保留其顺序。原文为6关节，本机扩展为7关节。这是明确记录的本机阈值适配，不是原文原值。
3. 在读取验证/测试特征之前保存并散列阈值；不校准或修改原始速度，不增设速度滤波，不移位力矩。
4. 在每个完整原始H5 episode从首帧因果递推MD，包括此前的探测、转场和启停历史，再按已有父行范围取监督片段。逐字段验证派生数据与父行精确相同。独立会话、原始episode或>40 ms间断重置。
5. 初始状态为0，表示录制之前历史未知；已更新掩码仅供审计，不额外输入网络。本次不保证未知初始状态下已具备完整摩擦信息。

阈值、源哈希和处理规则保存在 `md_recipe.json`、`preparation.json`。MD缓存 `md_features.h5` 不改原始H5。训练归一化只在训练集拟合，窗口权重算法与原版完全相同。

## 测试和运行

42项相关测试通过，覆盖MD方向/阈值、长时间停留、未来数据不影响过去、流式分块和状态恢复、断流重置、完整父会话历史、错行拒绝、训练专属校准、原始列/标签/归一化一致、父会话隔离、模型加载哈希及配对初始化/随机顺序，并包含既有训练契约和NEXT回归测试。

可复用的新代码：

- `factr2_next/src/factr2_next/factr2_next/training/motion_discriminator.py`
- `scripts/run_w3_md_ablation.py`
- `scripts/evaluate_w3_md_ablation.py`
- `scripts/test_w3_md_ablation.py`

新建实验（输出目录必须尚不存在）：

```bash
cd /home/venom/factr2
scripts/next.sh python scripts/run_w3_md_ablation.py prepare \
  --manifest data/right/targeted_training_20261016_v1/split.json \
  --baseline-run runs/targeted_next_targeted_training_20261016_v1_20261016_191157_629792 \
  --output data/right/新的唯一实验目录
```

准备器冻结包源码、训练/评估入口及哈希，生成 `start_training.py`。本次CPU8线程、nice=10：

```bash
ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
scripts/next.sh env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 nice -n 10 \
  python -u data/right/md_ablation_20261016_v2/start_training.py \
  > data/right/md_ablation_20261016_v2/training.log 2>&1
```

上面是本轮已经完成的命令，勿重复启动。进度与历史见 `training_state.json` 和 `training.log`。每次最佳模型原子保存为 `next_md_seed0/best.pt`；正常完成导出模型及归一化后，执行两模型同数据评估，生成 `comparison.json`、`COMPARISON.md`、`replay_comparison.png`。后续实验可创建其目录下 `STOP_TRAINING`，请求在轮次边界正常结束。

MD checkpoint使用单独的 `next_md_ablation_v1` schema、126维输入；不能直接放入现有21维在线推理入口。

## 本轮已完成结果

MD组训练约20.7分钟，在第28轮早停，按验证集选择第8轮。原版NEXT与NEXT+MD的测试力矩MSE分别为0.0066566和0.0246266 Nm²，MD约为原版的3.70倍；静止最大绝对关节偏置分别为0.02122和0.08386 Nm。

对独立空载日志做相同时间戳对照：运动中两组共同有效的样本占54.4%，这些样本上原版NEXT与NEXT+MD的力幅值P95分别为3.131和5.111 N。运动后静止段两组均100%有效，平均力幅值分别为0.796和4.111 N。MD未改善本轮的空载偏置或误差，不建议以此模型替换现有部署。该结果仅针对本机的MD特征拼接消融，不能据此否定论文完整HRDL方法。

完整表格和曲线见 [对照报告](../../data/right/md_ablation_20261016_v2/COMPARISON.md)。共同有效样本补充由实验目录的 `postprocess_common_valid.py` 对已保存的 `replay_arrays.npz` 计算，无需重新推理；输入、脚本和输出哈希见 `postprocess_provenance.json`。训练完成后已核对34个冻结源文件及manifest、MD阈值配置和特征缓存哈希均未改变。

## 用户要求的独立在线观测

已增加 `factr2_next.inference.md_observer`，显式加载MD专属schema并检查模型、归一化、阈值配置及原版来源的哈希；没有改写训练模型文件或原部署副本。每个接收帧因果更新MD，50帧窗口只截取已更新的特征；输入异常、超时或断流时同时清除历史、MD和EMA。启动前运动历史未知，MD从零初始化，不自动执行激励运动。

两组共同读取现有100 Hz适配器；MD输出位于 `/next/right/md/`，不覆盖原版话题。原版末端页面为 http://127.0.0.1:8082/；MD关节外力矩页面为 http://127.0.0.1:8083/；MD末端力页面为 http://127.0.0.1:8084/。MD页面保留相同训练包络、EMA和末端求解拒绝条件，不扣零。此为用户请求的实验观测，不表示正式验收。

配置为 `config/w3/right/inference_md_observation_20261016.yaml`、`visualize_md_observation_20261016.yaml`、`endpoint_wrench_md_observation_20261016.yaml`。过程、控制参数现场审计及模型/源码哈希见 `log/deploy_md_20261016/`。已有进程时不要重复启动；以下命令仅供停止后重启对应观测程序，不需要重启机器人控制器或适配器：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# 分别在三个终端运行
scripts/next.sh python -m factr2_next.inference.md_observer --ros-args \
  -r __node:=next_md_inference_right \
  -p config_file:=/home/venom/factr2/config/w3/right/inference_md_observation_20261016.yaml
scripts/next.sh python -c 'from factr2_next.visualization.web_node import main; main()' --ros-args \
  -r __node:=next_md_web_right \
  -p config_file:=/home/venom/factr2/config/w3/right/visualize_md_observation_20261016.yaml
scripts/next.sh python scripts/w3_endpoint_wrench.py \
  --config config/w3/right/endpoint_wrench_md_observation_20261016.yaml
```

在线接入验证：离线相关回归38项通过；独立localhost ROS域173上的MD测试3项通过（含其中2项离线检查），覆盖实际节点输出与离线预测的一致性、MD话题隔离及超时状态清除。合成数据测试从未在机器人域74发布。

## 如何读结果

同时报告总体MSE、逐关节静止偏置/散布/P95、按单关节/联动/速度的指标，以及同目标正反到位最后1秒的差值和实际姿态差。整体静止均值可能抵消相反方向误差，不能只据该均值评优。

对刚才的真实空载完整H5进行两模型共同因果回放：同样50帧、EMA alpha=.2、末端模型、范围及SVD门限。回放从日志首帧初始化MD，首个共同50帧输出初始化EMA，因此初始历史与此前已运行的实时节点可能不同。末端无效样本统计原因和可用率，不填0；力幅值只统计有效输出。

本轮是单seed初步消融，MD增加输入投影参数，尚不能排除容量因素。已有验证/测试只覆盖6°/s参考姿态，且为事后整会话划分；这些日志已经被查看，不能再当作全新盲测。冷热机和接触—卸载仍未验证。结果不自动改变部署；最终还需要新的独立运动和真实接触验证。
