# 8. 真机只读部署、接触可辨识性与最终交付

## 目标与前置条件

把第 7 部分合格的单侧模型通过第 6 部分只读推理工具部署到 W3，验证无接触基线、可重复轻微接触、运行时效性和故障隔离。依赖 [6](6_inference_and_visualization.md)、[7](7_real_data_and_model_validation.md)。左右分别通过后才算第一阶段双臂复现完成。

需要操作者和真机。没有现场条件时完成部署配置、离线回放、测试步骤和报告模板，真机验收 BLOCKED。用户后续授权和现有 W3 流程负责真机使能、运动和接触；NEXT 节点始终只读。

## 文件与运行范围

- 使用 `config/w3/{left,right}/{inference,visualize}.yaml`、real adapter profile、checkpoint metadata 和冻结 acceptance 配置。
- `docs/deployment.md` 汇总准确的终端/source/启动/停止/故障恢复命令。
- `docs/experiments/contact_protocol.md` 定义接触位置、方向、时刻标记、持续时间、重复次数和禁止超出范围的条件。
- 只补部署配置、日志/分析工具和必要诊断，不引入闭环、末端 wrench 反演、teacher-arm feedback。新增功能若改变第 6 部分代码，必须重跑相关离线验收。

## 实施步骤

1. 部署前核验实际 W3 状态：标定/URDF/工具/负载/增益/前馈与训练 metadata 相符、位置控制器 active、健康门禁 OK、四输入正常。模型 side/order/history/rate/normalization 必须匹配；不匹配就停止 NEXT 估计并报告。
2. 按既有 W3 流程保持运动系统运行，启动独立健康检查、adapter、inference、visualizer 与原始/估计 rosbag 归档。NEXT 启动/停止不会调用 W3 控制服务。记录节点/话题/服务拓扑作为只读证据。
3. 先每侧连续至少 10 分钟无接触静态/动态观测，涵盖部署范围内多个姿态和速度。记录 tau_measured、tau_free、raw/filtered residual、status、contact、健康、源时间/接收时间、温升条件。
4. contact norm/scale/filter 与迟滞阈值由第 7 部分 train/val 无接触分布和明确的调参试验确定。正式测试前冻结。阈值不是模型学习的物理触力值；不要把默认 low=1/high=2 直接视为 W3 已标定阈值。
5. 单独做操作者执行的轻微、可重复外部接触，不修改自由空间模型。每側至少 2 个姿态、每姿态 5 次完整“无接触→轻微接触→撤离”试验，共至少10次。预声明接触位置/方向、保持>=0.5 s、独立时间标记方式；记录对照无接触片段。
6. 接触 ground truth 仅来自操作时间/视频/人工事件标记，不能用 NEXT contact_state 本身生成标签。没有力传感器时报告检测/残差响应，不报告绝对 Nm/牛顿误差、六维末端力或安全级检测能力。
7. 在 mock/回放先完成全部故障注入；真机只测试现场允许且不会使 W3失去托举/控制的观测进程故障，例如停止 NEXT inference/adapter、错 checkpoint 启动。不要为测试 NEXT 去拔 CAN、关停 bridge 或发送故障命令。硬件断线案例用 mock 证明门禁行为并明确证据范围。
8. 双侧合格后同时只读运行至少 10 分钟，确认模型/命名空间/端口隔离、频率/时延/资源预算正常。停止一侧 NEXT，另一侧和 W3 控制持续运行。

## 定量验收标准

正式测试前在 acceptance 配置记录以下默认工程门槛；如需不同门槛，必须在测试前给出具体依据和版本。不能测试后改阈值把失败改成通过。

| 编号 | 指标 | 第一版默认门槛 |
|---|---|---|
| LIVE-01 | 无接触残差 | 每关节/关键分组 RMSE和bias达到第7部分冻结预算；不能只看一条静态曲线 |
| LIVE-02 | 无接触误报 | 在模型 valid 的无接触时间内，contact=true 时间比例<=1%；另报事件数/时长和全部 invalid 时间 |
| LIVE-03 | 接触检出 | 每侧至少10试验，至少9次在接触开始后<=0.5 s 开启contact；接触段raw norm超过匹配无接触P99，filtered magnitude持续>=high至少0.2 s |
| LIVE-04 | 撤离恢复 | 至少9/10次在撤离后<=1.0 s关闭contact；回到匹配基线范围，并记录raw/filtered延迟差 |
| LIVE-05 | 吞吐与时延 | warmup后 valid 输出>=45 Hz；推理P95<20 ms，源输入→输出接收P95<=60 ms；报告P50/P95/P99/max与测法 |
| LIVE-06 | 断流语义 | 超时<=0.27 s标stale，停有效torque；恢复后连续50帧预热；false contact不能被误当有效无接触 |
| LIVE-07 | 故障隔离 | kill NEXT / adapter停止 / bad checkpoint不改变W3已存在控制状态、增益、目标来源；mock故障与真机观测明确区分 |
| LIVE-08 | 双侧运行 | 左右各合格checkpoint，7关节全量显示和归档，10分钟同时运行无串流、持续积压或资源失控 |

`raw norm` 使用预声明的 norm（如 L1），无接触 P99 由匹配姿态/速度的独立基线计算，保留原始序列。contact 事件检测与恢复时间以独立事件标记为零点；50帧窗口是历史上下文，不等于必然1秒检测时延。

误报统计不可删除不方便的运动区域；分布外状态和 invalid 区间单独报告，若关键工作区大量 invalid 则不能以剩余 valid 时间的低误报通过。覆盖范围必须与第7部分一致。

## 最终验收清单

- [ ] LIVE-01..08 有逐项原始数据/脚本/统计证据，左右独立结论明确。
- [ ] 只读拓扑已审查：NEXT进程仅发布/factr2或/next命名空间数据/诊断，未连接W3 trajectory/motor command和使能/模式切换接口。
- [ ] 停止 NEXT 故障实验只是验证进程隔离；不声称 W3 因此具备完整反馈联锁或失效停车能力。
- [ ] 每次实验关联checkpoint/normalization/config/metadata、软件commit、标定/工具/负载、时间、日志/视频/接触事件表及SHA256。
- [ ] 第1..6离线验收报告完整，第7..8左右真机验收完整；失败项不隐藏，重新采集/重训/修改后的回归记录可追溯。
- [ ] README/部署文档提供别人可复现的启动、观察、停止NEXT和回退步骤，不要求安装Piper/Gello或修改W3 Python。

## 交付与项目完成定义

提交 `docs/deployment.md`、接触实验协议、最终左右配置、分析工具、精简结果摘要和 `docs/acceptance/8_readonly_deployment_and_contact_validation.md`。在 README 给出最终合格模型/数据清单索引；大体积结果存在 data/runs/reports 的忽略目录。

只有左右两侧都完成自由空间模型与只读接触验收，才将“W3 双臂 NEXT 第一阶段复现”标为完成。结论限定为在已声明范围内的关节外力残差估计与接触指示；任何力反馈控制或接触运动策略都属于后续独立开发阶段。
