# 6. 只读在线推理、断流恢复与七关节可视化

## 目标与依赖

在 mock / H5 回放条件下实现可靠的单臂在线推理和诊断，验证训练与在线结果一致，为第 8 部分准备只读部署。依赖 [3](3_adapter_and_health.md)、[5](5_training_and_offline_evaluation.md)；不需要真机。不引入 teacher-arm feedback、运动策略、力矩回写或控制模式切换。

## 源码现状

前缀：`factr2_next/src/factr2_next/factr2_next/`。

- `inference/inference_node.py`：四流同步后直接 append/predict；尚无名字/维数/有限值/源年龄验证；无断流 watchdog、窗口重置或恢复预热。
- `inference/history_buffer.py`：无 clear，跨断流会复用旧行。
- `inference/checkpoint.py`：第 5 部分负责严格加载，在线调用不能绕过其验证。
- `TorqueFilter`：EMA/lowpass 有内部 y 状态；contact hysteresis 有 contact_state，断流时两者也要重置。
- `_joint_state` 当前输出 position 数值但不填 name。继续用 position 发布 NEXT 力矩，补正确 7 关节 name。
- `visualization/web_node.py`：Python `JOINT_COUNT=6` 与 HTML 的六关节列表均硬编码；不能只改 Python 常量而漏掉第 7 个前端曲线。

## 实施步骤

1. 用 `config/w3/{left,right}/inference.yaml` 与第 5 部分 metadata 校验 checkpoint，明确 `/factr2/{side}` 输入和 `/next/{side}` 输出。输入字段遵守契约，加载失败在订阅和运行前给出明确错误，不使用零模型 fallback。
2. 同步回调检查四流 name 完全一致且等于期望 0..6，指定字段各 7 维、有限、同 stamp（W3 profile）；源 stamp/接收年龄合法且严格前进。乱序四流应拒收而非“凑合计算”，因为 adapter 已保证固定顺序。
3. 配置 max_gap_seconds=0.04；遇到更长源间隔、回退/坏帧或 adapter 非 OK 时清 history、filter、contact，并标记 invalid。恢复后必须重新收到连续 50 个合法样本；第 50 个可以首次预测，前 49 个不输出有效 torque。
4. 建立单调时钟 watchdog，默认 input_timeout_seconds=0.25；所有消息消失时仍能标记 stale，不依赖下一次同步回调才能发现断流。清 message_filters 待配队列或拒绝所有过期队列帧，避免恢复时混入旧样本。
5. 发布 `/next/{side}/status` DiagnosticArray，至少含 loading/warming/valid/stale/invalid 状态、history_count、source_age、drop/reset counters、实际输出频率、推理耗时。无效时不输出伪造零力矩；contact 可复位 false，但消费者必须用 status 理解它不是“已证明无接触”。
6. 保持公式与输出语义：free=模型反归一化预测；raw=measured-free；filtered=配置的平滑；mse=mean(raw²)，Nm²；contact magnitude=指定 norm/scale；low<=high 的迟滞。拒绝非法 alpha、scale、rate 和阈值，不能靠 clip/静默修改掩盖错误配置。
7. 三路 torque JointState 带相同末帧源 stamp、正确 name、position 7 维；Float32/Bool 无 header，因此归档时用 status/接收时间关联，不能伪称它们自带源时间。若新增 stamped 诊断，兼容保留现有输出。
8. 推理在独立进程，不进入 W3 update。控制输入积压，过期窗口丢弃并重新预热，不用无限排队产生“50 Hz 但延迟越来越大”的输出。先测 CPU 基线；只有实测需要时再用有界 worker/合适 GPU，明确队列策略。
9. 七关节可视化支持实际 joint_names/可配置 joint_count，前端和后端一致，保留 raw/filtered、free/MSE/contact 诊断。W3 profile 不默认展示/订阅 feedback 曲线。页面显示 warming/stale/invalid 和最后有效数据年龄，断流不持续把旧曲线标为实时。
10. 提供单侧和双侧只读 launch/启动脚本及 H5→四流回放工具，复用第 4 部分 mock。不同 side 的 node 名、输出 root、checkpoint、web port 隔离；launch 不启动 bridge/controller，不包含 Piper/Gello 包。

## 验收标准

| 编号 | 验证 | 必须结果 |
|---|---|---|
| INF-01 | 相同 H5 顺序回放 | 每个合法末帧 free/raw 与第5部分离线结果 allclose(atol=1e-5,rtol=1e-5)，timestamp 对应末帧 |
| INF-02 | 窗口预热 | 1..49 帧无有效 torque，第50帧首次预测；连续 N 帧有 N-H+1 次预测 |
| INF-03 | 0.3 s 断流、恢复 | <=0.27 s status stale、清窗口/filter/contact；恢复后连续50帧重新预热，不复用断流前历史 |
| INF-04 | >40 ms 缺口、坏 name/shape/NaN、回退 stamp | 拒收并清连续窗口，明确 invalid 原因；旧排队消息不能恢复 valid |
| INF-05 | filter 与 hysteresis | none/EMA/lowpass 小序列结果可手算，raw 不受滤波改变；high 开启、low 关闭、两者之间保持 |
| INF-06 | checkpoint/config 不匹配 | 启动失败且无有效输出，无 command publisher/service side effect |
| INF-07 | 60 s、50 Hz 回放（另测10分钟稳定性） | warmup 后有效输出>=45 Hz；无积压持续增长，峰值内存无持续线性增长，无 NaN |
| INF-08 | 性能测量 | 默认目标单次推理 P95<20 ms、输入源时刻到输出接收 P95<=60 ms；计时含义与实际设备写报告 |
| INF-09 | 可视化 | 0..6 共7关节 raw/filtered 均可切换，第7关节独立变化可见；单位/关节名/侧别明确；断流显示 stale |
| INF-10 | 双侧同时启动 | 彼此不混消息/模型，不抢同一 web port；停止左侧不会停止右侧或任何 W3 节点 |

- [ ] 性能目标为当前 50 Hz 工程门槛，未达到则报告 FAIL 和瓶颈，不静默调低 rate/history 来通过。可显式选择更合适计算设备并重新记录结果。
- [ ] mock / 回放测试采用真实 InferenceNode 发布路径与真实 checkpoint，不仅测试裸模型。
- [ ] `ros2 node info`、launch 和源码检查证明所有 NEXT 进程无 motor command/trajectory publisher，无 enable/disable/switch_controller client；可 kill NEXT 进程验证 mock W3 source 持续工作。
- [ ] 启动、Ctrl-C、端口占用、异常 checkpoint 等故障均有可操作错误信息与正常资源释放。

时延用相同主机和 ROS 时钟测量：输出接收 ROS 时间减末帧源 stamp；单次模型耗时用单调时钟（CUDA 需同步或事件计时）。记录回放节奏、调度、可视化是否同时运行；不把 `ros2 topic hz` 当时延证明。

## 交付与交接

提交在线校验/重置/诊断、七关节可视化、左右配置与只读 launch、回放工具、行为测试、`docs/inference.md` 及 `docs/acceptance/6_inference_and_visualization.md`。交给第 8 部分的是真实可用的部署命令、性能报告和失败处理语义；smoke checkpoint 不允许直接用作真机外力估计结论。
