# W3 只读推理与七关节页面

输入适配器必须已运行；NEXT 不启动 W3 bridge、controller、CAN、teacher-arm feedback 或任何运动接口。复制左右 inference/visualize 模板到项目内运行目录，填写第 5 部分严格验收的 checkpoint 路径，保持左右 side/order/topic root 和不同 web port（默认 8080/8081）。环境将临时文件固定在项目 `log/tmp`，图表 cache 位于 `log/matplotlib`。CPU 默认单线程、H=50、50 Hz，不降低历史或频率来通过性能检查。

```bash
scripts/next.sh ros2 launch factr2_next readonly.launch.py side:=left inference_config:=/home/dingyj/factr2/data/left/inference.yaml web_config:=/home/dingyj/factr2/data/left/visualize.yaml
scripts/next.sh ros2 launch factr2_next dual_readonly.launch.py left_inference_config:=/home/dingyj/factr2/data/left/inference.yaml left_web_config:=/home/dingyj/factr2/data/left/visualize.yaml right_inference_config:=/home/dingyj/factr2/data/right/inference.yaml right_web_config:=/home/dingyj/factr2/data/right/visualize.yaml
```

单独节点仍可使用 `ros2 run factr2_next next_infer/next_visualize --ros-args -p config_file:=...`。启动先验证 config、checkpoint 完整性和 side/order/history/50 Hz，再建立推理订阅及输出。非法 alpha、scale、采样率、low>high、缺模型/错模型均给明确错误，不使用零模型回退。

输入四路 name 必须相同且为 side_joint_0..6，字段各 7 维、有限、其余字段为空、同一个正源 stamp。拒绝未来/超过 0.25 s 的源帧、回退/重复 stamp 和重置前排队的旧帧；默认也要求对应 adapter 的新鲜 OK diagnostics。ATS 队列默认 5，ROS subscription 为 sensor-data 有界深度 5。推理独立进程内同步执行，每窗 stateless，不排无限任务。

状态机：loading → warming → valid。连续 50 个新鲜合法帧才首次输出，前 49 个无有效 torque。源间隔 >40 ms、坏帧、时钟回退或 adapter 非 OK 清空 history/filter/contact，状态 invalid；恢复需重新预热。间隔后的当前合法帧可计为新窗第 1 帧。单调 watchdog 每 20 ms 检查接收超时，即便 ROS clock 停止仍能在 0.25 s 附近标 stale。清 ATS 队列在同步回调外执行，避免 ATS 的回调后删除逻辑冲突。推理后再次检查源年龄，过期结果直接丢弃并重新预热。

`/next/{side}/status` 为 DiagnosticArray，status name=`factr2/next/{side}`，values 用 JSON 编码，包括 state/reason、history_count、source/receive age、outputs/dropped/resets、last_output_stamp_ns、近 1 s output_hz、最近及 P95 infer_ms、真实 device。历史耗时保留最多 512 项，频率时间戳最多 100 项，history 最多 50 项。无效时不发布伪造 torque；contact 可置 false，消费者须结合 status 判断有效性。

输出三路 JointState 都填 7 个 name，position 用于 Nm，header 为窗口末帧源 stamp：free=反归一化模型预测，raw=measured-free，filtered=配置的 none/EMA/一阶 lowpass。mse=mean(raw²)，Nm²；contact magnitude=配置 norm/scale 和 low/high 迟滞。Float32/Bool 本身无 header，审计通过接收时间与 status 的末帧 stamp 关联。有限输入造成的派生 overflow 也会拒绝，避免 Inf torque/scalar 发布。

页面从配置 joint_names 动态生成 0..6 的 raw/filtered/free 曲线，含侧别、Nm、MSE、contact 和最后有效数据年龄，不默认订阅 feedback。`web.refresh_hz=10` 合并 50 Hz torque/status 到有界页面刷新，plot max_points=500。状态和 torque 接收年龄超时会显示 stale；HTTP 连接断开显示 disconnected，不能把旧曲线持续标实时。HTTP `/snapshot` 提供同样 JSON，`/events` 为 SSE。端口被占用报 `web_bind_failed host:port`，Ctrl-C 关闭 server/socket/thread，端口可再次绑定。

只读回放接受已严格检查的 H5：

```bash
scripts/next.sh python scripts/replay_w3_h5.py data/left/session.h5 --episode ep_0000 --loop
```

回放保留数值/顺序，header 重映射到当前 ROS clock，50 Hz。启动前 3 s 发送明确 synthetic replay 的 adapter OK 状态供节点发现/准备。loop 边界插入 0.3 s 停顿，保证历史不会跨越 episode 尾首。回放不发布标准 W3 command/state，也不控制硬件；第 4 部分 `w3_mock.py` 用于标准 W3 → 实际 adapter 的闭环测试。

本机离线 DDS 使用 `config/w3/dds_loopback.xml`、隔离 domain 和 localhost。双侧 launch 增加独立参与者数量，`maxInitialPeersRange=32` 保证回环发现范围覆盖测试进程；Fast DDS 2.6 文档给出的默认值为 4，见 [官方 transport descriptor 文档](https://fast-dds.docs.eprosima.com/en/2.6.x/fastdds/xml_configuration/transports.html)。这仅是回环测试配置，真实跨机部署另行验收。

```bash
scripts/next.sh python scripts/test_inference_replay.py
scripts/next.sh python scripts/test_inference_stream.py --seconds 600 --domain 87
scripts/next.sh python scripts/test_web_dual.py
scripts/next.sh python -m pytest factr2_next/src/factr2_next/test/test_inference_math.py -q
```

replay 测试经过实际 InferenceNode 发布路径验证离线 allclose、50 帧边界、断流、错 name/shape/NaN、回退、旧帧、派生 overflow、adapter 拒绝和 gap 恢复。stream 测试左右独立真实模型进程，标准 300 Hz mock 经 50 Hz adapter；使用共同源 stamp 区间比较三路输出，避开观测起止回调不同步。单次耗时用 monotonic 包含窗口预处理、模型和反归一化，延迟为同主机接收 ROS time 减末帧源 stamp，目标 P95<20 ms / <=60 ms。RSS 从真实推理子进程 `/proc` 读取，不把探针缓存算进模型内存。

`test_web_dual.py` 运行实际双侧 launch，可用 `--hold-seconds` 留浏览器检查时间。使用专用 18100/18101 端口；正常程序释放后才运行下一份测试。第七曲线用 NEXT 输出上的独立可识别脉冲验证；测试停止左 infer 后右 infer/W3 mock 持续工作，左 HTTP 页面 stale。所有子进程和 socket 在 finally 清理。

smoke 模型只用于开发验收。第 8 部分须使用第 7 部分的真实冻结模型和阈值，先只读部署，再评估真实效果。
