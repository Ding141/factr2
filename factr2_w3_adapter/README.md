# W3 → NEXT 只读适配与健康门禁

统一契约：../docs/interfaces/w3_next_contract.md（w3_next_v1）。7 个关节固定 `{side}_joint_0..6`；core 仅标准 ROS 消息，不 import torch 或 W3 自定义消息，不建立任何 motor command、trajectory、enable/disable/switch_controller client。

```bash
cd /home/dingyj/factr2
scripts/next.sh python -m colcon build --base-paths factr2_next/src/factr2_next factr2_w3_adapter --symlink-install
# 没有硬件时使用 mock；真实采集使用 left_real.yaml/right_real.yaml
scripts/next.sh ros2 run factr2_w3_adapter w3_next_adapter --ros-args \
  --params-file "$PWD/factr2_w3_adapter/config/left_mock.yaml" -r __node:=w3_next_adapter_left
# launch 强制显式 side；profile 默认 real
scripts/next.sh ros2 launch factr2_w3_adapter adapter.launch.py side:=right profile:=real
```

参数为 ROS --params-file 格式，启动时只读，动态修改拒绝；side 必须显式 left/right。output_root 必须是契约的 /factr2/{side}，输入 topic 可配置为合法绝对名。publish_hz 默认50、input_timeout_seconds=.25、max_source_skew_seconds=.03、hardware_health_timeout_seconds=.25。Mock 标记 mock/no-hardware-health；real 配置 require_hardware_health=true。

输入分别按 name 重排。实测 position/velocity/effort、目标 position/velocity 都与 name 等长，重复/缺关节、非法所用数值、目标非空 effort 拒绝；额外另一侧/夹爪不进入向量。四路输出字段分别 position、velocity、position、effort，其他字段空，单位 rad/rad/s/rad/Nm。不重复做零位/符号变换。

50Hz STEADY_TIME timer 检查 monotonic 接收年龄和 ROS 源年龄，两者≤.25s，两源 skew≤.03s。只输出新状态源 stamp，四流共用该 stamp；不补零/外推/给旧值重打当前时间。无效新消息清空两源缓存，恢复需两路新合法消息。重复/回退/未来/零 stamp 拒收；ROS clock 回退清空缓存及高水位，进入新的时间段。下游 recorder 需在时钟回退/缺口时切 episode/segment（第4部分）。静态值但 stamp 正常推进合法。

/factr2/{side}/adapter_status 为 DiagnosticArray，status= factr2/adapter/{side}，包含 receive/source age、source stamp/skew、近1s输出频率、累计 invalid/stale/duplicate/skew/health_rejected/clock_reset/outputs、recent_rejection。Header 是生成时间；四路数据 stamp 始终为源时间。每原因告警最多1Hz，计数逐事件累计。

独立 W3 shell 运行健康检查器（不激活 NEXT venv、不 source FACTR2 install）：

```bash
cd /home/dingyj/w3_dual_arm_ws
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1 ROS_LOG_DIR="$PWD/log/ros"
/usr/bin/python3.10 /home/dingyj/factr2/factr2_w3_adapter/tools/w3_health_monitor.py
# 只要求右臂：附加 --ros-args -p 'sides:=[right]'
```

双方必须同 domain、兼容RMW、同 ROS clock（模拟时双方显式 use_sim_time=true），跨机器时双方 localhost-only=0。健康节点仅订阅 /w3_robot_bridge_node/state，发布 /factr2/w3_health；默认双方、50Hz、.25s timeout。左右 channel 固定0/1，7slot齐全唯一、finite、online、enabled、ERR=1才OK。DiagnosticArray stamp是生成时间，raw_source_stamp_ns是bridge发布时间，raw_receive_age_seconds是monotonic接收年龄，source age同时检查。原始反馈过期会ERROR，刷新diagnostic header不能掩盖过期。real adapter 健康缺失/过期/非OK就停发；恢复到OK后仍需新两源。

健康器只能检查 bridge 已报告的状态；不识别所有 online=true 的冻结反馈，不知道每电机真实接收时间。未改W3反馈联锁。

离线验收（只发 mock 状态，测试 domains 78/79/80 与生产 domain 隔离，使用 loopback XML）：

```bash
scripts/next.sh python -m pytest factr2_w3_adapter/test -q
scripts/next.sh python scripts/test_adapter_controller_stream.py --side left --seconds 60 --domain 78
scripts/next.sh python scripts/test_adapter_controller_stream.py --side right --seconds 60 --domain 79
scripts/next.sh python scripts/test_health_cross_environment.py
```

前两条 stream 测试调用 W3 已构建的 test/command_state_mock（实际 controller + double mock），不加载硬件。最后一条独立 W3进程注入原始missing/duplicate/offline/disabled/ERR8/NaN，并在NEXT real profile验证对应侧停发、另一侧不受影响、恢复和原始缓存过期。结果在 reports/generated/3，报告/证据清单在 docs/acceptance/。
