# 3. W3 状态适配与只读反馈健康检查

## 目标与依赖

将标定后的 W3 状态和控制器实际目标转换为单臂 NEXT 四路输入，并在真机 profile 中阻止已被 bridge 报告为无效的电机反馈进入数据/模型。依赖 [1 的契约](1_environment_and_contract.md) 与 [2 的 command_state](2_w3_command_state.md)。可先用 mock 实现，最终需和第 2 部分接口联调。

只修改 FACTR2。不控制电机，不改 W3 hardware interface。左右各一个 adapter 实例，单臂缺另一臂输入不应阻塞本臂。

## 实际代码问题与阅读入口

- `factr2_w3_adapter/` 目前只有空目录，需要完整 ament_python 包。
- 对照上游 `factr2_next/src/factr2_next/setup.py`、`package.xml` 和 recorder/inference 的 topic/field 提取。
- W3 `src/ieir_controllers/src/dm_hardware_interface.cpp::on_motor_state/read`：offline 或非有限反馈不会更新缓存，read 仍返回 OK。上层 broadcaster 可继续发布缓存值。
- W3 `src/W3_ROBOT/w3_robot_bridge/msg/{MotorState,MotorStateArray}.msg`。
- W3 `src/W3_ROBOT/w3_robot_bridge/src/{w3_robot_bridge_node,can_channel}.cpp`：bridge 的 state 和 motor header stamp 是发布时间，`online` 才反映 bridge 的反馈超时判定；不能把该 stamp 当逐电机接收时刻。
- W3 `src/docs/CONTROL_PIPELINE.md`：左 channel=0、右=1，slot=0..6；正常使能 ERR=1，不能把所有非零 error_flags 都判为故障。

## 建议文件与包组织

```text
factr2_w3_adapter/
  package.xml / setup.py / setup.cfg
  resource/factr2_w3_adapter
  factr2_w3_adapter/__init__.py
  factr2_w3_adapter/adapter_node.py
  factr2_w3_adapter/validation.py
  launch/adapter.launch.py
  config/left_mock.yaml / right_mock.yaml / left_real.yaml / right_real.yaml
  tools/w3_health_monitor.py
  test/test_validation.py / test_adapter_integration.py / test_health_monitor.py
```

建议 console script `w3_next_adapter`。注册 resource marker、配置/launch 安装规则、ROS 依赖和测试依赖。核心包仅依赖标准 `rclpy,sensor_msgs,diagnostic_msgs` 等，不依赖 torch，不依赖 W3 自定义消息。

健康检查器是独立脚本，**在系统 Python 3.10 + Humble + W3 overlay shell 中直接执行 FACTR2 源文件**。它读取 W3 自定义 msg，发布标准 DiagnosticArray，NEXT shell 无需 source W3 overlay，也无需把 W3 包装进 venv。

## adapter 实施要求

1. 参数包括 side、state_topic、command_state_topic、output_root、publish_hz、input_timeout_seconds、max_source_skew_seconds、health_topic、require_hardware_health、hardware_health_timeout_seconds。负/零频率、无效 side、非法 timeout/root 在启动时拒绝；参数动态更新可暂不支持，但要明确。
2. 状态输入 position/velocity/effort 长度必须与 name 相等；目标 position/velocity 长度与 name 相等。检查必需关节各出现一次、无重复名字、所用数值有限；extra joint 合法但也不能造成字段长度错配。分别按名字映射两源，不能复用某个数组下标假定顺序一致。
3. 遵守第 1 部分的两类年龄、源 skew、stamp 回退/重复、50 Hz 抽样规则。缓存无效的新消息后，不继续使用上一次好消息伪装输入恢复；重新收到两路合法输入才恢复。超时停发有效四流，不补零、不外推。
4. 四流同 stamp、同 name；输出 velocity 存在 `.velocity`、力矩存在 `.effort`，不是 Piper 的“所有值塞 position”。适配器不重复做符号/零位变换，不从 raw state 生成 q/tau。
5. BEST_EFFORT 的 sensor-data 输入可兼容常见 W3 publisher；四路输出与 NEXT message_filters sensor-data 订阅兼容。交付实际 endpoint QoS 检查结果，不能仅写“使用默认 QoS”。
6. 输出诊断，例如 `/factr2/{side}/adapter_status` DiagnosticArray：各源 age/skew、有效输出频率、累计 invalid/stale/duplicate/skew/health_rejected 计数、最近原因。告警限频，建议每原因最多 1 次/s；计数不因告警限频漏记。
7. `require_hardware_health=true` 时，指定侧健康 status 必须存在、级别 OK、消息新鲜；否则停发四流。false 只用于 mock profile，诊断明确标记 mock/no-hardware-health。

## 健康检查器实施要求

1. 只订阅 `/w3_robot_bridge_node/state`，不建立 W3 command publisher/service client。发布 `/factr2/w3_health`，默认 50 Hz；基于缓存的原始反馈也需超时。
2. 按 `(channel,motor_index)` 检查指定侧 7 个 slot 齐全且唯一、position/velocity/torque 有限、online=true、enabled=true、error_flags=1。missing/duplicate/offline/disabled/fault/stale 任一项使对应侧非 OK；只开右臂时保留 channel=1。
3. diagnostic 至少含 side、missing_slots、offline_slots、fault_slots、raw_source_stamp_ns、raw_receive_age_seconds、healthy。header 为诊断生成时间，但不断刷新 header 不得掩盖 raw 缓存超时。
4. bridge header 只证明 bridge 在发布；online 的准确性由现有 bridge 负责。本任务不升级 W3 反馈联锁，不声称能识别所有“硬件值冻结但 online=true”的失效。

## 验收标准

| 编号 | 输入/条件 | 预期结果 |
|---|---|---|
| ADP-01 | 两源各自乱序、14 关节、含夹爪 | 本侧四流严格 0..6，逐关节数值对齐正确，不混侧、不取夹爪 |
| ADP-02 | 单侧 7 关节、静态但 stamp 推进 | 正常发布，不把位置不变误判成断流 |
| ADP-03 | 缺关节、重复名、短数组、NaN/Inf | 本周期无有效四流，诊断原因和计数正确；新合法消息后恢复 |
| ADP-04 | 未收到一源、接收/源 stamp 超时 | 无有效四流；超过 0.25 s 后最迟下一个 20 ms timer 周期拒收 |
| ADP-05 | skew>30 ms、旧/重复/回退/未来 stamp | 拒收；时钟回退清缓存；不能靠统一 stamp 变成有效帧 |
| ADP-06 | 两源正常 300 Hz、连续 60 s mock | 四流接收频率均 45..55 Hz，stamp 完全相同且严格递增，每字段 7 维、name 正确 |
| ADP-07 | 状态停止、目标继续推进 | 不重复旧状态；不能只靠目标 stamp 维持 50 Hz 假样本 |
| ADP-08 | 健康门禁启用、缺健康/过期/非 OK | 拒收对应侧；另一侧健康不受影响；门禁恢复后重新取合法输入 |
| ADP-09 | 原始 motor 状态 missing/duplicate/offline/disabled/ERR=8 | 健康检查器输出非 OK；ERR=1 且其它条件正常为 OK |
| ADP-10 | `/joint_states` 仍新鲜但 raw offline | real profile 停发；验证缓存反馈不会混入 NEXT 数据 |

- [ ] 无硬件单元测试和真实 rclpy pub/sub 集成测试覆盖上述场景。可让 NEXT mock 与健康 mock 分属不同进程验证标准诊断互通。
- [ ] 核心 adapter 可在不加载 W3 overlay 的 NEXT 环境导入、运行；健康脚本可在 W3 系统环境运行，不加载 torch。
- [ ] `ros2 node info` / 源码检查确认两个节点没有 trajectory、motor commands publisher、enable/disable 或 switch_controller client。
- [ ] 启停、异常输入和 Ctrl-C 正常退出；没有启动 bridge、切换控制器、修改标定或 CAN 配置。

运行入口应由本阶段实际提供，例如：

```bash
# NEXT shell，包构建安装后；配置是 ROS 参数格式
ros2 run factr2_w3_adapter w3_next_adapter --ros-args --params-file /home/dingyj/factr2/factr2_w3_adapter/config/left_mock.yaml
# 独立 W3 shell，不激活 NEXT venv
/usr/bin/python3.10 /home/dingyj/factr2/factr2_w3_adapter/tools/w3_health_monitor.py
```

## 交付与交接

提交完整 ROS 包、mock/real 左右参数、健康工具、行为测试和 `docs/acceptance/3_adapter_and_health.md`。交接给 4 的是经验证的标准四流与 invalid/health 诊断；交给 6 的是断流/恢复语义。真机 profile 默认不能关闭健康门禁来绕过不合格反馈。
