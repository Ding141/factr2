# 2. W3 插值后控制目标只读接口

## 目标、依赖和边界

在 W3 `JointPositionController` 中发布每周期已经写入 command interfaces 的位置/速度目标，让 NEXT 得到真实插值 q_cmd。依赖 [第 1 部分](1_environment_and_contract.md) 的契约；本阶段不需要 CAN。仅修改 W3 的位置控制器、相关依赖与测试，不改变轨迹插值和控制行为。

## 必读源码与当前行为

工作区：`/home/dingyj/w3_dual_arm_ws`。

- `src/ieir_controllers/include/ieir_controllers/joint_position_controller.hpp`。
- `src/ieir_controllers/src/joint_position_controller.cpp`：`on_configure`、`on_activate`、`on_deactivate`、`try_pickup_new_trajectory`、`sample_setpoint`、`update`。
- `src/ieir_controllers/CMakeLists.txt`、`package.xml`、`config/dual_arm_controllers.yaml`、`launch/dual_arm.launch.py`。
- `src/docs/CONTROL_PIPELINE.md`、`src/scripts/build_workspace.sh`；已有 `test/test_w3_migration.py` 主要是离线 Python 回归，不是该 C++ 控制器的运行验证。

当前 update 逐关节计算 `pos_des,vel_des`，更新 `hold_pos_`，写 position/velocity/stiffness/damping，尚无目标状态 publisher。默认双臂 14 关节，launch 单臂可裁成 7 关节；控制器不应硬编码 14 或 7。

依赖现状：package.xml 已显式声明 sensor_msgs/realtime_tools，CMake 也已 find_package；`joint_position_controller` target 的依赖已有 realtime_tools，但缺 sensor_msgs。不要无谓重复添加包级依赖，补齐 target 依赖和新增测试依赖。

**现存边界行为**：`sample_setpoint` 在 `t_elapsed<=0` 时会返回首点位置、速度 0；首点 time_from_start>0 且 t>0 时才走 pre-roll。未来开始时间与 t=0 的行为可能和直觉不同。本任务须测出并记录现状，发布值与实际 command interfaces 一致；不要借新增观测接口顺手改这段控制语义。若发现需要改变的运动问题，单独列为后续控制任务。

## 实施步骤

1. 在非实时 configure 阶段建立 `sensor_msgs::msg::JointState` publisher 和 `realtime_tools::RealtimePublisher`，默认话题为 `~/command_state`，默认控制器名下解析为 `/joint_position_controller/command_state`。如增加配置参数，保持此默认值并更新契约/配置。
2. 配置时预分配 message 的 name、position、velocity 数组，name 为 `joint_names_` 原始顺序，effort 始终为空。update 内只按索引写数值，避免为新发布路径分配内存、resize、拼字符串、普通 publish 或等待锁。
3. 在同一 update 内将最终写入 command interfaces 的 pos_des/vel_des 填入消息，不从 trajectory subscriber 或稀疏目标点发布，不另算一遍可能不同的插值。stamp=time；rad/rad/s；保持原 gains、effort 所有权和接口写入顺序。
4. 每周期尝试实时发布；`trylock` 失败直接跳过这一帧，接口仍照常写入并返回 OK。无需强求实际发布恰好 300 Hz，不让 NEXT subscriber 或 DDS 影响控制周期。
5. 只在 active/update 中发布；inactive/unconfigured 不制造新有效目标。生命周期切换后可能残留一帧 DDS 排队消息，以 stamp 判断，不将旧帧重新打时间戳。重新激活仍清轨迹，从当前实测姿态保持。
6. 新增 C++ controller / mock interface 测试，使用当前安装的 Humble API。必要时引入 ament_cmake_gtest；测试以手工 time 调用真实控制器采样/update，而不是在测试里复制插值算法。
7. 在 W3 控制链路文档补充消息字段、生命周期和 QoS；保持 NEXT 依赖在另一个仓库。

## 验证场景与验收标准

| 编号 | 场景 | 必须观察到的结果 |
|---|---|---|
| CMD-01 | active 无轨迹 | position=激活时测量姿态，velocity=0；消息与本周期 command interfaces 相等 |
| CMD-02 | 线性段中点 | 例 q=0→0.2 rad、1 s 段，中点目标=0.1、斜率=0.2 rad/s；实际消息等于接口值 |
| CMD-03 | 首点延迟 pre-roll | t>0 的多个采样按冻结的 pickup 姿态做线性坡道，不因 live hold_pos_ 变成指数响应 |
| CMD-04 | t<=0 / 未来 stamp | 覆盖当前返回首点、vel=0 的分支，记录此边界；观测接口不改变控制输出 |
| CMD-05 | 末点及之后 | 保持末点 position，velocity=0；stamp 随有效 update 前进 |
| CMD-06 | 部分/乱序关节、新轨迹替换 | 被约束关节按名字取值，未约束关节保持旧目标、vel=0；新轨迹语义与原实现一致 |
| CMD-07 | 双臂和单臂 | 14/7 关节输出 name 和数组长度正确；包含相应侧，命令接口无变化 |
| CMD-08 | 停用、重新激活 | 停用后不产生新活动目标；再次激活从当前测量保持，不消费 inactive 期间的旧轨迹 |
| CMD-09 | publisher 锁竞争 | 无订阅者/订阅者慢/锁不可用时，update 可正常写接口并返回；没有阻塞发布重试 |
| CMD-10 | 时间和内容 | stamp 精确为传入 update time，effort 空，测试数值有限，位置/速度误差容差 1e-9（double mock） |

- [ ] W3 相关包构建和新增 C++ 行为测试通过；原有 ieir_controllers 回归无新增失败。若基线失败，单列原因与改动无关的证据。
- [ ] 验证使用 mock state/command interfaces 或经过明确选择的仿真；不能用启动真实 `dual_arm.launch.py` 来冒充无硬件测试。
- [ ] 对新增实时发布路径完成代码审查：配置期分配，update trylock 跳过，未引入普通 publisher 等待、文件 IO、训练依赖。
- [ ] 建立真实 ROS subscriber 验证话题解析、QoS、字段和 stamp，不能只靠源码字符串断言验收。

构建/测试参考（W3 shell 无 NEXT venv；按当前工作区实际依赖选择）：

```bash
cd /home/dingyj/w3_dual_arm_ws
bash src/scripts/build_workspace.sh "$PWD" --packages-up-to ieir_controllers
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --base-paths src --packages-select ieir_controllers --event-handlers console_direct+
colcon test-result --verbose
```

## 交付与交接

提交 W3 变更及测试；FACTR2 中保存 `docs/acceptance/2_w3_command_state.md`，记录 W3 commit、构建命令、CMD-01..10 结果、QoS、真实话题名与未来 stamp 的现存边界。第 3 部分依赖该只读消息接口；真机观察在第 7 部分进行，不以 mock 声称实机时序已经验证。
