# 第 3 部分：状态适配与反馈健康验收

结论：**PASS**，范围 **PASS_OFFLINE**。日期 2026-10-05（Asia/Shanghai）。全部为 mock / 模拟时钟 / double interfaces，无真机运动、CAN 或控制模式操作。

实际基线：FACTR2 `a334ab1`（阶段1 `bf18abb`），W3 `d32262d`（阶段2），两个 dingyj-dev 工作区开始时干净。W3 本阶段无差异。FACTR2 新增完整 ament_python adapter、左右 mock/real 配置、launch、独立系统 Python 健康器、纯验证逻辑和自动测试；更新运行说明/状态链接/整体规划终端分工，修正第1报告的模块计数为13。本报告所在提交为第3部分交付 commit，避免自引用 SHA。

环境：NEXT /home/dingyj/factr2/.venv/bin/python，Humble→venv→FACTR2 overlay；健康器 /usr/bin/python3.10，Humble→W3 overlay；无NEXT/W3 overlay混用。core无torch或W3自定义消息导入，健康器无torch。ROS日志、build/install、pytest结果和原始证据仅在两个项目忽略目录。已安装ROS入口 `factr2_w3_adapter w3_next_adapter`，launch和4个ROS参数配置均可发现。

所有NEXT命令从 /home/dingyj/factr2 经 scripts/next.sh 执行。mock DDS为 rmw_fastrtps_cpp，双方 localhost-only=1、loopback UDP profile；单元/短集成 domain73，左右60秒 domain78/79，跨环境健康 domain80，CLI domain81。未测试跨机器或真机反馈。

| 验收条目 | 命令/实验（退出码） | 证据 | 结论 |
|---|---|---|---|
| ADP-01 | colcon adapter测试，两输入独立乱序、14关节+夹爪、左右同时运行（0） | test_ADP01_02_fields_names_static_and_qos + unit independent_orders；字段/name正确，侧别不混 | PASS |
| ADP-02 | 静态值+推进stamp，单臂7关节实际C++ mock（0） | stream_left/right.json；合法静态不被判断流 | PASS |
| ADP-03 | 缺/重复名、短向量、NaN/Inf、非法目标effort；清缓存、单源不恢复、两新源恢复（0） | test_validation + 实际pub/sub bad_messages；invalid/recent_rejection与原因匹配 | PASS |
| ADP-04 | 缺源、独立ROS源年龄/monotonic接收年龄超时；模拟clock暂停（0） | test_ADP04_missing_and_independent_ages、实际timeout/paused-clock测试；STEADY_TIME timer 20ms检查 | PASS |
| ADP-05 | skew>30ms、零/重复/回退/未来stamp、ROS clock回退（0） | test_ADP05_stamp_rejection/skew_and_clock_rollback，真实/模拟时钟pub/sub测试 | PASS |
| ADP-06 左 | python scripts/test_adapter_controller_stream.py --side left --seconds 60 --domain 78（0） | stream_left.json；实际controller 300Hz→独立adapter→subscriber，60.00985s，每路3000帧，49.99179Hz，最大间隔26.766706ms | PASS |
| ADP-06 右 | 同上 --side right --seconds 60 --domain 79（0） | stream_right.json；60.01177s，每路3000帧，49.99019Hz，最大间隔26.785797ms | PASS |
| ADP-07 | 状态停止而command推进（0） | test_ADP02_static_advancing_stamps_and_ADP07_no_resampling；实际 timeout_skew_future_and_no_old_resampling；不重复已输出状态 | PASS |
| ADP-08 | real profile无健康/陈旧/错误侧/非OK；故障左侧，右侧仍健康；恢复再等新两源（0） | 实际health_real_side_isolation + pure health_gate_side_timeout_error_recovery + 跨环境健康报告 | PASS |
| ADP-09 | python scripts/test_health_cross_environment.py（0） | missing/duplicate/offline/disabled/ERR8/NaN左侧均非OK且左侧0帧、右侧7–8帧；ERR1恢复；右臂单独channel=1正常 | PASS |
| ADP-10 | 同上，joint_states始终新鲜，原始反馈offline/停止（0） | health_cross_environment.json：offline_left停发左、右持续；raw_timeout双侧停止，raw age约.4966s、raw_stale；新诊断header不掩盖raw缓存年龄 | PASS |
| QoS/下游兼容 | 实际endpoint检查、NEXT ApproximateTimeSynchronizer订阅四流（0） | integration_qos.json；三输入与四输出BEST_EFFORT/VOLATILE/KEEP_LAST depth5；message_filters 同stamp、velocity/effort字段保真 | PASS |
| 包/测试入口 | python -m colcon build --base-paths factr2_next/src/factr2_next factr2_w3_adapter --symlink-install；colcon test --base-paths factr2_w3_adapter --packages-select factr2_w3_adapter --event-handlers console_direct+；colcon test-result --test-result-base build/factr2_w3_adapter --verbose（均0） | build_final.log，package_identification.txt=ros.ament_python，executables.txt；colcon_test/results：57 passed，0 errors/failures/skips | PASS |
| 隔离/只读graph | 直接导入core；跨环境实际node endpoint与client检查（0） | isolation.txt、health_raw/monitor.log；health_cross_environment.json.graph和stream JSON；无command/trajectory publisher，无client | PASS |
| 启动/动态参数/退出 | python scripts/test_adapter_cli.py（0）；单元及ROS set_parameters（0） | cli.json、cli_launch.log：缺side/频率0/负timeout/错root退出1为预期拒绝，launch获诊断、SIGINT退出0；只读参数修改拒绝 | PASS |

四流各自name=对应侧0..6，position/velocity/position/effort长度7、未用字段为空；每路源stamp严格递增，四路接收stamp序列完全相同。只发布新状态源stamp，允许静态值，未重复标定/变号。adapter_status含各源年龄/skew、源stamp、近1s频率和累计原因计数，mock/no-hardware-health明确；告警每原因最多1Hz，计数不被限频丢弃。

报告摘要见 [3_summary.json](3_summary.json)。所有证据绝对路径、SHA256和大小见 [3_evidence.json](3_evidence.json)，原始证据在 reports/generated/3，pytest.xml在build/factr2_w3_adapter。左右stream fixture均使用W3提交d32262d的实际JointPositionController，无trajectory或电机command publisher；所调用binary的SHA256同样在清单。

开发中修复：Humble DiagnosticStatus.level实际为byte，core兼容b'\x00'（纯mock整数0）；package.xml维护者email须有合法域，修正后由普通Python识别恢复为ament_python并重新构建/验收；测试NaN定位到本侧、模拟暂停时钟用单个合法stamp避免重复消息先清缓存。最终无失败项。长测记录在最终诊断字段补充和包索引修正前完成，数据选择/发布逻辑未变；随后正式ROS包测试和健康跨环境测试重跑通过。

后续输入：第4部分可使用已验证四流/diagnostic/断流恢复行为，但上游recorder仍需该阶段的连续性与质量门禁；第6部分需history窗口在断流/时钟回退时复位。第7部分由现场操作者做真机自由空间/反馈观察。此PASS不证明真实外力精度、CAN实际接收时间、所有online=true冻结故障，或硬件闭环联锁；真实profile默认健康门禁启用。硬件实验按任务留第7/8部分执行。
