# 第 4 部分验收：录制与数据质量

2026-10-05，FACTR2 `dingyj-dev`。无硬件开发验收完成；所有来源显式 synthetic。软件基线 FACTR2 `a4eb34b`、W3 `d32262d`，本阶段变更随此报告提交。

| 验收项 | 结果和证据 |
|---|---|
| REC-01 | 左右各运行 60 s，标准 W3 mock → 实际 adapter → 实际 RecorderNode；左 3000 行、右 2999 行，均单个连续 episode，四流严格 PASS，平均约 50 Hz |
| REC-02 | 逐行按源 stamp 还原 q、独立 velocity、固定目标 q_cmd、effort；全部 7 维逐值比较 atol=2e-7 |
| REC-03 | 三次启停；停录前后总行数不变；每次手动开启首帧源 stamp 晚于开启；录制服务和伪终端 r 键交叉启停，SIGINT 退出 0、TTY 原样恢复、两个 episode H5 重开严格 PASS |
| REC-04 | pause/NaN/rollback/skew 经 adapter 门禁触发停止输出，recorder 保存 7 个连续片段（4 个故障边界及 3 次手动开启），不写零、不复用时间戳、不跨边界拼接；短片段自动排除 |
| REC-05 | 21 个回归测试通过：缺 key、schema、6/8 维、异长、float 时间、重复/回退、流间差异、NaN/Inf、低频、gap、哈希和 metadata；坏夹具 JSON 原因和 CLI 退出 1 |
| REC-06 | 三份非空 split 以 (内容 SHA256,episode) 去重；复制文件改名、来源重叠和不同工具 profile 拒收；错 side/order/version 严格拒收 |
| REC-07 | 可复现命令、左右 metadata 模板、split 模板、逐项 JSON 报告及内容哈希清单已交付 |
| REC-08 | 四个原 console scripts 保留；新增严格检查 CLI 和仅控制录制的 SetBool 服务，无电机/控制模式客户端 |

开启边界测试发现 DDS 队列可能在开启后送达先前生成的帧，已增加源 stamp 门禁。数值 fixture 的 q_cmd 使用独立固定目标，符合 adapter 允许 state/command 小范围异步且以 state stamp 输出的契约。W3 模式按实际同步帧写入，单调 timer 负责断流和频率检查，上游非 W3 模式保留。

复现步骤见 [录制说明](../data_collection.md)。执行新功能测试、`colcon build` 两包成功、`ros2 run factr2_next next_check_h5 check ...` 严格 PASS。JSON 摘要见 [4_summary.json](4_summary.json)，证据路径/大小/SHA256 见 [4_evidence.json](4_evidence.json)。完整日志、H5、侧车和 CLI JSON 留在项目内 `reports/generated/4/`、`data/mock/`，由 Git 忽略。

真实传感器、真实无接触数据、控制增益有效性和冻结反馈问题属于第 7/8 部分。侧车不能替代实际健康门禁和 rosbag 审计。
