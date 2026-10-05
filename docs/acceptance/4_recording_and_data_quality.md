# 第 4 部分验收：录制与数据质量

2026-10-05，FACTR2 `dingyj-dev`。**结论：PASS，范围：PASS_OFFLINE**；所有来源显式 synthetic。软件基线 FACTR2 `a4eb34b`、W3 `d32262d`，本阶段变更随此报告提交，W3 仓库无改动。

| 验收条目 | 命令/实验 | 证据 | 结论 |
|---|---|---|---|
| REC-01 | `scripts/next.sh python scripts/test_recording_e2e.py --side left --domain 81 --seconds 60`；右侧改为 `right`、domain 82 | [4_summary.json](4_summary.json)：左 3000 行、右 2999 行，均单个连续 episode、约 50 Hz，四流严格检查通过 | PASS |
| REC-02 | 同一录制实验逐行按源 stamp 还原 q、独立 velocity、固定目标 q_cmd、effort | [4_summary.json](4_summary.json)：全部 7 维逐值比较 atol=2e-7 | PASS |
| REC-03 | 三次手动启停；`scripts/next.sh python scripts/test_recorder_shutdown.py` 交叉使用服务与伪终端 r 键，再发送 SIGINT | [4_evidence.json](4_evidence.json)：停录不增行、首帧晚于开启、退出 0、TTY 恢复、两个 episode 重开严格通过 | PASS |
| REC-04 | `scripts/next.sh python scripts/test_recording_e2e.py --side left --domain 83 --fault-test --session-label faults` | [4_summary.json](4_summary.json)：pause/NaN/rollback/skew 与三次开启形成 7 个片段；无零填充/重复时间/跨边界拼接，短段排除 | PASS |
| REC-05 | `scripts/next.sh python -m pytest factr2_next/src/factr2_next/test/test_w3_quality.py -q`；坏夹具严格检查 CLI | [4_evidence.json](4_evidence.json)：本阶段 21 项回归通过，坏 schema/维度/时间/有限值/频率/gap/哈希等返回 JSON 原因、CLI 退出 1 | PASS |
| REC-06 | 同一 quality 测试中的内容哈希 split、profile 和来源隔离实验 | [4_evidence.json](4_evidence.json)：三份非空 split 按 (内容 SHA256,episode) 去重；改名复制、来源重叠、错工具/side/order/version 拒收 | PASS |
| REC-07 | 配置/工具/证据清单检查；复现命令见 [录制说明](../data_collection.md) | 左右 `config/w3/{side}/record.yaml`、`config/w3/split.template.json`、[4_evidence.json](4_evidence.json) | PASS |
| REC-08 | 两包 `colcon build`；console scripts 与录制服务源码检查 | [4_evidence.json](4_evidence.json)：原四个入口保留、新增严格 CLI 与仅控制录制的 SetBool；无电机/模式客户端 | PASS |

开启边界测试发现 DDS 队列可能在开启后送达先前生成的帧，已增加源 stamp 门禁。数值 fixture 的 q_cmd 使用独立固定目标，符合 adapter 允许 state/command 小范围异步且以 state stamp 输出的契约。W3 模式按实际同步帧写入，单调 timer 负责断流和频率检查，上游非 W3 模式保留。

复现步骤见 [录制说明](../data_collection.md)。执行新功能测试、`colcon build` 两包成功、`ros2 run factr2_next next_check_h5 check ...` 严格 PASS。JSON 摘要见 [4_summary.json](4_summary.json)，证据路径/大小/SHA256 见 [4_evidence.json](4_evidence.json)。完整日志、H5、侧车和 CLI JSON 留在项目内 `reports/generated/4/`、`data/mock/`，由 Git 忽略。

真实传感器、真实无接触数据、控制增益有效性和冻结反馈问题属于第 7/8 部分。侧车不能替代实际健康门禁和 rosbag 审计。
