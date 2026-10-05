# W3 录制和数据门禁

所有命令在 FACTR2 根目录通过 `scripts/next.sh` 执行。该脚本仅加载 Humble、项目 `.venv` 和 FACTR2 overlay；W3 bridge/rosbag 使用独立的 W3 shell。产物放 `data/`、`reports/generated/`，不修改系统 Python 或 shell 配置。

复制 `config/w3/{side}/record.yaml` 到项目数据目录，填写 session、工具、负载、标定文件、控制增益、前馈、轨迹、温度来源和接触标签。真实来源必须开启健康门禁并提供可哈希标定文件。mock 必须标为 `source: synthetic`、`health_gate_enabled: false`，不能当作真机数据。

启动第 3 部分适配器后执行：

```bash
scripts/next.sh ros2 run factr2_next next_record --ros-args -p config_file:=/home/dingyj/factr2/data/left/session/record.yaml
```

TTY 中 `r` 启停；自动化可以调用仅控制录制状态的 `/factr2/left/record` (`std_srvs/SetBool`)。缺 publisher 或无新鲜合法同步帧会拒绝开启。每次开启清空缓存和同步队列，第一帧必须在开启后收到。Ctrl-C 刷新、关闭 H5 并恢复终端，生成同名 `.metadata.json`。

W3 四流每帧检查名称顺序、7 维、有限浮点、未使用字段为空、同一正整数源 stamp 和源年龄。每个合法同步帧直接写入，adapter 已承担 50 Hz 采样。超过 40 ms 的源缺口、回退、坏帧或单调时钟接收超时使当前连续片段结束，恢复开启新 episode。短于 50 行的片段保留原始记录并标记排除原因；训练不跨 episode 拼接。停录期间不写数据。

```bash
scripts/next.sh ros2 run factr2_next next_check_h5 check data/left/session/file.h5 --json reports/generated/quality.json
scripts/next.sh ros2 run factr2_next next_check_h5 manifest --dataset-id w3_left_v1 --output data/left/split.json --train data/left/train.h5 --val data/left/val.h5 --test data/left/test.h5
scripts/next.sh ros2 run factr2_next next_check_h5 verify-manifest data/left/split.json
```

`check` 默认检查全部 episode，短段导致失败；可用 `--episodes ep_0000 ep_0002` 明确选择。manifest 自动选择未标记排除的片段，然后严格检查全部选中数据。失败输出 JSON 原因且退出 1。manifest 验证三份非空 split、每个 `(H5 SHA256, episode)` 唯一、配置契约和工具/负载/标定一致以及无接触标签。复制 H5 改名不能绕过隔离。manifest 路径为绝对路径，移动数据后须重新生成。

H5 根保留 `factr2_next_h5_v1` schema。sidecar 保存源类型、输入 topic/field、关节顺序、50 Hz、两个软件 commit、录制配置 SHA256、标定 SHA256、来源记录、episode 边界原因及关闭后的 H5 SHA256。文件修改后哈希不匹配会拒收。侧车的健康声明是采集配置记录，实际门禁由 adapter 执行；真实录制应保留审计证据。

审计 rosbag 在 W3 独立环境只读记录 `/joint_states`、`/joint_position_controller/command_state`、`/factr2/w3_health`、两侧 adapter diagnostics，并把 bag 路径写入 `metadata.audit_paths`。自定义 raw msg 在该 shell 中可解码，NEXT 环境不加载 W3 overlay。采集前用 `ros2 topic info -v` 检查来源和 QoS，采集后只读检查 H5 和 sidecar。

无硬件复现：

```bash
scripts/next.sh python scripts/test_recording_e2e.py --side left --domain 81 --seconds 60
scripts/next.sh python scripts/test_recording_e2e.py --side right --domain 82 --seconds 60
scripts/next.sh python scripts/test_recording_e2e.py --side left --domain 83 --fault-test --session-label faults
scripts/next.sh python -m pytest factr2_next/src/factr2_next/test/test_w3_quality.py -q
```

`w3_mock.py` 发布标准 state 和 command_state，seed 固定，支持 `--hz` 与 `--faults '[{"at":5,"duration":0.3,"kind":"pause"}]'`。故障种类包括 pause、missing、skew、nan、rollback、duplicate、order；乱序 names 的重排仍可能被 adapter 正确按名映射，不能据此判定物理映射正确。该工具没有电机命令和模式切换接口。
