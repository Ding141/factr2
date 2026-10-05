# 4. 录制链路、元数据与严格数据质量门禁

## 目标与依赖

在无硬件条件下完成 mock → adapter → recorder → H5 → 质量检查的闭环，使第 7 部分可以直接采集有效数据。依赖 [1](1_environment_and_contract.md)、[3](3_adapter_and_health.md)。本阶段实现数据工具和录制集成，不采真机、不训练模型。

## 必读代码与不足

目录前缀：`factr2_next/src/factr2_next/factr2_next/`。

- `data_collection/recorder_node.py`：ApproximateTimeSynchronizer 回调缓存样本，独立 timer 写入；相同 stamp 不重复写，接收超过 sample_timeout 后跳过。缺 publisher 目前只是 warn，不阻止开启 episode。
- `data_collection/h5_writer.py`：schema=`factr2_next_h5_v1`，四 stream 各 `data/timestamps`；每次 append 写同一个 stamp。文件根只有 schema/session_name/created_at 元数据。
- `data_collection/check_h5.py`：已有有限数、每流数据/时间行数和多流行数检查；不严格要求四 key、7 维、schema 值、正整数纳秒、严格递增和流间同 stamp；低频/大缺口主要展示/告警，不是强制失败。
- `training/dataset.py::_episode_arrays` 会以最短流长度截断，且不读取 timestamps；不应依赖它纠正坏数据。第 5 部分补训练入口拒收，本部分提供统一质量检查能力。
- `setup.py` 未注册 check_h5 console script，当前可用 `python -m factr2_next.data_collection.check_h5`。

## 文件范围与实施步骤

1. 使用 `config/w3/{left,right}/record.yaml`，无前缀四键、正确字段、50 Hz target / 45 Hz min、slop=30 ms、sample_timeout=0.2 s。采用第 3 部分实际 adapter，不另写一个只输出“理想 NEXT 四流”的工具替代端到端集成。
2. 提供无硬件 W3 标准消息 mock 工具及确定性 seed。mock 可控制源频率、乱序、缺失、断流、skew、非有限值、时钟回退，使用隔离的测试 domain；支持 left/right。复用到第 6 部分。
3. 提供可自动开始/停止 episode 的测试入口。可对真实 RecorderNode 调用 toggle 并驱动 executor，或新增明确的 record start/stop 服务；该服务只能操作录制状态。保留 `r` 交互，不能用另一个简化 writer 代替 recorder 的实际路径。
4. 新增 W3 严格检查模式/模块，复用通用 check_h5，并提供 CLI、JSON 结果与非零失败退出码。普通上游模式可保留；W3 training/real-data 使用 strict 模式。共享检查逻辑供第 5 部分调用，避免各处校验规则分叉。
5. recorder 在 W3 模式写前验证四路内容/长度/stamp；防止回退 stamp 写入，以及 episode 开始时把切换前缓存当首帧。丢帧和长缺口需标记并切成新连续片段，或终止该 episode 并报告，不用“删除坏行后继续”拼接时间历史。
6. 完成 session/episode 元数据 sidecar（JSON/YAML），不需要破坏原 H5 schema。元数据包括 contract_version、side、joint_order、采样率、输入 topic/field、软件 commits、配置/标定文件 SHA256、工具/夹爪/负载、控制增益/前馈配置、轨迹 ID、接触标记、时间/温度来源、episode 排除原因。
7. 提供 session 清单与 split-manifest 生成/检查工具。`(H5 内容哈希,episode)` 唯一标识样本来源，同内容复制文件不能绕过隔离；整个 session/episode 划分，优先独立文件。输出明确的 train/val/test 清单和 dataset_id。
8. 在 `docs/data_collection.md` 写录制、故障中止、检查、清单生成、只读审计命令；原始 state/command_state/health/adapter diagnostics 的 rosbag 审计路径写入清单。rosbag 自定义 msg 在独立 W3 shell 记录，不要求 NEXT venv 加载 W3 overlay。

## 严格 H5 接受规则

| 项目 | W3 strict 要求 |
|---|---|
| 根属性 | schema 精确匹配 `factr2_next_h5_v1`；session/created_at 有值，sidecar 关联文件哈希 |
| 结构 | episode 四 key 齐全，分别存在 data/timestamps；额外元数据允许，但不能静默选错 stream |
| 数据 | 每路 `[N,7]`，float 数值有限，四流 N 相等；输入名称/顺序由 sidecar 契约确认 |
| 时间 | 每路 timestamps 为 `[N]` int64 纳秒，正值、严格递增；四流整列完全相同 |
| 时长 | H=50 至少需 N>=50；正式训练 episode 推荐 >=10 s，短段剔除原因要记录 |
| 频率 | `(N-1)/(末stamp-首stamp)` >=45 Hz；正常 50 Hz profile 不应超过 55 Hz，报告 timer 接收频率与源频率 |
| 缺口 | 正式训练连续片段 max dt<=40 ms；出现更长缺口切 episode/排除，不跨缺口组窗 |
| 来源 | real 数据必须有健康门禁启用、工具/负载/标定/接触记录；mock 数据显式标为 synthetic |

启停和故障边界可产生短片段，这些片段可以保留原始记录，但不进入“已接受训练集”。严格校验不靠文件名推断 side，不从数值猜 joint_order。字段内容与名字的来源约定必须有配置和 mock 证据支撑。

## 验收标准

- [ ] **REC-01**：连续 60 s mock 经真实 adapter/recorder 录得至少 2700 行；四流 shape=[N,7]、时间完全相同，strict PASS；左右各执行一次。
- [ ] **REC-02**：已知 mock 数值能在 H5 对应行还原，证明 joint_vel 取 velocity、力矩取 effort，q_cmd 是目标位置；不能只验证 shape。
- [ ] **REC-03**：录制启停至少 3 个 episode，未录制时不写，episode 首帧来自开启后的新样本，恢复终端和 Ctrl-C 关闭文件后 h5py 可正常重开。
- [ ] **REC-04**：注入断流/坏输入时不补零、不重复写旧 stamp；恢复时长缺口被切段/排除，strict 失败样本不能进入 manifest。
- [ ] **REC-05**：损坏夹具分别覆盖缺 key、错 schema、6/8 维、异长、非 int64 时间、重复/回退时间、流间 stamp 偏移、NaN/Inf、低频、长缺口。每个严格失败都有机器可读原因和非零退出码。
- [ ] **REC-06**：train/val/test manifest 按来源隔离，交集为空；复制同 H5 改名、重叠 episode、错误 side/joint_order/contract_version 会被拒绝。
- [ ] **REC-07**：产出可复现 e2e 命令、metadata 示例、mock manifest、JSON 检查报告；不把 synthetic 数据混标为真机无接触数据。
- [ ] **REC-08**：采集工具未新增电机命令/模式切换；键盘交互兼容，当前四个上游 console scripts 仍可用。

原始检查入口可作为辅助：

```bash
python -m factr2_next.data_collection.check_h5 /home/dingyj/factr2/data/mock/recording.h5
```

W3 strict 模式的确切命令由本阶段实现并写入报告。仅得到上游宽松检查器的 PASS 不满足 REC-05。

## 交付与交接

提交工具、必要的 recorder/checker 修正、左右配置、测试、元数据与 split-manifest 模板、`docs/acceptance/4_recording_and_data_quality.md`。H5 fixture 可运行时生成到 tmp/data，不提交二进制。交给 5 的输入是严格校验 API 和独立合成 train/val 文件；交给 7 的输入是可直接执行的数据采集与审计流程。
