# W3 可复现训练与离线评估

使用项目 `scripts/next.sh`，训练和评估不需要 ROS graph 或 W3 进程。Matplotlib 的缓存固定在项目 `log/matplotlib`，数据、模型、图表和日志留在忽略目录。PyTorch CPU 运行时使用配置声明的线程数；请求不可用 CUDA 会直接失败，禁止静默回退。

正式配置 `config/w3/{left,right}/train.yaml` 填写 `data.manifest` 和 run 名称。W3 必须使用第 4 部分生成的三份非空 split，训练前验证数据质量、内容哈希和来源隔离、关节/侧别/工具/负载/标定 profile。禁止随机 window split，禁止按最短流截断。独立 session 优先；同文件不重叠 episode 仅适合开发测试。

```bash
scripts/next.sh python -m factr2_next.training.train --config config/w3/left/train.yaml
```

每个 episode 单独生成 N-H+1 个 `[50,21]` 窗口，特征为 `[q,qdot,q_cmd-q]`，标签取末帧 7 维 measured torque。只缓存 step 数组，DataLoader 取一个窗口时才复制；归一化按每帧在训练窗口中的出现次数加权，等价于堆叠窗口后计算均值/std。x/y std 均加 `1e-6`，计算使用 float64 后存 float32；验证/测试只复用训练统计。baseline 固定为训练集所有有效窗口末帧标签的每关节平均值。

`data.memory_budget_mb` 限制缓存预算，不能替代整个 Python/PyTorch 进程的 RSS 预算。1 小时 50 Hz 的 180000 行用 step cache 20.16 MB；窗口全量复制约 755.79 MB。代表性检查命令 `scripts/next.sh python scripts/test_training_resources.py` 包括严格检查、加载、加权归一化和首/中/末窗口比较。

默认 stateless LSTM 架构保持不变，2 层 hidden=128、head hidden=256/2、dropout=0.1。训练保存独立 validation normalized MSE 最低的 epoch（并列取最早）。test 不参与归一化或 checkpoint 选择。run 中包含：

- `model.pt`：最佳模型 state_dict 和尺寸/架构；`config.yaml`：解析后绝对 manifest 路径和真实 device。
- `normalization.npz`：21 维 x mean/std、7 维 y mean/std；`metrics.json`：每 epoch train/val、best_epoch、选择规则、window 数、时间、RSS/内存估算。
- `metadata.json`：w3_checkpoint_v1、side/order/features/H/21/7/50 Hz、dataset ID、manifest SHA256、工具/负载/标定、源码 commit、软件版本、seed/真实 device、四份 artifact SHA256。

加载器校验 artifact 完整性、metadata/config/权重尺寸一致、归一化有限且 std>0 和运行侧别/历史/频率。使用 torch 2.5.1 的 `weights_only=True` 读取本项目受信 artifact；哈希用于完整性，不代表来自未知来源的模型可信。

离线评估默认使用模型 config 的 data.keys 和 manifest，不需要 `--arm`，不限制前 1000 行：

```bash
scripts/next.sh python -m factr2_next.inference.offline_eval --run-dir runs/SET_RUN --split test --output reports/generated/test_eval
scripts/next.sh python -m factr2_next.inference.offline_eval --run-dir runs/SET_RUN --h5-path data/left/session.h5 --episode ep_0000 --output reports/generated/explicit_eval
```

`--h5-path`、`--episode` 可多次传入；未指定 episode 则选文件所有 episode，W3 全部须严格通过。直接 H5 模式标为 explicit_h5，来源/工具仍必须匹配模型。非 W3 原显式双臂 config 使用 `{arm}_...` keys 时传 `--arm left/right`。原必填 arm 和隐式 1000 步限制已取消；`--max-steps` 仅用于用户明确要求的调试截断。`--output` 现在必填，`--no-plots` 可禁用图表。

`report.json`、`metrics.csv` 给出 Nm 反归一化后的 per-joint RMSE/MSE、signed bias、残差 std、绝对 P95/P99、数量/有效时间。残差符号明确为 measured-predicted；MSE 单位 Nm²。`samples.csv` 保留每一帧源 stamp、标签/预测/残差。报告按 session、trajectory、speed norm(<0.5 rad/s 或 >=0.5)、pose norm(<0.5 rad 或 >=0.5) 分组。误差曲线和去均值 FFT 幅度谱分别在单个连续 episode 中计算，不跨间断拼频谱。baseline 在完全相同样本上比较。

物理验收使用 `--acceptance config/w3/SET_FROZEN_ACCEPTANCE.yaml`。模板含 null 时结果 UNCONFIGURED、CLI 退出 2；不允许自动 PASS。第 7 部分必须在 pilot/val 阶段冻结版本和物理门槛。false-positive time rate 使用每个 episode 内 EMA+L1/scale+迟滞，边界重置；sigma_floor 用于诊断 sigma 的下限展示，不改变已经冻结的模型归一化。报告每个指标和 baseline improvement，再给出 PASS/FAIL。

本次可复现 synthetic smoke：

```bash
scripts/next.sh python scripts/prepare_w3_smoke.py
scripts/next.sh python -m factr2_next.training.train --config config/w3/left/smoke_train.yaml
scripts/next.sh python -m factr2_next.training.train --config config/w3/right/smoke_train.yaml
scripts/next.sh python scripts/evaluate_w3_smoke.py --side left
scripts/next.sh python scripts/evaluate_w3_smoke.py --side right
scripts/next.sh python -m pytest factr2_next/src/factr2_next/test/test_w3_quality.py factr2_next/src/factr2_next/test/test_training_contract.py -q
```

smoke 配置只有 epoch=2、batch=64、CPU 单线程等开发设置，架构仍是契约基线。合成结果只能用于工具正确性验收，不能用于真机外力估计结论。
