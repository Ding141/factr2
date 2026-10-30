# 当前 8082：冻结均值与学习 sigma 的只读观察部署

2026-10-23 已按用户要求部署。页面：<http://127.0.0.1:8082/>。

模型包：`/home/venom/factr2/models/w3/right/frozen_sigma_observer_20261023`。均值使用上一轮 NEXT_MSE 的验证最佳 epoch 13，sigma 使用冻结均值试验的验证 NLL 最佳 epoch 4。原始关节外力矩定义仍为 `measured - predicted_free_mean`，滤波、接触幅值和迟滞沿用原配置，sigma 不参与这些计算。

这是用户明确要求的**只读研究观察部署**，`deployment_ready=false` 和 `calibration_state=uncalibrated` 保持不变。测试中学习 sigma 的总体 NLL 与 J1/J4 覆盖率未优于固定 sigma；部署不代表精度或 95% 覆盖率验收通过。结果见 [冻结均值方差试验](NEXT_FROZEN_SIGMA_20261023.md)。

## 当前运行状态

部署时旧窗口已经长时间收不到四路新鲜输入。新窗口确认状态为 `stale: input_timeout`，当前力矩和 sigma 显示“—”，不显示旧值或零值。域 74 中未发现上游控制器/机器人数据源，尚不能进行真机数值或当前物理控制 profile 的在线核验。没有为部署启动控制器、改变控制模式或发送运动。

服务状态与 PID：`log/deploy_frozen_sigma_20261023/service_state.json`。完整记录包括 `deployment.json`、`live_topology.json`、`verified_snapshot.json` 和 `service.log`。

四路输入、健康状态和适配器恢复后，服务自动等待 50 帧新鲜同步数据。仍须沿用原启动流程核对工具、空载、标定与控制增益是否匹配模型；模型加载器不会自动读取真机当前物理配置。训练范围检查保留，超范围时不把不确定度当作有效参考。恢复输入不代表模型精度已认证。

## 页面与话题

8082 已切换到独立命名空间 `/next/right/frozen_sigma`。原 `/next/right` 推理和 8081 页面保留，便于回退；两边均值来自不同训练快照，数值不保证相同。

页面上方保留 J1–J7 原始/滤波外力矩曲线；下方为同一帧的原始外力矩、学习 sigma、固定 sigma 对照与名义参考带半宽，单位均为 N·m。表格数据通过一个原子 JSON 包传输并校验时间戳、七关节顺序、有限数值和区间关系。

**±1.96σ 是以零为中心的自由运动残差名义参考带，不是外力真值的 95% 置信区间。**它不用于调零、降权或改变接触判定。数据过期、断连、状态非 valid 或超训练范围时当前 sigma 表格清空。

|话题后缀（位于 `/next/right/frozen_sigma`）|类型与内容|
|---|---|
|`/external_joint_torque/raw`|JointState.position：七关节原始外部力矩|
|`/external_joint_torque`|JointState.position：滤波外部力矩|
|`/free_joint_torque_pred`|JointState.position：自由运动力矩均值|
|`/free_joint_torque_sigma`|JointState.position：学习 sigma，N·m|
|`/free_joint_torque_sigma_fixed`|JointState.position：训练残差估计的固定 sigma，N·m|
|`/distribution`|String JSON：同帧均值、外力矩、两种 sigma、零中心参考带、状态、时间戳|
|`/status`|DiagnosticArray：数据有效性、范围、频率、延迟、失效原因|

推理节点为 `/next_frozen_sigma_right`，网页节点为 `/right_frozen_sigma_dashboard`。仅发布观察结果和 ROS 自身诊断话题，不创建运动控制客户端或发布控制命令。

## 重启

先停止当前观察进程，避免重复占用端口或重复发布。可在启动终端按 Ctrl+C；本次后台进程 PID 可从 `service_state.json` 读取并核对命令行为 `observe_w3_frozen_sigma.py`。停止这个观察进程不会停止原 `/next/right` 推理或机器人控制端。

```bash
cd /home/venom/factr2
ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 \
FASTRTPS_DEFAULT_PROFILES_FILE=/home/venom/factr2/config/w3/dds_loopback.xml \
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/observe_w3_frozen_sigma.py \
  --bundle models/w3/right/frozen_sigma_observer_20261023 \
  --state-file log/deploy_frozen_sigma_20261023/service_state.json
```

仅检查模型包与配置、不启动 ROS：

```bash
scripts/next.sh python scripts/observe_w3_frozen_sigma.py \
  --bundle models/w3/right/frozen_sigma_observer_20261023 --check-only
```

配置为模型包内 `inference.yaml`、`web.yaml`。固定 100 Hz、50 帧、CPU、0.25 s 超时与 40 ms 断流复位。W3 配置只对明确标记 `observation_mode: uncalibrated_research` 的右臂专用命名空间开放；普通生产加载器仍拒绝研究 artifact schema。

## 回退到旧 8082

停止新观察进程后，启动旧窗口即可；旧推理未停止：

```bash
cd /home/venom/factr2
ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 scripts/next.sh \
  python scripts/w3_endpoint_wrench.py \
  --config config/w3/right/endpoint_wrench_targeted_20261016.yaml
```

回退记录：`log/deploy_frozen_sigma_20261023/rollback.json`。没有覆盖原模型目录、原推理配置或控制参数。

## 实现与验证

- `scripts/export_w3_frozen_sigma.py`：导出自包含模型包、来源记录、物理训练范围及运行配置。
- `inference/frozen_sigma_observer.py`：专用严格加载器、一次 LSTM 前向同时得到均值和方差、原有新鲜度/范围/健康门禁。
- `visualization/frozen_sigma_web.py`：同帧 sigma 表格、明确未校准标签和失效清空。
- `scripts/observe_w3_frozen_sigma.py`：同一只读进程运行推理与网页。
- `scripts/test_w3_frozen_sigma_observer.py`：模型与方差绑定、哈希拒绝、在线单窗口对离线结果、原子包时间戳及隔离传输测试。

44 项普通回归通过；另在独立 localhost ROS 域 199 验证新观察器与旧关节窗口，共 2 项集成测试通过。集成测试覆盖同帧残差与 sigma、没有控制发布者、超范围隐藏与停止输入后清空。合成数据未进入真机域 74。实际页面、模型包身份与发布拓扑已核验；由于真实输入缺失，没有宣称真机力矩或 sigma 数值验证通过。
