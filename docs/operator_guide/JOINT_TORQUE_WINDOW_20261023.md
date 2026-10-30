# 外力检测窗口：逐关节外部力矩

2026-10-23 后续部署：当前 8082 已切换到 [冻结均值＋学习 sigma 的只读研究窗口](FROZEN_SIGMA_DEPLOYMENT_20261023.md)，使用独立 `/next/right/frozen_sigma` 话题。下文原命令可恢复旧窗口；新窗口的重启和回退以新部署说明为准。当前真实输入缺失时力矩和 sigma 均显示“—”。

更新：2026-10-23。8081 和常用的 8082 外力检测窗口现在默认显示论文中的逐关节外部力矩，单位 N·m：

```text
tau_ext = tau_measured - tau_free_pred
```

默认启用 J1–J7 的原始残差曲线；另有逐关节滤波曲线可勾选对照。J1 对应 `right_joint_0`，J7 对应 `right_joint_6`，保留正负号。窗口不进行雅可比换算，也不订阅机器人描述或关节位置；末端奇异性不会阻断此显示。数据失效或过期时状态明确提示，当前数值显示“—”，已记录的历史曲线保留。

## 已有推理运行时

在原 8082 观测程序的终端按 Ctrl+C，然后使用原命令重启观测程序。只重启网页服务即可加载本次 Python 修改；已有推理/适配器继续运行。以下配置对应 10 月 16 日待验收模型：

```bash
cd /home/venom/factr2
export ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/next.sh python scripts/w3_endpoint_wrench.py \
  --config config/w3/right/endpoint_wrench_targeted_20261016.yaml
```

打开 <http://127.0.0.1:8082/>。启动器沿用配置中的端口和 `inference_config` 指定的输出话题，并保留模型待验收说明；它只订阅结果，不重新加载或启动模型。已有 8081 服务也需在其所在终端重启后才能显示新版布局；8082 可独立使用。

## 输出话题

|话题|显示内容|
|---|---|
|`/next/right/external_joint_torque/raw`|默认显示的原始外部关节力矩|
|`/next/right/external_joint_torque`|滤波外部关节力矩，可勾选|
|`/next/right/status`|推理状态与新鲜度|

现有接口使用 `sensor_msgs/JointState.position` 承载这七个力矩值，单位为 N·m。状态有效表示数据链路正常，不代表模型已经通过物理精度验收；空载残差不会在网页中自动扣除。

## 可选末端六维诊断

需要原先的末端力/力矩换算时，显式添加 `--view endpoint_wrench`：

```bash
scripts/next.sh python scripts/w3_endpoint_wrench.py \
  --config config/w3/right/endpoint_wrench_targeted_20261016.yaml \
  --view endpoint_wrench
```

两种视图沿用同一配置端口，不能同时占用 8082。显式指定原有 `--mode full_wrench` 或 `--mode force_only` 时也继续采用末端诊断；关节窗口不使用这些求解器参数。
