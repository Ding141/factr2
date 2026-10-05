# 第 6 部分验收：只读推理与七关节页面

2026-10-05，FACTR2 `dingyj-dev`，前置提交 `c7339c4`；W3 `d32262d`，本阶段无改动。使用第 5 部分左右可重载 synthetic smoke checkpoint，无硬件、回环 DDS。**结论：PASS，范围：PASS_OFFLINE**。

| 验收条目 | 命令/实验 | 证据 | 结论 |
|---|---|---|---|
| INF-01 | `scripts/next.sh python scripts/test_inference_replay.py`：实际 H5 player → 独立 InferenceNode → DDS subscriber | [6_summary.json](6_summary.json) replay：600 行、551 次预测，末帧 stamp=input[49:]，free 最大误差 0 Nm、raw allclose(1e-5) | PASS |
| INF-02 | 同一 replay 实验的正常和每次恢复预热边界 | [6_summary.json](6_summary.json) replay：前 49 帧无输出、第 50 帧首次输出，N-H+1 正确 | PASS |
| INF-03 | 同一 replay 实验全部消息停发 .30 s | [6_summary.json](6_summary.json) replay：从最后输入发布计 stale 0.26685 s <=0.27 s；history 清空、无旧 torque、恢复重新 50 帧 | PASS |
| INF-04 | 同一 replay 实验注入坏 name/8维/NaN、回退、旧排队帧、gap、adapter ERROR 和有限大输入 | [6_summary.json](6_summary.json) replay：全部 invalid/reset，派生 Float32 overflow 拒收 | PASS |
| INF-05 | `scripts/next.sh python -m pytest factr2_next/src/factr2_next/test/test_inference_math.py -q`：滤波和迟滞手算实验 | [6_evidence.json](6_evidence.json) tests.log：none/EMA/lowpass/reset、独立 raw、high 开启/low 关闭/中间保持 | PASS |
| INF-06 | `scripts/next.sh python scripts/test_inference_config.py`；web dual 实验传错臂 checkpoint；同一数学测试检查诊断门禁 | [6_summary.json](6_summary.json) config_rejection/web：六类非法配置退出 1，错臂失败且无 NEXT 输出，门禁不可关闭 | PASS |
| INF-07/08 | `scripts/next.sh python scripts/test_inference_stream.py --seconds 600 --domain 87`；最终代码另以 `--seconds 60 --domain 87 --label final60` 执行 | [6_summary.json](6_summary.json) stability_600s/final_60s：45 Hz/20 ms/60 ms 门槛全通过；最终两侧各 3000 帧约 50 Hz，RSS 见下表 | PASS |
| INF-09 | `scripts/next.sh python scripts/test_web_dual.py` 七关节脉冲；实际浏览器 DOM 操作和 HTTP 停止实验 | [6_summary.json](6_summary.json) web/browser：7.123 Nm 仅进入 ext_j7，0..6 控件、valid/年龄、ext_j7/raw_j7 切换、disconnected 均符合预期 | PASS |
| INF-10 | 同一 web dual 实验：实际双侧 launch、SIGINT 左 inference、完整退出、端口重绑定 | [6_summary.json](6_summary.json) web：左 stale、右 valid、W3 mock 持续运行，18100/18101 释放、子进程退出 0，clients=[] | PASS |

| CPU 单线程测试 | 左 | 右 |
|---|---:|---:|
| 600 s 有效输出 | 29898 | 29950 |
| 600 s 有效 Hz | 49.830 | 49.917 |
| 600 s 推理 P95 ms | 6.842 | 6.631 |
| 600 s 源到接收 P95 ms | 16.365 | 16.321 |
| 600 s 峰值 RSS MiB | 255.281 | 255.500 |
| 600 s RSS 增长 MiB | 0.156 | 0.219 |
| 600 s 后半程 RSS 斜率 MiB/min | 0.000283 | 约 0 |
| 最终 60 s 推理 P95 ms | 6.866 | 7.108 |
| 最终 60 s 源到接收 P95 ms | 17.033 | 17.319 |

推理耗时使用单调时间，包括窗口预处理、forward 和反归一化；延迟是同主机输出接收 ROS time 减窗口末帧 source stamp。W3 标准 mock 300 Hz，经实际 adapter 50 Hz，不降采样/历史通过验收。稳定性探针观测自身保留消息，不计入真实 inference PID 的 `/proc` RSS。性能测试不运行 web，但双侧页面在独立 domain 另做联测；主机压力可造成 gap，节点会按契约重新预热，不把断点历史拼起来。

双侧页面最初没有收到 NEXT 消息，定位到回环配置 initial peer 默认范围不足；扩到 32 后多进程通信与页面恢复，并仅绑定 127.0.0.1。参数依据 [Fast DDS 2.6 官方 transport 文档](https://fast-dds.docs.eprosima.com/en/2.6.x/fastdds/xml_configuration/transports.html)。SSE 合并为 10 Hz，避免每个 50 Hz torque/status 更新重复推送完整曲线。端口占用给 `web_bind_failed` 并保留原服务可用。浏览器两种截图接口均返回不可用，因此截图未作为验收证据；保留真实 DOM 操作记录和 HTTP 快照。

最终功能回归 **51 项通过**（录制/数据质量、训练/重载/评估、推理数学与配置）；CLI 六种启动拒绝、实际 ROS 回放/故障、双侧 launch、600 s 稳定性和最终 60 s 均通过。严格 H5 检查还补充 NaN 时间/坏 sidecar 的机器可读失败，以及 int64 纳秒精确的 40 ms 边界。`colcon build` 两包成功，左右模板校验和 `git diff --check` 通过。

运行期图信息显示 inference 仅发布 `/next/{side}`、rosout/parameter_events，clients=[]；launch 源码仅 NEXT 两种节点。测试结束后的只读进程检查没有遗留验收进程，HTTP 端口释放，临时浏览器标签关闭。使用项目 .venv、ROS_LOG_DIR、Matplotlib cache 和 TMPDIR，未系统级安装、修改 shell 全局配置或修改 W3 仓库。

部署/故障语义和复现命令见 [推理说明](../inference.md)。逐项摘要和真实进程资源采样见 [6_summary.json](6_summary.json)，全部原始证据路径/大小/SHA256 见 [6_evidence.json](6_evidence.json)。数据、模型、日志留在项目忽略目录。smoke 不代表真机外力估计质量；真实数据/模型/跨机部署由第 7/8 部分验收。
