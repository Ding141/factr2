# NEXT 运动残差补偿与平滑窗口

2026-10-23，右臂，ROS domain 74、100 Hz。新增只读观测分支，不发送机械臂运动或电流/力矩控制命令。

- 新窗口：[8083：补偿与平滑输出](http://127.0.0.1:8083/)。默认显示最终平滑输出，可勾选原 NEXT、补偿后输出滤波前两组对照。
- 原窗口：[8082：冻结均值与未校准 sigma](http://127.0.0.1:8082/)。保持运行，新分支依赖此基础推理服务。
- 实验：`/home/venom/factr2/data/right/kinematic_compensation_20261023_v2`；留出报告 `REPORT_ZH.md`，模型与适用范围 `model.json`。
- 状态：`/home/venom/factr2/log/deploy_kinematic_compensation_20261023/service_state.json`。

## 本次适用范围

验证选择局部速度＋加速度模型，启用 J1/J3/J4/J6/J7 的剩余误差补偿，J2/J5 仅滤波。主要激励来自 J1 往复，其他关节的改善主要是对 J1 引起的耦合残差，不能当作它们独立运动已验证。当前姿态尤其 J3≈−2.77°；超范围补偿量回退到零，并保留未补偿残差，状态原因含 `compensation_bypassed_out_of_support`。

原 LSTM 输入不变；新增支路使用位置 KF 的因果速度/加速度。输入力矩与自由预测相同 12 Hz 低通；最终 5 Hz 低通。合成阶跃 90% 响应约 80 ms，短脉冲会衰减。没有验证真实接触，不用于安全保护或力控，旧 sigma 不适用于新残差。

## 重启或回退

先确认基础 8082 服务与健康器、real adapter 已运行，再在没有同名补偿服务时启动：

```bash
cd /home/venom/factr2
ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 \
FASTRTPS_DEFAULT_PROFILES_FILE=/home/venom/factr2/config/w3/dds_loopback.xml \
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/observe_w3_kinematic_compensation.py \
  --experiment data/right/kinematic_compensation_20261023_v2 \
  --port 8083 \
  --state-file log/deploy_kinematic_compensation_20261023/service_state.json
```

只检查文件和模型绑定可加 `--check-only`，不启动 ROS。Ctrl+C 结束这个只读进程；回看 8082 即为原始模型，不需要改机械臂控制器。

## 再训练和报告

```bash
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/fit_w3_kinematic_compensation.py --output data/right/NEW_EXPERIMENT
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/report_w3_kinematic_compensation.py --experiment data/right/NEW_EXPERIMENT
```

输出目录必须不存在，不覆写旧模型。模型选择只用验证段；测试记录以前已查看，因此是回顾性比较。信号参数在训练时固定，不能只改实时滤波而继续使用旧补偿参数。

可观测话题前缀：`/next/right/kinematic_compensated`。`external_joint_torque` 为最终输出，`external_joint_torque/raw` 为补偿后输出滤波前，`upstream_external_joint_torque/raw` 为原 NEXT 未处理残差。`correction` 为学习补偿量，`distribution` 为同帧 JSON；`input/` 提供滤波位置、电机速度、估计速度/加速度和输入力矩。运动学单位分别为 rad、rad/s、rad/s²，力矩为 N·m。

参见 [文献依据与设计边界](../research/KINEMATIC_RESIDUAL_FILTER_REVIEW_20261023.md)。
