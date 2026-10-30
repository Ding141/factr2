# 2026-10-25 新采空载数据与速度/加速度补偿

右臂实机采集已经完成：35/35 段，两档规划速度 1.2/2.4°/s，工作中心
`[1,55,-4,-58,1,-2,0]°`，J1–J4 最大 ±3°、J5–J7 最大 ±2°。
规划运动总时长 601.99 秒；最终进位段因保守导数边界延长到 234.14 秒。
原始 H5 62,705 帧，100 Hz，最大源间隔 16.67 ms，质量检查通过。
同帧观察记录 64,189 帧（两种记录器的起止时刻不同），无大于40ms间隔。
停止后8秒检查显示姿态稳定，控制器状态符合采集前约定。没有修改控制器增益。

## 当前查看入口

- **8083：新补偿后的最终输出**，http://127.0.0.1:8083/ 。窗口保留两位小数。
- **8082：原 NEXT 残差与 sigma 对照**，http://127.0.0.1:8082/ 。它不包含此次附加补偿。
- 模型实验：`/home/venom/factr2/data/right/kinematic_compensation_20261025_v1`。
- 采集、授权、冻结轨迹、碰撞检查与诊断：`/home/venom/factr2/log/kinematic_collection_20261025_v1`。
- 原始 H5、bag、CAN 与参数快照：`/home/venom/factr2/data/quick_capture/right/20261025_152004_776010`。
- 补偿服务状态：`log/kinematic_collection_20261025_v1/deployed_service_state.json`。
- 模型 SHA256：`89af2ae40f122e512287dac6fe0798d388701ad10c43cdfcaaa50a5888fc459e`。

## 结果与适用范围

训练、验证、测试角色在运动前按完整块固定，段首尾各排除1秒。
训练输入使用原始精度，未使用窗口四舍五入值。保留历史数据的记录级划分。
模型、特征模式、范围、正则化和启用关节只依据训练/验证，测试不参与选择。
选中局部速度＋加速度模型，ridge=0.1，七个关节均通过验证启用。

新增测试运动段6,172帧，补偿支持率100%；下表为回放 RMS，单位N·m。

|关节|原 NEXT|仅滤波|新补偿与滤波|相对原 NEXT 改善|
|---|---:|---:|---:|---:|
|J1|0.23911|0.23746|0.17361|27.4%|
|J2|0.23251|0.23099|0.08800|62.2%|
|J3|0.25289|0.24970|0.12987|48.6%|
|J4|0.25389|0.25180|0.16127|36.5%|
|J5|0.05156|0.05047|0.02423|53.0%|
|J6|0.14699|0.14611|0.10776|26.7%|
|J7|0.04431|0.04390|0.02563|42.2%|

加速度相对仅速度的额外收益有限，测试中J2/J5/J6略差于仅速度。
低速J1仅改善4.2%；静止测试J1/J4 RMS仍约0.09/0.11N·m，J6静止RMS略差2.1%。
这些结果没有被整体平均或自动归零掩盖。该数据不能辨识独立的真实惯量、
摩擦或电流偏置，也不能证明其他姿态、温度和真实接触的精度。

详细分档、偏置、峰值和相关性见实验内`heldout_detail.json`与`REPORT_ZH.md`。
原模型保持冻结；实际/指令位置因果KF提供速度和加速度，配对力矩输入12Hz，
最终输出5Hz。加速度特征尺度至少0.005rad/s²，并用训练静止段三倍MAD噪声
尺度提高门槛；未充分激励的特征权重固定为零。
没有力矩残差作为输入、截距或在线学习；超范围时回退补偿，仍保留外力残差。

64,189帧逐帧实时实现/离线回放最大差异2.22e−16N·m。
固定运动学的0.3N·m合成阶跃稳态增益为1，90%响应约80ms。
实机新服务另核查600个同帧数据，恒等式误差4.44e−16N·m，各输入发布源唯一。
21项离线测试通过。未验证真实接触；该结果用于研究观察，不用于力控或保护。

## 重启补偿窗口

前置只读数据链必须存在：ROS domain74，健康器同时监测双臂，右臂real
adapter为100Hz，原始冻结均值服务在8082。不要重复启动同一话题的发布器。
缺少adapter时8082会提示`input_timeout`；当前姿态超出原模型训练范围时会提示
`out_of_scope`，不能通过关闭健康/范围检查来替代数据恢复。

在没有同名补偿进程时启动：

```bash
cd /home/venom/factr2
ROS_DOMAIN_ID=74 ROS_LOCALHOST_ONLY=1 \
FASTRTPS_DEFAULT_PROFILES_FILE=/home/venom/factr2/config/w3/dds_loopback.xml \
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/observe_w3_kinematic_compensation.py \
  --experiment data/right/kinematic_compensation_20261025_v1 \
  --port 8083 \
  --state-file log/kinematic_collection_20261025_v1/deployed_service_state.json
```

加`--check-only`仅检查模型与文件绑定。回退旧模型时停止这个只读进程，
将experiment改为`data/right/kinematic_compensation_20261023_v2`；也可查看8082原始模型。
新旧服务都没有机械臂执行器接口。

## 复现拟合

输出目录必须是新目录，不覆盖已部署实验。

```bash
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/fit_w3_kinematic_compensation.py \
  --observation log/kinematic_collection_20261025_v1 \
  --split-manifest log/kinematic_collection_20261025_v1/test_spec.json \
  --acceleration-floor 0.005 \
  --output data/right/NEW_EXPERIMENT
scripts/next.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/report_w3_kinematic_collection.py --experiment data/right/NEW_EXPERIMENT
```

不要重新执行日志内的`run.py`来重做轨迹：已有执行状态会拒绝重试；任何新采集
需要新目录、当前姿态检查、重新冻结轨迹与碰撞报告，以及相应现场确认。
