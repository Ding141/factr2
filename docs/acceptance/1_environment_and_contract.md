# 第 1 部分验收

结论：**PASS**，范围 **PASS_OFFLINE**。日期：2026-10-05（Asia/Shanghai）。无真机实验、无任何电机命令。实际任务入口 docs/development/README.md（用户所述 docs/README.md 不存在）。

基线：FACTR2 `627ae22` / dingyj-dev，W3 `00740ac` / dingyj-dev；开始时两工作区干净。FACTR2 本次差异为环境脚本、固定依赖、统一契约、左右模板、模板运行检查及验收工具；W3 无源码差异。本报告所在提交即阶段 1 的交付 commit，避免自引用 SHA。

环境：Ubuntu 22.04，Humble，Python 3.10.12，项目 .venv --system-site-packages；NumPy 1.26.4、torch 2.5.1+cpu、h5py 3.11.0、scipy 1.13.1、setuptools 68.2.2。完整模块路径/版本见 1_environment.json；锁文件 next-py310.lock.txt 记录 venv 全部包，ROS apt 版本记录于 requirements/ros-humble-packages.txt。系统 Python 未安装/覆盖包。构建入口 shebang 为 /home/dingyj/factr2/.venv/bin/python，未构建其他硬件包。

所有下列命令在 /home/dingyj/factr2 执行，`scripts/next.sh` 是干净 env -i 子 shell。证据的绝对路径、SHA256、字节数见 [1_evidence.json](1_evidence.json)，实际日志保存在 Git 忽略的 reports/generated/。

| 验收条目 | 命令/实验（退出码） | 证据 | 结论 |
|---|---|---|---|
| ENV-01 | scripts/next.sh python scripts/environment_probe.py（0） | 1_environment.json，12 模块路径/版本，prefixes 仅 NEXT/Humble，Python 3.10 | PASS |
| ENV-02 | scripts/next.sh python -m colcon build --base-paths factr2_next/src/factr2_next --symlink-install（0）；ros2 pkg executables factr2_next；ros2 run factr2_next next_train --help（均 0） | 1_build.log；1_executables.txt 的四个入口；1_train_help.txt；log/latest_build/factr2_next/command.log 指定 venv 解释器 | PASS |
| ENV-03 | bash scripts/test_environment_dds.sh（0） | 1_dds.log、1/dds.json：系统 Python + W3 overlay subscriber 与 NEXT venv publisher 保真接收 JointState；stamp=[123,456789]，q=.125、qdot=-.25、tau=1.5 | PASS |
| ENV-04 | scripts/next.sh python scripts/check_w3_configs.py（0）；--runtime train --config config/w3/left/train.yaml（预期拒绝，1） | 1_configs.txt 八个模板；1_invalid_runtime.log 拒绝不存在的训练数据路径；节点/trainer 调用同一 validator | PASS |
| ENV-05 | 配置检查，读取 config/w3/contract.yaml（0） | docs/interfaces/w3_next_contract.md 与机器 YAML 版本/字段/单位/时间/连续性/metadata 一致 | PASS |
| ENV-06 | environment_probe 中 CPU LSTM [2,50,21] → [2,7]，有限输出（0） | 1_environment.json 的 cpu_forward；docs/environment.md；requirements/*.txt | PASS |

DDS：domain=73，rmw_fastrtps_cpp，sensor-data BEST_EFFORT/VOLATILE/KEEP_LAST depth=5；两个 shell source 后均恢复 ROS_LOCALHOST_ONLY=1。本机 mock 用 config/w3/dds_loopback.xml，UDP 仅 127.0.0.1，未证明跨机器 DDS。最初受沙箱 socket 禁止，随后授权本地 DDS。首次普通发现失败时两进程 ROS_LOCALHOST_ONLY 不一致（NEXT 继承宿主值 0）；最终两进程显式指定 1 并使用 loopback profile，重跑通过。pip 最初沙箱 DNS 不可用，联网后仅安装项目 venv（--no-cache-dir）。

失败项：最终验收无失败。`pip check` 对继承的系统 ipykernel/pynacl 报缺 debugpy/cffi，属于系统站点已有不完整依赖；与本阶段全部导入/CPU/DDS 实测无关，未为修复这些无关包修改系统。ROS 节点仅检查入口，不以 --help 声称已运行。

后续输入：第 2/3 部分使用冻结 w3_next_v1；第 4/5/6 使用已构建环境与左右模板，运行前需填写数据/模型路径。本阶段不完成 recorder 连续性门禁、模型元数据工具、七关节可视化或真实力矩精度验收。
