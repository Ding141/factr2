# NEXT / W3 隔离环境

CPU 基线：Ubuntu 22.04、Humble、/usr/bin/python3.10，NEXT `.venv --system-site-packages` 读取 apt ROS 绑定；科学依赖仅装入 `.venv`。不 source W3 overlay，不写 bashrc，不安装 Piper/Gello SDK，不启硬件。固定版本见 requirements/next-py310.txt 和 next-py310.lock.txt（仅 venv 安装包）；ROS/colcon apt 版本见 ros-humble-packages.txt。

```bash
cd /home/dingyj/factr2
bash scripts/setup_next_env.sh
scripts/next.sh python -m colcon build --base-paths factr2_next/src/factr2_next --symlink-install
scripts/next.sh python scripts/environment_probe.py
scripts/next.sh ros2 pkg executables factr2_next
scripts/next.sh ros2 run factr2_next next_train --help
scripts/next.sh python scripts/check_w3_configs.py
bash scripts/test_environment_dds.sh
```

DDS mock 验收使用 config/w3/dds_loopback.xml（限定 127.0.0.1 UDP），避免共享内存/默认发现受本机环境影响；生产跨机器不使用此 profile。脚本保留调用者 ROS_LOCALHOST_ONLY；测试在双方显式指定 1，避免继承宿主设置造成不一致。

`next.sh` 使用 env -i 的独立子 shell：Humble → venv → FACTR2 local_setup。默认 domain=73、localhost-only=1、rmw_fastrtps_cpp，可通过同名变量覆写；DDS peer 必须相同 domain、兼容 RMW，跨机器时双方 ROS_LOCALHOST_ONLY=0。`next_env.sh` 可 source，但拒绝外来 overlay/venv。ROS 日志写项目 log/ros，pip 不使用缓存。

W3 独立 shell 不激活 NEXT venv：

```bash
cd /home/dingyj/w3_dual_arm_ws
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
  ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ROS_LOG_DIR="$PWD/log/ros" bash --noprofile --norc
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export ROS_LOCALHOST_ONLY=1
```

W3 构建脚本本身清理 overlay，构建产物在 W3 工作区 build/install/log；不启动 dual_arm.launch.py 做离线验收。FACTR2 第 3 部分后构建 base-paths 增加 factr2_w3_adapter。健康脚本使用 W3 shell 系统解释器直接运行 FACTR2 源文件。

左右普通 NEXT YAML 在 config/w3/{side}，每个都显式 side、契约版本、joint_names。先填写 SESSION_ID、TRAIN_SESSION、VAL_SESSION、DATASET_ID、TIMESTAMP 路径，再用 `scripts/next.sh python scripts/check_w3_configs.py --runtime train --config config/w3/left/train.yaml` 校验。节点经 `--ros-args -p config_file:=/绝对路径/config.yaml`，trainer 经 `--config`；不能把普通 YAML 传 --params-file。模板缺路径会拒绝运行，不自动选择 Piper。

数据/venv/权重/build/install/log/reports/generated 被 Git 忽略。接受报告的 manifest 保存证据绝对路径与 SHA256。第 4/5/6 部分负责 recorder 的连续性门禁、模型元数据和七关节可视化，本阶段不声称已完成。
