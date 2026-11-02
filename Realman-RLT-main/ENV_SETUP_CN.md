# RealMan-RLT 环境配置

环境由三部分组成：自动安装的 Python 环境、RealMan RM_API2 和 XRoboToolkit。

## 目录约定

在仓库根目录执行：

```bash
export REPO_ROOT="$PWD"
export THIRD_PARTY_DIR="$REPO_ROOT/third_party"
```

所有厂商文件统一放入 `third_party/`，不会提交到 Git：

```text
third_party/
├── RM_API2/
└── XRoboToolkit/
    ├── downloads/
    └── XRoboToolkit-PC-Service-Pybind/
```

## 1. 自动安装 Python 环境

前提：Ubuntu 22.04、Conda、Python 3.10；GPU 训练还需要 NVIDIA 驱动。

```bash
bash scripts/tools/setup_env.sh
```

脚本会创建或复用 `lerobot`，安装 PyTorch 和项目 Python 依赖，并生成 `.env.local`。

终端会显示真实的 pip 下载信息：当前 wheel 名称、已下载大小、速率和进度条。若包已缓存，pip 会显示 `Using cached`，表示没有发生下载。`[setup n/m]` 仅表示脚本阶段，不表示下载百分比。

GPU 使用 CUDA 11.8：

```bash
bash scripts/tools/setup_env.sh --cuda cu118
```

仅 CPU：

```bash
bash scripts/tools/setup_env.sh --cuda cpu
```

完成后：

```bash
conda activate lerobot
source .env.local
python scripts/tools/check_env.py --skip-hardware
```

## 2. 安装 RealMan RM_API2

真机采集和部署需要 RM_API2；纯训练和数据转换可以跳过。

1. 从 [RealManRobot/RM_API2](https://github.com/RealManRobot/RM_API2) 下载或克隆官方 SDK 到 `third_party/RM_API2/`。
2. 配置 SDK 路径并检查：

```bash
export RM_API2_PYTHON_PATH="$THIRD_PARTY_DIR/RM_API2/Python"
export RM_API_PATH="$RM_API2_PYTHON_PATH"
python scripts/tools/check_env.py --rm-api-path "$RM_API2_PYTHON_PATH"
```

首次运行自动安装时，也可以记录该路径：

```bash
bash scripts/tools/setup_env.sh --rm-api-path "$PWD/third_party/RM_API2/Python"
```

将两个 `RM_API` 变量添加到 `.env.local` 的脚本管理区块之外；再次运行 setup 不会覆盖它们。

## 3. 安装 XRoboToolkit-PC-Service

Quest 遥操作需要 Quest 3 兼容的 XR App、PC Service、Python binding `xrobotoolkit_sdk`。这些需要用户手动下载和安装，不属于 `setup_env.sh`。

### 3.1 下载与安装 PC Service

1. 打开 [XRoboToolkit-PC-Service Releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)，下载与系统架构匹配的 PC Service `.deb`。
2. 将下载的文件放入 `third_party/XRoboToolkit/downloads/`。
3. 显式指定下载的 `.deb` 并安装；不要通过搜索目录或“取第一个文件”自动选择安装包：

```bash
export XROBO_DEB="$THIRD_PARTY_DIR/XRoboToolkit/downloads/<PC-Service-deb>"
test -f "$XROBO_DEB" || { echo "未找到指定的 PC Service 安装包"; exit 1; }
sudo apt install "$XROBO_DEB"
```

启动并确认服务：

```bash
/opt/apps/roboticsservice/run3D.sh
pgrep -a RoboticsServiceProcess
```

无图形界面时使用：

```bash
/opt/apps/roboticsservice/runService.sh
```

### 3.2 安装与使用 ADB

ADB 只用于通过 USB 安装、启动和诊断 Quest App；日常遥操作仍通过 Quest 与 PC 的局域网完成。在 Ubuntu 安装：

```bash
sudo apt update
sudo apt install -y adb
adb version
```

在 Quest 中开启开发者模式和 USB 调试，通过 USB 连接 PC，然后在头显中确认“允许 USB 调试”。下列命令应显示一个状态为 `device` 的序列号；`unauthorized` 表示还未在头显中授权。

```bash
adb devices
```

上游 XRoboToolkit 文档主要面向 Pico。只有在 APK 或供应方明确标注支持 **Meta Quest 3** 时，才可安装到 Quest。安装 APK 后可以核对包名并启动 App：

```bash
export XROBO_APK="$THIRD_PARTY_DIR/XRoboToolkit/downloads/<Quest-App-apk>"
adb install -r "$XROBO_APK"
adb shell pm list packages | grep xrobotoolkit
adb shell monkey -p com.xrobotoolkit.client.quest 1
```

`monkey` 命令仅适用于包名确为 `com.xrobotoolkit.client.quest` 的 App；如果查询结果为其他包名，应使用实际包名。

### 3.3 配置 Quest 无线 ADB（可选但推荐）

无线 ADB 只用于**无线安装、启动和诊断 Quest App**，不是 Quest 遥操作数据链路本身。遥操作数据仍然是：

```text
Quest App ──局域网──> PC Service ──Python binding──> 本项目脚本
```

也就是说，即使不用无线 ADB，只要 Quest App 已经启动并连上 PC Service，遥操作也可以工作；无线 ADB 的价值是不用每次插 USB 线启动 App、看日志或重装 APK。

前提：

- PC 和 Quest 连接到同一个可互访的 Wi-Fi / 局域网。
- Quest 已开启开发者模式。
- 已通过 USB 授权过一次 ADB 调试；`adb devices` 显示 `device`，不是 `unauthorized`。

#### 3.3.1 首次从 USB 切到无线 ADB

先用 USB 连接 Quest 和 PC，在头显里允许 USB 调试，然后执行：

```bash
adb devices
adb tcpip 5555
```

读取 Quest 在 Wi-Fi 上的 IP：

```bash
adb shell ip -f inet addr show wlan0 | sed -n 's/.*inet \([0-9.]*\).*/\1/p'
```

假设输出为 `192.168.13.120`，拔掉 USB 线后连接：

```bash
export QUEST_IP=192.168.13.120
adb connect "$QUEST_IP:5555"
adb devices
```

成功时应看到类似：

```text
192.168.13.120:5555    device
```

此后可以无线启动 Quest App：

```bash
adb shell pm list packages | grep xrobotoolkit
adb shell monkey -p com.xrobotoolkit.client.quest 1
```

如果包名不是 `com.xrobotoolkit.client.quest`，用上一条命令查到的实际包名替换。

#### 3.3.2 无线 ADB 常用诊断命令

查看设备：

```bash
adb devices
```

断开无线连接：

```bash
adb disconnect "$QUEST_IP:5555"
```

重新连接：

```bash
adb connect "$QUEST_IP:5555"
```

查看 Quest App 日志，便于判断 App 是否连上 PC Service：

```bash
adb logcat | grep -i -E 'xrobotoolkit|robotics|service|socket|connect'
```

如果同时连了 USB 和无线设备，给命令显式指定设备：

```bash
adb -s "$QUEST_IP:5555" shell monkey -p com.xrobotoolkit.client.quest 1
```

#### 3.3.3 常见问题

- `adb connect` 超时：确认 PC 和 Quest 在同一局域网，且路由器没有开启 AP 隔离 / 客户端隔离。
- `adb devices` 显示 `offline`：执行 `adb disconnect "$QUEST_IP:5555"` 后重连；仍失败就重新插 USB，再执行 `adb tcpip 5555`。
- 找不到 Quest IP：可以在 Quest 系统 Wi-Fi 详情页查看 IP；或 USB 连接后再次运行 `adb shell ip -f inet addr show wlan0`。
- 每次重启 Quest 后无线 ADB 失效：这是正常现象，重新 USB 授权并执行 `adb tcpip 5555` 即可。
- Quest App 仍连不上 PC Service：无线 ADB 只说明 PC 能调试 Quest，不代表 XRoboToolkit App 已连上 PC Service。继续检查 Quest App 中填写的 PC IP、PC Service 进程和 4.2 节的端口监听。

### 3.4 安装 Python binding

Python binding 不包含在 PC Service `.deb` 中。克隆官方 [XRoboToolkit-PC-Service-Pybind](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind) 到 `third_party/XRoboToolkit/`。推荐复用已安装 PC Service 中的同版本 SDK，再构建 binding：

```bash
conda activate lerobot
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git \
  "$THIRD_PARTY_DIR/XRoboToolkit/XRoboToolkit-PC-Service-Pybind"
cd "$THIRD_PARTY_DIR/XRoboToolkit/XRoboToolkit-PC-Service-Pybind"
cp /opt/apps/roboticsservice/SDK/include/PXREARobotSDK.h include/
cp /opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so lib/
conda install -c conda-forge pybind11 -y
python setup.py install
```

需要 `git`、`cmake`、C++ 编译器和网络。在 `.env.local` 的脚本管理区块之外设置：

```bash
export XROBO_PYBIND_ROOT="$THIRD_PARTY_DIR/XRoboToolkit/XRoboToolkit-PC-Service-Pybind"
export XROBO_PC_SERVICE_ROOT="/opt/apps/roboticsservice"
export LD_LIBRARY_PATH="$XROBO_PYBIND_ROOT/lib:$XROBO_PC_SERVICE_ROOT/SDK/x64:${LD_LIBRARY_PATH:-}"
```

验证 Python binding：

```bash
python -c 'import xrobotoolkit_sdk; print("xrobotoolkit_sdk import OK")'
```

启动 PC Service 后，运行本机连通性测试：

```bash
/opt/apps/roboticsservice/runService.sh
python - <<'PY'
import xrobotoolkit_sdk as xrt
xrt.init()
try:
    print("right pose:", xrt.get_right_controller_pose())
finally:
    xrt.close()
PY
```

未连接头显时，位姿为全零是预期现象。PC Service 正在运行、Quest App 已连接且 PC 与 Quest 位于同一局域网时，才可开始 Quest 数据采集。

## 4. Quest3 → RealMan 最小遥操作测试

此测试只验证 Quest 3、PC Service、Python binding、RM65 和夹爪的完整控制链路；不会启动相机、录制 HDF5 或保存数据。右手柄的增量控制和 Grip “离合”方式参考 [Quest3-Teleoperation](https://github.com/is-aHuan/Quest3-Teleoperation)，并核对了 [lerobot-realman-vla](https://github.com/Humble2Full/lerobot-realman-vla) 的 Quest 真机操作流程；此处直接使用 RM_API2，不启动参考项目的 ROS 2、Pink、相机和录制节点。

### 4.1 连接链路

```text
Quest App ──局域网──> PC Service ──本地 SDK──> test_quest_realman.py ──TCP──> RM65 控制器 ──> 夹爪
```

这里有两条独立的网络链路：

- Quest 与 PC：两者需要在可互相访问的局域网中，不能启用 AP/客户端隔离。
- PC 与 RM65：PC 需要能访问机械臂控制器 IP；默认端口为 `8080`。

PC 可以使用 Wi-Fi 连 Quest、另一个有线网口连 RM65，不要强行把 Quest 和 RM65 放在同一子网。`xrobotoolkit_sdk` 连接的是 PC 上的 PC Service，不直接连头显或机械臂。

当前工位的连接参数：

| 链路 | 地址 | 用途 |
| --- | --- | --- |
| Quest App → PC Service | `192.168.13.81` | 填写到 Quest App 的 PC Service IP 字段 |
| PC → RM65 控制器 | `192.168.1.18:8080` | 传给 `--arm-ip` 和默认 `--arm-port` |

### 4.2 连接 Quest 3 与 PC Service

1. 将 Quest 3 和 PC 连入同一个可互访的局域网。当前工位应在 Quest App 中填写 `192.168.13.81`。如果日后更换网络，可用下列命令查看 PC 网口地址：

```bash
ip -br address
```

2. 在 Quest 3 中启动明确支持 Meta Quest 3 的 XRoboToolkit App，保持头显唤醒、右手柄已配对且正在跟踪。首次启动时授予 App 要求的跟踪和网络权限。
3. 按 Quest App 当前版本的配对界面选择 PC；如果 App 要求输入 PC 地址，填写上一步中与 Quest 同网的 PC 地址，不要填 `127.0.0.1`。不同 Quest App 版本的界面不同，以其自带的连接提示为准。
如果 Quest 已通过 USB 或无线 ADB 授权连接 PC，可用 3.2/3.3 节的 ADB 命令确认并启动已安装的 App：

```bash
adb devices
adb shell pm list packages | grep xrobotoolkit
adb shell monkey -p com.xrobotoolkit.client.quest 1
```

`monkey` 命令仅适用于包名确为 `com.xrobotoolkit.client.quest` 的 App；如果上一条命令显示其他包名，应使用实际包名。也可直接在头显中启动 App，ADB 不是日常遥操作的必要条件。

4. 在 PC 端二选一启动 PC Service：

```bash
# 有图形界面
/opt/apps/roboticsservice/run3D.sh

# 无图形界面
/opt/apps/roboticsservice/runService.sh
```

不要重复启动多个服务进程。确认进程和 Python SDK 端口：

```bash
pgrep -a RoboticsServiceProcess
ss -ltnp | grep -E '60061|63901'
```

当前 PC Service 安装中，`127.0.0.1:60061` 监听表示 Python SDK 可以访问 PC Service；部分版本或连接状态还会出现 `63901`。不要把 `127.0.0.1`、`60061` 或 `63901` 当作 Quest App 的 PC 地址；App 中应填写 PC 在 Quest 局域网上的实际 IP。

### 4.3 先验证 Quest 数据，不连机械臂

先只读取右手柄，确认头显到 Python 的链路完整：

```bash
conda activate lerobot
source .env.local
python - <<'PY'
import time
import numpy as np
import xrobotoolkit_sdk as xrt

xrt.init()
try:
    print("10 秒内移动右手柄，并分别按下 Grip 和 Trigger。")
    for _ in range(50):
        pose = np.asarray(xrt.get_right_controller_pose(), dtype=float)
        grip = float(xrt.get_right_grip())
        trigger = float(xrt.get_right_trigger())
        print(f"pose={np.round(pose[:3], 3)} grip={grip:.2f} trigger={trigger:.2f}")
        time.sleep(0.2)
finally:
    xrt.close()
PY
```

正常现象：

- `pose` 不是全零，移动右手柄时数值持续变化。
- 按下右 Grip 时 `grip` 明显增大，按下右 Trigger 时 `trigger` 明显增大。

`xrt.init()` 成功只说明 Python binding 已连到 PC Service。如果位姿一直为全零，表示 Quest 数据流尚未到达 PC；此时不要进入真机测试。

### 4.4 连接 RM65

1. 接通机械臂控制器与 PC 的网络，在示教器或机械臂网络配置中确认 RM65 IP。
2. 设置本次测试的 IP（当前工位为 `192.168.1.18`），并检查 PC 的路由：

```bash
export ARM_IP=192.168.1.18
ip route get "$ARM_IP"
ping -c 3 "$ARM_IP"
```

`ping` 被控制器禁用时可能失败，因此它只是初步检查；最终以 RM_API2 能否创建机械臂连接为准。确保没有另一个程序同时向该机械臂发送运动命令。

### 4.5 启动前安全确认

测试会真实移动机械臂。运行前必须同时满足：

- 急停按钮可触及且已确认可用，操作者始终留在旁边。
- 机械臂、夹爪、工装和电缆周围无人员、障碍物或可夹持物。
- 采集零位 `[0, 0, 90, 0, 90, 30]` 在当前工位可安全到达。
- 基座坐标系下的工作空间 `x=[-0.5, 0.1] m`、`y=[-0.3, 0.3] m`、`z=[0.1, 0.6] m` 适合当前工位。
- Quest 位姿、Grip 和 Trigger 已经通过 4.3 节的无机械臂测试。

脚本不会自动抬升机械臂或先打开夹爪，第一个真机动作是操作者输入 `HOME` 后低速回到上述零位。

### 4.6 运行测试

首次或需要重新检查环境时，通过 setup 运行：

```bash
bash scripts/tools/setup_env.sh \
  --rm-api-path "$PWD/third_party/RM_API2/Python" \
  --quest-smoke-test \
  --arm-ip "$ARM_IP" \
  --arm-port 8080
```

环境已安装并通过自检后，可直接重复运行：

```bash
conda activate lerobot
source .env.local
python scripts/tools/test_quest_realman.py \
  --arm-ip "$ARM_IP" \
  --arm-port 8080
```

必须在可交互终端中运行，不要放到后台，也不要将标准输入重定向。`--arm-port` 默认已是 `8080`；只有在 RM65 控制器明确使用其他端口时才修改。

### 4.7 完整操作顺序

1. 终端显示 `RealMan 已连接` 后，再次检查工作空间和急停。`Quest SDK 已连接到 PC Service` 仅表示 SDK 本地连接成功，Quest 数据流应已在 4.3 节单独确认。
2. 输入大写 `HOME` 并回车。脚本以速度 `20` 阻塞回零，然后读取关节角；最大误差不超过 `0.8°` 才继续。
3. 观察真机已正确到位，再输入大写 `READY` 并回车。在此之前，手柄输入不会触发遥操作。
4. 查看实时状态：`rg` 是右 Grip，`rt` 是右 Trigger，`pose` 是 Quest 右手柄 XYZ。先小幅移动手柄，确认 `pose` 在变化。
5. 保持右手柄静止，再按住右 Grip。按下的第一个有效样本只建立增量基准，此时机械臂不应跳动；若发生明显跳动，立即松开 Grip 并按 `q` 结束。确认静止稳定后，再沿单一 XYZ 方向缓慢移动右手柄。松开 Grip 立即暂停跟随，可在不动机械臂的情况下重新调整手的位置。
6. 右 Trigger 完整按下一次，夹爪在开和闭之间切换。如果要再次切换，先完全松开 Trigger，然后再按。此最小测试使用 toggle，不是参考项目的模拟量夹爪映射。
7. 全部项目通过后，脚本会自动结束遥控并低速回零。需要提前结束时按 `q`，脚本也会停止遥控并回零。不要把 `Ctrl+C` 当作正常退出方式。

末端姿态在测试期间锁定，Quest 位移缩放默认为 `0.2`，目标 XYZ 会被限制在 4.5 节的工作空间内。首次真机只建议在 `0.1` 至 `0.3` 之间调整；确认安全前不要使用参考项目的 `1.0`。脚本使用与两个参考项目一致的 Quest 坐标轴映射 `[[0,0,1],[1,0,0],[0,1,0]]`，并在机械臂 XY 平面上校正 `180°` yaw。末端命令使用 RM CANFD 低跟随模式，不启用对实时周期要求更高的高跟随。

### 4.8 与参考项目的差异

| 项目 | 本最小测试 | Quest3-Teleoperation / lerobot-realman-vla |
| --- | --- | --- |
| 位移缩放 | 默认 `0.2`，首次建议 `0.1`–`0.3` | 参考实机命令中可为 `1.0` 或由用户调整 |
| 末端姿态 | 始终锁定，只测 XYZ | 可选跟随手柄旋转 |
| 右 Trigger | 每次按下切换夹爪开/闭 | 不同流程可使用 toggle 或模拟量控制 |
| 左手柄 | 不使用 | 用于录制和保存 |
| 正常退出 | `q`，然后回零 | 参考脚本的退出和收尾行为各不相同 |
| 其他组件 | 无 ROS 2、Pink、相机或数据写入 | 可包含 IK、相机和数据采集 |

不要将参考项目中的位移缩放、`Ctrl+C` 退出、左手柄录制或自动抬升流程套用到本测试。

### 4.9 通过标准

终端会分开显示“命令返回成功”与“硬件状态确实变化”，不会只因 API 返回成功就判定真机通过。完整通过时应看到：

```text
Quest输入=OK | 末端命令=OK | 末端实际位移=OK | 夹爪命令=OK | 夹爪状态=OK
[PASS] Quest→PC Service→RM65→夹爪完整控制链路已验证。
```

提前按 `q` 时仍会正常回零，但只要有一项仍为 `WAIT`，测试就会以未通过状态退出。

### 4.10 常见问题

- `No module named xrobotoolkit_sdk`：未在 `lerobot` 环境安装 binding，或当前不是该 Conda 环境。
- 提示找不到 `libPXREARobotSDK.so`：先执行 `source .env.local`，再检查 `LD_LIBRARY_PATH` 中的 Pybind `lib/` 和 PC Service `SDK/x64/`。
- `xrt.init()` 失败或连接被拒绝：先用 `pgrep` 和 `ss` 确认 PC Service 进程与 `60061` 监听。
- Quest 位姿始终为全零：检查 Quest App 是否显示已连接、PC 地址是否选对、头显是否唤醒、右手柄是否正在跟踪，以及局域网是否启用了客户端隔离。
- `无法连接 RealMan`：确认 `ARM_IP`、PC 网口地址、路由、端口和控制器上电状态，并停止其他可能占用机械臂连接的程序。
- 回零超时或最大关节误差超过 `0.8°`：不要继续遥操作，检查机械臂告警、软限位和零位可达性。
- `末端命令=OK` 但 `末端实际位移=WAIT`：保持按住 Grip 并缓慢增大手柄位移；不要为了通过而调大位移缩放。
- `夹爪命令=OK` 但 `夹爪状态=WAIT`：确认夹爪未被物体卡住，并检查 RM_API2 返回的夹爪位置是否变化。
- 运行中出现机械臂通信或状态读取异常：脚本会停止继续发送运动命令，**不会**再尝试自动回零。请人工确认机械臂状态，必要时使用急停。

## 5. 最小验证

```bash
conda activate lerobot
source .env.local
python scripts/tools/check_env.py --skip-hardware
python scripts/collect_data/convert_realman_hdf5_to_lerobot.py --help
python scripts/train_smolvla_realman/train_realman_smolvla.py --help
```

真机使用前额外执行：

```bash
python scripts/tools/check_env.py --rm-api-path "$RM_API2_PYTHON_PATH"
```

Quest 采集还应确认 `xrobotoolkit_sdk` 可以导入、PC Service 正在运行、Quest App 已连接。
