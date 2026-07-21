# SORT VISION 0720-final

本目录是当前用于 Orange Pi 5 Pro 的 OpenCV 视觉识别版本。程序在 Orange Pi 本机运行，USB 工业相机接在 Orange Pi 上，画面通过 Orange Pi 本地显示屏使用 `cv2.imshow` 显示。

本项目不使用 Flask、网页视频流、VNC 或远程桌面。Windows 端主要负责 VS Code 开发、PowerShell 一键同步和状态查看。

## 当前能力

- 识别 A/B 两个货物区域，并以最终选中的绿色 A/B 候选对作为唯一正式数据源。
- 基于原始视角裁剪图进行物块识别，透视变换主要用于坐标映射和角度计算。
- 支持 `pink`、`blue`、`orange`、`green` 四种视觉颜色。
- 支持 `cube`、`cuboid`、`ball` 三类形状。
- 支持机械臂平面坐标 `X/Y` 输出，单位为 cm，发送前乘 100 并按有符号 int16 大端编码。
- 支持机械爪角度计算、发送补偿和角度帧发送。
- 支持 STM32 串口题目选择、发挥题配置、HOME 回零事件、强制复位。
- 支持 Windows 到 Orange Pi 的 `deploy.ps1` 同步、运行、停止、重启。
- 支持 `status.ps1` 只读查看 Orange Pi 状态、摄像头节点、进程和日志。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `main.py` | 程序入口，负责摄像头读取、串口轮询、视觉任务调度、OpenCV 窗口显示 |
| `config.py` | 摄像头、窗口、串口、A/B 区、稳定帧、坐标补偿、角度补偿等参数 |
| `camera.py` | 摄像头初始化、读取和释放 |
| `detector.py` | 当前稳定视觉算法的封装层，连接 `first.py` 中的检测逻辑 |
| `first.py` | 旧稳定视觉检测源代码，作为底层算法来源 |
| `target_selector.py` | A/B 区配对、区域裁剪、目标列表构建、各题目标选择 |
| `coordinate.py` | A/B 统一平面映射、机械坐标计算、角度计算 |
| `vision_task.py` | 视觉任务执行器：稳定 A/B 区、稳定目标、发送角度帧和坐标帧 |
| `serial_protocol.py` | 6 字节串口协议、题号映射、配置键值、角度/坐标帧打包 |
| `serial_comm.py` | pyserial 串口线程、bytearray 缓存、半帧/粘包处理、RX/TX 日志 |
| `task_state.py` | 题目选择、配置接收、HOME 计数、任务状态机 |
| `display.py` | OpenCV 窗口显示和键盘处理 |
| `roi_utils.py` | ROI 辅助函数 |
| `run.sh` | Orange Pi 本地启动脚本 |
| `deploy.ps1` | Windows 一键同步、运行、停止、重启脚本 |
| `status.ps1` | Windows 只读状态检测脚本 |
| `deploy_config.example.ps1` | 部署配置模板 |
| `requirements.txt` | Python 依赖 |
| `tests/test_protocol.py` | 不依赖真实摄像头和 STM32 的轻量测试 |

## Orange Pi 环境准备

建议在 Orange Pi 上安装系统 OpenCV，这样 `cv2.imshow` 更容易正常连接本机桌面环境：

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-opencv python3-numpy
pip3 install -r requirements.txt
```

如果使用项目内虚拟环境，`run.sh` 会自动激活 `.venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

串口依赖：

```bash
pip3 install pyserial
```

## 在 Orange Pi 本机运行

进入项目目录：

```bash
cd /home/orangepi/sort_vision
chmod +x run.sh
./run.sh
```

`run.sh` 会：

- 切换到脚本所在目录；
- 自动激活 `.venv`；
- 检查 `main.py`；
- 检查 `cv2` 和 `numpy`；
- 设置本地显示环境变量；
- 使用 `python3 main.py` 启动。

程序退出快捷键：

- `q`：退出
- `c`：强制重新获取黑框
- `w`：切换透视显示
- `m`：显示/隐藏颜色 mask
- `b`：显示/隐藏黑色 mask
- `f`：显示/隐藏候选框调试信息

## Windows 一键同步到 Orange Pi

先进入 `0720-final` 目录：

```powershell
cd D:\py_pro\电赛模拟赛\2026-07-17_1023_opi-opencv-modular\0720-final
```

首次复制配置文件：

```powershell
Copy-Item .\deploy_config.example.ps1 .\deploy_config.ps1
notepad .\deploy_config.ps1
```

配置示例：

```powershell
$OrangePiIp = "192.168.1.100"
$SshUser = "orangepi"
$RemoteProjectDir = "/home/orangepi/sort_vision"
$SshPort = 22
$RunAfterSync = $false
```

不要在配置文件中保存 SSH 密码。`deploy_config.ps1` 是本机真实配置文件，应该保持被 Git 忽略。

只同步代码：

```powershell
.\deploy.ps1
```

同步并启动：

```powershell
.\deploy.ps1 -Run
```

停止本项目启动的远程程序：

```powershell
.\deploy.ps1 -Stop
```

重启：

```powershell
.\deploy.ps1 -Restart
```

远程运行日志保存在：

```text
/home/orangepi/sort_vision/logs/
```

远程进程 PID 保存在：

```text
/home/orangepi/sort_vision/vision.pid
```

停止逻辑只会基于 `vision.pid` 停止本项目进程，不使用 `pkill python` 或 `killall python`。

## 查看 Orange Pi 状态

```powershell
.\status.ps1
```

脚本会只读查询：

- SSH 连接状态；
- 主机名、时间、运行时间；
- IP、CPU 负载、CPU 温度；
- 内存、Swap、根目录磁盘；
- SSH 服务状态；
- `vision.pid` 对应视觉进程；
- `/dev/video*` 摄像头节点；
- `v4l2-ctl --list-devices`；
- 最近日志文件和最后 20 行。

`status.ps1` 不会修改 Orange Pi 上的文件，也不会打开摄像头。

## 串口配置

当前串口参数在 `config.py`：

```python
SERIAL_PORT = "/dev/ttyS2"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0
```

接收线程在 `serial_comm.py` 中实现：

- 使用 `bytearray` 缓存连续字节流；
- 支持半帧；
- 支持一次收到多个 6 字节帧；
- 支持错位重同步；
- 根据帧头检查帧尾；
- 输出 `[RX]` 和 `[TX]` 十六进制日志。

## STM32 到视觉模块帧

所有帧固定 6 字节。

题目选择：

```text
01 00 00 00 TASK_ID 10
```

当前题号：

| 题目 | TASK_ID |
| --- | --- |
| 基础 1-1 | `0x11` |
| 基础 1-2 | `0x22` |
| 基础 1-3 | `0x33` |
| 发挥 2-1 | `0x44` |
| 发挥 2-2 | `0x55` |
| 发挥 2-3 | `0x66` |

配置帧：

```text
02 LAST_FLAG POSITION_KEY SHAPE_KEY COLOR_KEY 20
```

- `LAST_FLAG=0x01` 表示配置列表结束；
- `POSITION_KEY` 当前只保存，不参与视觉匹配；
- 发挥 2-2 和发挥 2-3 根据配置列表数量确定目标数量。

形状键值：

| 形状 | SHAPE_KEY |
| --- | --- |
| 任意 | `0x00` |
| 正方体 cube | `0x05` |
| 长方体 cuboid | `0x06` |
| 球 ball | `0x07` |

颜色键值到视觉颜色映射：

| 串口颜色 | COLOR_KEY | 视觉颜色 |
| --- | --- | --- |
| 任意 | `0x00` | any |
| 红色 | `0x41` | pink |
| 蓝色 | `0x42` | blue |
| 黑色 | `0x43` | green |
| 黄色 | `0x44` | orange |

HOME 回零帧：

```text
05 01 00 00 00 50
```

强制复位：

```text
05 FF 00 00 00 50
```

HOME 帧 500 ms 内重复只计一次，参数：

```python
HOME_EVENT_DEBOUNCE_SEC = 0.5
```

## 视觉到 STM32 帧

识别成功后发送顺序固定：

```text
机械爪角度帧 -> 机械坐标帧
```

角度帧：

```text
04 angleH angleL 00 00 40
```

- 角度单位：度；
- 发送前乘 100；
- int16 有符号补码；
- 高字节在前；
- 发送角度使用 `gripper_angle + GRIPPER_TX_ANGLE_OFFSET_DEG` 后归一化到 `[-90, 90]`。

坐标帧：

```text
03 xH xL yH yL 30
```

- X/Y 单位：cm；
- 发送前乘 100；
- int16 有符号补码；
- 高字节在前；
- 坐标帧格式保持不变。

## 坐标和角度

坐标转换在 `coordinate.py`：

- A/B 两个区域使用统一平面；
- 机械臂原点由 A/B 区几何关系推导；
- `frame_point_to_arm_coordinate()` 输出原始机械坐标；
- 角度计算使用原始机械坐标。

最终发送前在 `vision_task.py` 加机械坐标偏移：

```python
x_send = x_cm + ARM_X_OFFSET_CM
y_send = y_cm + ARM_Y_OFFSET_CM
```

偏移参数在 `config.py`：

```python
ARM_X_OFFSET_CM = 0.0
ARM_Y_OFFSET_CM = 0.0
```

机械爪角度发送补偿：

```python
GRIPPER_TX_ANGLE_OFFSET_DEG = 90.0
```

日志示例：

```text
[COORD]
raw: x=-3.30 cm, y=-4.10 cm
offset: x=+0.00 cm, y=+0.00 cm
send: x=-3.30 cm, y=-4.10 cm

[ANGLE] shape=cube, object_axis=109.91 deg, base=-113.88 deg, gripper=43.79 deg
[ANGLE_TX] raw_gripper=43.79 deg, offset=+90.00 deg, tx=-46.21 deg
[SERIAL] send angle frame: 04 ED F3 00 00 40
[SERIAL] send coordinate frame: 03 FE B6 FE 66 30
```

## A/B 区和目标选择

场地固定：

- 左侧区域为 A；
- 右侧区域为 B；
- 以最终配对成功的绿色 A/B 候选对作为正式数据源。

目标选择规则：

- 基础 1-1：选择 A 区最右侧物体。
- 基础 1-2：全画面从左到右选择第一个正方体。
- 基础 1-3：每次重新拍照，选择当前剩余正方体中最左侧目标，共 4 个。
- 发挥 2-1：顺序为 `pink cube`、`blue cube`、`green cube`。
- 发挥 2-2：按配置列表中的 shape/color 匹配目标。
- 发挥 2-3：按配置队列顺序逐个匹配 shape/color。

## 调试窗口

窗口开关在 `config.py`：

```python
SHOW_SOURCE_CANDIDATES_WINDOW = True
SHOW_RESULT_WINDOW = False
SHOW_AREA_WINDOWS = True
SHOW_AREA_RESULT_WINDOWS = True
SHOW_AREA_WARP_WINDOWS = False
SHOW_BLACK_MASK_WINDOW = False
SHOW_COLOR_MASK_WINDOWS = False
```

常用窗口：

- `source_candidates`：原始画面和 A/B 候选调试；
- `area_A`、`area_B`：A/B 原始视角裁剪图；
- `area_A_result`、`area_B_result`：A/B 区识别结果；
- `area_A_warp`、`area_B_warp`：可选透视调试窗口；
- `result`：全局识别和状态叠字窗口。

`DEBUG_DRAW_ENABLED` 只控制辅助绘制，不应该影响 `cv2.imshow` 是否显示窗口。

## 现场可调参数

常调参数集中在 `config.py`：

- 摄像头：`CAMERA_ID`、`CAMERA_WIDTH`、`CAMERA_HEIGHT`
- 窗口开关：`SHOW_*`
- 串口：`SERIAL_PORT`、`SERIAL_BAUD`
- 稳定帧：`CALIBRATION_STABLE_FRAMES`、`TARGET_STABLE_FRAMES`
- HOME 去重：`HOME_EVENT_DEBOUNCE_SEC`
- 角度发送补偿：`GRIPPER_TX_ANGLE_OFFSET_DEG`
- 坐标发送补偿：`ARM_X_OFFSET_CM`、`ARM_Y_OFFSET_CM`
- A/B 区配对：`PAIR_*`
- 中线检测：`DIVIDER_*`
- 黑框去重：`BLACK_FRAME_*`
- 目标过滤：`OBJECT_*`、`MIN_OBJECT_*`、`MAX_OBJECT_*`
- 正方体抗阴影兜底：`SHADOW_CUBE_*`
- 粉色 LAB 兜底：`PINK_LAB_*`
- 橙色阈值：`ORANGE_HSV_RANGES`

## 测试

本地不需要真实摄像头和 STM32，可运行轻量测试：

```powershell
py -3.12 -m unittest tests.test_protocol
```

语法检查：

```powershell
py -3.12 -X pycache_prefix=$env:TEMP\sort_vision_pycache -m compileall .
```

如果 Windows 当前 `python` 命令不可用，可以使用 `py -3.12` 或 VS Code 当前解释器路径。

## 常见问题

`.\deploy.ps1` 无法识别：

请确认 PowerShell 已进入 `0720-final` 目录：

```powershell
cd D:\py_pro\电赛模拟赛\2026-07-17_1023_opi-opencv-modular\0720-final
.\deploy.ps1
```

`cv2` 找不到：

Windows 端需要给 VS Code 选择安装了 OpenCV 的 Python 解释器；Orange Pi 端建议安装：

```bash
sudo apt install -y python3-opencv python3-numpy
```

`cv2.imshow` 没有窗口：

- 确认程序在 Orange Pi 本机桌面环境中运行；
- 确认 `DISPLAY` 有效；
- 确认至少一个 `SHOW_*_WINDOW` 为 `True`；
- 不要通过纯 SSH 终端期待 Windows 端出现 OpenCV 窗口。

串口打不开：

- 检查 `/dev/ttyS2` 是否存在；
- 检查用户是否有串口权限；
- 检查 STM32 是否占用或连接异常；
- 根据现场情况可临时关闭串口硬件，仅做视觉调试。

部署后程序没有启动：

```powershell
.\status.ps1
```

然后查看远程日志：

```bash
tail -n 80 /home/orangepi/sort_vision/logs/最新日志文件
```

## Git 使用建议

GitHub 用于稳定版本备份，不作为每次现场调试的同步方式。现场快速调试建议使用：

```powershell
.\deploy.ps1
.\deploy.ps1 -Run
.\deploy.ps1 -Restart
```

确认稳定后再提交到 Git。
