# Orange Pi OpenCV Vision Project

This project is organized for running directly on Orange Pi 5 Pro with a USB camera and local `cv2.imshow` display output.

## Files

- `main.py`: program entry point and main processing loop
- `config.py`: camera and window configuration
- `camera.py`: camera initialization, frame reading, and release
- `detector.py`: detector API wrapper around the current stable vision algorithm
- `display.py`: OpenCV windows and keyboard handling
- `run.sh`: Orange Pi startup script
- `requirements.txt`: Python dependencies
- `first.py`: current stable detector source kept as the algorithm reference
- `color1.py`: older/reference version kept unchanged

## Run On Orange Pi

Install dependencies:

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
pip3 install -r requirements.txt
```

`python3-opencv` is recommended on Orange Pi/RK3588 because it keeps `cv2.imshow` and local display support tied to the Ubuntu desktop libraries.

Run:

```bash
chmod +x run.sh
./run.sh
```

Or:

```bash
python3 main.py
```

## Windows One-Click Deploy

This project uses Windows built-in `ssh` and `scp` to sync code to Orange Pi over the local network. GitHub is only for stable version management, not for every small debugging sync.

Enable SSH on Orange Pi:

```bash
sudo apt update
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
hostname -I
```

Test SSH from Windows PowerShell:

```powershell
ssh orangepi@ORANGE_PI_IP
```

Optional SSH key login from Windows:

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh orangepi@ORANGE_PI_IP "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
ssh orangepi@ORANGE_PI_IP
```

Create local deploy config:

```powershell
Copy-Item .\deploy_config.example.ps1 .\deploy_config.ps1
notepad .\deploy_config.ps1
```

Edit at least:

```powershell
$OrangePiIp = "192.168.1.100"
$SshUser = "orangepi"
$RemoteProjectDir = "/home/orangepi/sort_vision"
$SshPort = 22
$RunAfterSync = $false
```

Do not put SSH passwords, private keys, or real secrets in `deploy_config.ps1`.

Sync code only:

```powershell
.\deploy.ps1
```

Sync and start on Orange Pi:

```powershell
.\deploy.ps1 -Run
```

Stop the program started by this project:

```powershell
.\deploy.ps1 -Stop
```

Restart:

```powershell
.\deploy.ps1 -Restart
```

Check Orange Pi status without changing remote files:

```powershell
.\status.ps1
```

The status script reads `deploy_config.ps1`, connects by SSH, and shows these read-only sections:

- SSH connection
- System status
- Vision program status from `vision.pid`
- Camera status without opening the camera
- Latest log file and last 20 lines

It does not stop, restart, delete, install, or create files on Orange Pi. It does not use `pkill`, `killall`, or `kill`.

View logs on Orange Pi:

```bash
cd /home/orangepi/sort_vision
ls -lh logs
tail -f logs/vision_*.log
```

The remote process id is saved as:

```bash
/home/orangepi/sort_vision/vision.pid
```

The deploy script only stops the PID recorded by this project after checking that it belongs to this project directory. It does not use `pkill python` or `killall python`.

## Placement ROI And Inner Frames

The current field model uses a configurable placement ROI and two paired inner frames. The full cargo placement area is still physically `40 cm x 15 cm`, but visual detection no longer tries to pick one full outer rectangle from the whole image.

The detector now:

- crops the configured ROI first
- searches only inside that ROI
- detects small A/B inner-frame candidates with an expected ratio near `17 / 13`
- evaluates all candidate pairs by `pair_score`
- pairs two similar candidates that are horizontally aligned
- maps all ROI-local candidate coordinates back to the original camera image
- marks the left candidate as area `A` and the right candidate as area `B`

This avoids selecting the mechanical arm placement rectangle when it appears elsewhere in the image or connects with cargo-area black lines.

Default ROI settings in `config.py` search the lower part of the image:

```python
PLACEMENT_ROI_X_MIN_RATIO = 0.00
PLACEMENT_ROI_X_MAX_RATIO = 1.00
PLACEMENT_ROI_Y_MIN_RATIO = 0.45
PLACEMENT_ROI_Y_MAX_RATIO = 1.00
```

The `source_candidates` debug window shows the ROI boundary, each inner-frame candidate's score, ratio, area ratio, rectangularity, border score, estimated border width, and pair debug labels. Pair labels include score, area/width/height/Y similarity, gap ratio, divider score, accepted/rejected state, and rejection reason. The selected pair is marked with `SELECTED PAIR`, `A`, and `B`.

It does not use internal color uniformity, so red, blue, black, yellow, or other blocks inside the small frames should not make the frame candidate fail by themselves.

When an A/B pair is selected, the program generates independent perspective images from the original camera frame:

- `area_A`
- `area_B`
- `area_A_result`
- `area_B_result`

Object detection runs separately on `area_A` and `area_B`; results are then mapped back through each area's own perspective geometry.

Tune these parameters in `config.py` on site:

```python
PLACEMENT_ROI_X_MIN_RATIO
PLACEMENT_ROI_X_MAX_RATIO
PLACEMENT_ROI_Y_MIN_RATIO
PLACEMENT_ROI_Y_MAX_RATIO
INNER_REGION_RATIO_MIN
INNER_REGION_RATIO_MAX
INNER_REGION_MIN_AREA_RATIO
INNER_REGION_MAX_AREA_RATIO
INNER_REGION_MIN_RECTANGULARITY
PAIR_MAX_AREA_DIFF_RATIO
PAIR_MAX_WIDTH_DIFF_RATIO
PAIR_MAX_HEIGHT_DIFF_RATIO
PAIR_MAX_CENTER_Y_DIFF_RATIO
PAIR_MIN_HORIZONTAL_GAP_RATIO
PAIR_MAX_HORIZONTAL_GAP_RATIO
PAIR_MIN_SCORE
SHOW_AREA_WINDOWS
SHOW_AREA_RESULT_WINDOWS
SHOW_PAIR_CANDIDATES
```

### Display Switches

Formal run defaults hide candidate debug clutter. By default:

- shown: `area_A_result`, `area_B_result`
- hidden: `source_candidates`, `result`, `area_A`, `area_B`, `black_mask`, `color_masks`
- hidden: ROI line, all white candidate boxes, candidate text, pair lines, pair text
- kept: final selected A/B region labels and global object overlay, controlled independently

Main switches in `config.py`:

```python
DEBUG_DRAW_ENABLED = False
DEBUG_DRAW_ROI = False
DEBUG_DRAW_ALL_CANDIDATES = False
DEBUG_DRAW_CANDIDATE_TEXT = False
DEBUG_DRAW_ALL_PAIRS = False
DEBUG_DRAW_PAIR_TEXT = False
DEBUG_DRAW_DIVIDER = False
DEBUG_DRAW_SELECTED_REGIONS = True
DEBUG_DRAW_SELECTED_LABELS = True
DEBUG_DRAW_GLOBAL_OBJECTS = True

SHOW_SOURCE_CANDIDATES_WINDOW = False
SHOW_RESULT_WINDOW = False
SHOW_AREA_WINDOWS = False
SHOW_AREA_RESULT_WINDOWS = True
SHOW_BLACK_MASK_WINDOW = False
SHOW_COLOR_MASK_WINDOWS = False
SHOW_EXTRA_DEBUG_WINDOWS = False
```

For full field debugging, set `DEBUG_DRAW_ENABLED`, `DEBUG_DRAW_ROI`, `DEBUG_DRAW_ALL_CANDIDATES`, `DEBUG_DRAW_CANDIDATE_TEXT`, `DEBUG_DRAW_ALL_PAIRS`, `DEBUG_DRAW_PAIR_TEXT`, `SHOW_SOURCE_CANDIDATES_WINDOW`, `SHOW_RESULT_WINDOW`, and `SHOW_AREA_WINDOWS` to `True`.

## STM32 Serial Protocol

The vision module uses the simplified count-based STM32 serial protocol.

- Serial device: `/dev/ttyS2`
- Baud rate: `115200`
- Timeout: `0`
- Frame length: fixed 6 bytes
- Main entry remains `main.py`
- Serial receive runs in a background thread; vision detection still runs in the main OpenCV loop
- Coordinate frames are event-triggered, sent once, and do not wait for ACK

Install the serial dependency on Orange Pi:

```bash
pip3 install -r requirements.txt
```

If `pyserial` is not available from pip in the field, install it from Ubuntu packages:

```bash
sudo apt install -y python3-serial
```

Run without changing the display workflow:

```bash
./run.sh
```

The program still uses local `cv2.imshow` on the Orange Pi screen. It does not start Flask, VNC, remote desktop, or a web video stream.

### Coordinate Send

Target coordinate conversion is implemented in `coordinate.py`. The program builds one unified A+B planar coordinate system from the stable A/B regions, then sends mechanical arm X/Y coordinates in centimeters.

The mechanical origin is derived from the A/B middle top point, shifted upward by:

```python
ARM_ORIGIN_ABOVE_TOP_CM = 7.5
```

The serial coordinate frame remains 6 bytes:

```text
03 xH xL yH yL 30
```

Both X and Y are signed int16 values in `cm * 100`, big-endian.

### Vision Target Flow

After STM32 sends the fixed home frame `05 01 00 00 00 50`, the main loop starts one visual target round:

- reset current-round frame and object caches
- detect one stable pair of A/B inner frames inside the configured ROI
- detect up to 3 objects per area, 6 total
- map ROI object centers back to one global image coordinate system
- sort targets by global X
- select the current target according to `current_task` and `current_target_index`
- wait for stable target frames
- calculate mechanical arm X/Y coordinates
- send one coordinate frame `03 xH xL yH yL 30`

If stable A/B planar calibration is unavailable, the program prints `[CALIB]` and does not send coordinates.

### Hardware-Free Protocol Tests On Windows

These tests do not require STM32, a serial port, a camera, OpenCV, or NumPy:

```powershell
py -3 -m unittest discover -s .\tests -v
```

The tests cover serial half-frames, sticky packets, wrong tails, force reset, signed X/Y encoding, simplified home counting, task target selection, configuration queues, and placement-frame region splitting.

### First STM32 Joint Debug Frames

Use the following frames in order for the first bench test. All values are hex bytes.

1. Force reset at any time:

```text
05 FF 00 00 00 50
```

2. Select basic task 1-1:

```text
01 00 00 00 11 10
```

3. Notify arm is home and request the next target:

```text
05 01 00 00 00 50
```

4. After the arm finishes and returns home, send the same fixed home frame again:

```text
05 01 00 00 00 50
```

For advanced task configuration frames, use `02 LAST_FLAG SLOT_ID SHAPE_ID COLOR_ID 20`. Color IDs are `00 any`, `01 pink`, `02 green`, `03 blue`, `04 orange`.

### Deploy Troubleshooting

- `Connection timed out`: check Orange Pi IP, same LAN, SSH service, firewall, and port `22`.
- `Permission denied`: check username, password, SSH key, and remote directory permissions.
- `Host key verification failed`: remove the old key with `ssh-keygen -R ORANGE_PI_IP`, then connect again.
- `scp` not found: install Windows OpenSSH Client in Windows Optional Features, or add it to PATH.
- `run.sh` has no execute permission: run `chmod +x run.sh` on Orange Pi. `deploy.ps1` also runs this after syncing.
- `cv2.imshow` cannot open a window: run the program on the Orange Pi local desktop session, confirm the small screen is active, check `DISPLAY=:0`, and avoid launching from a headless SSH-only session without access to the local X display.

## Keys

- `q`: quit
- `c`: force black-frame reacquire
- `w`: toggle perspective warp
- `m`: show/hide color masks
- `b`: show/hide black mask
- `f`: show/hide all black-frame candidates

## Notes

- The camera is opened only once in `camera.py`.
- No Flask, VNC, remote desktop, or web video stream is used.
- Display remains local OpenCV `cv2.imshow`, suitable for the Orange Pi screen.
- If the camera id changes on Orange Pi, edit `CAMERA_ID` in `config.py`.
- For large future changes, copy this whole dated folder first, then modify the copy. This keeps each runnable version easy to roll back.
