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
- `增加球定点判断版本.py`: current stable detector source kept as the algorithm reference
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
