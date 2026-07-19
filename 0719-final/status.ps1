$ErrorActionPreference = "Continue"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
try {
	$null = & chcp 65001 2>$null
}
catch {
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $ProjectRoot "deploy_config.ps1"

function T {
	param([string]$EscapedText)
	return [System.Text.RegularExpressions.Regex]::Unescape($EscapedText)
}

function Write-Section {
	param([string]$Title)
	Write-Host ""
	Write-Host ("==== " + (T $Title) + " ====") -ForegroundColor Cyan
}

function Write-Ok {
	param([string]$Message)
	Write-Host ((T "\u3010\u6210\u529f\u3011") + $Message) -ForegroundColor Green
}

function Write-Warn {
	param([string]$Message)
	Write-Host ((T "\u3010\u8b66\u544a\u3011") + $Message) -ForegroundColor Yellow
}

function Write-Bad {
	param([string]$Message)
	Write-Host ((T "\u3010\u5931\u8d25\u3011") + $Message) -ForegroundColor Red
}

function Quote-WindowsArgument {
	param([string]$Value)

	if ($Value -notmatch '[\s"]') {
		return $Value
	}

	return '"' + ($Value -replace '\\', '\\' -replace '"', '\"') + '"'
}

function Quote-BashArgument {
	param([string]$Value)
	return "'" + ($Value -replace "'", "'\''") + "'"
}

function Require-Config {
	if (-not (Test-Path -LiteralPath $ConfigPath)) {
		Write-Section "\u0053\u0053\u0048\u0020\u8fde\u63a5"
		Write-Bad (T "\u672a\u627e\u5230 deploy_config.ps1\u3002\u8bf7\u5148\u590d\u5236 deploy_config.example.ps1 \u4e3a deploy_config.ps1\u3002")
		exit 1
	}

	. $ConfigPath

	$script:OrangePiIp = $OrangePiIp
	$script:SshUser = $SshUser
	$script:RemoteProjectDir = $RemoteProjectDir
	$script:SshPort = $SshPort

	if (-not $script:SshPort) {
		$script:SshPort = 22
	}

	$missing = @()
	if ([string]::IsNullOrWhiteSpace($script:OrangePiIp)) { $missing += "OrangePiIp" }
	if ([string]::IsNullOrWhiteSpace($script:SshUser)) { $missing += "SshUser" }
	if ([string]::IsNullOrWhiteSpace($script:RemoteProjectDir)) { $missing += "RemoteProjectDir" }

	if ($missing.Count -gt 0) {
		Write-Section "\u0053\u0053\u0048\u0020\u8fde\u63a5"
		Write-Bad ((T "\u914d\u7f6e\u9879\u4e0d\u5b8c\u6574\uff1a") + ($missing -join ", "))
		exit 1
	}
}

function Test-SshCommand {
	if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
		Write-Bad (T "\u672a\u627e\u5230 ssh \u547d\u4ee4\u3002\u8bf7\u5b89\u88c5 Windows OpenSSH \u5ba2\u6237\u7aef\u3002")
		return $false
	}

	Write-Ok (T "Windows \u5df2\u627e\u5230 ssh \u547d\u4ee4\u3002")
	return $true
}

function Get-SshArgumentText {
	param(
		[string]$RemoteCommand,
		[int]$TimeoutSeconds = 8
	)

	$target = "$($script:SshUser)@$($script:OrangePiIp)"
	$args = @(
		"-o", "ConnectTimeout=$TimeoutSeconds",
		"-o", "ServerAliveInterval=5",
		"-p", "$($script:SshPort)",
		$target,
		$RemoteCommand
	)

	return ($args | ForEach-Object { Quote-WindowsArgument $_ }) -join " "
}

function Invoke-SshRaw {
	param(
		[string]$Command,
		[int]$TimeoutSeconds = 8
	)

	$psi = [System.Diagnostics.ProcessStartInfo]::new()
	$psi.FileName = "ssh"
	$psi.Arguments = Get-SshArgumentText -RemoteCommand $Command -TimeoutSeconds $TimeoutSeconds
	$psi.RedirectStandardOutput = $true
	$psi.RedirectStandardError = $true
	$psi.UseShellExecute = $false
	$psi.CreateNoWindow = $true
	$psi.StandardOutputEncoding = [System.Text.UTF8Encoding]::new($false)
	$psi.StandardErrorEncoding = [System.Text.UTF8Encoding]::new($false)

	$process = [System.Diagnostics.Process]::Start($psi)
	$stdout = $process.StandardOutput.ReadToEnd()
	$stderr = $process.StandardError.ReadToEnd()
	$process.WaitForExit()

	$output = (($stdout, $stderr) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"
	return [pscustomobject]@{
		ExitCode = $process.ExitCode
		Output = $output.TrimEnd()
	}
}

function Invoke-RemoteBash {
	param(
		[string]$BashScript,
		[string[]]$Arguments = @(),
		[switch]$SyntaxOnly,
		[int]$TimeoutSeconds = 12
	)

	$remoteCommand = "LC_ALL=C LANG=C bash"
	if ($SyntaxOnly) {
		$remoteCommand += " -n"
	}
	$remoteCommand += " -s"

	if ($Arguments.Count -gt 0) {
		$remoteCommand += " --"
		foreach ($arg in $Arguments) {
			$remoteCommand += " " + (Quote-BashArgument $arg)
		}
	}

	$normalizedScript = ($BashScript -replace "`r`n", "`n" -replace "`r", "`n")
	if (-not $normalizedScript.EndsWith("`n")) {
		$normalizedScript += "`n"
	}

	$psi = [System.Diagnostics.ProcessStartInfo]::new()
	$psi.FileName = "ssh"
	$psi.Arguments = Get-SshArgumentText -RemoteCommand $remoteCommand -TimeoutSeconds $TimeoutSeconds
	$psi.RedirectStandardInput = $true
	$psi.RedirectStandardOutput = $true
	$psi.RedirectStandardError = $true
	$psi.UseShellExecute = $false
	$psi.CreateNoWindow = $true
	$psi.StandardOutputEncoding = [System.Text.UTF8Encoding]::new($false)
	$psi.StandardErrorEncoding = [System.Text.UTF8Encoding]::new($false)

	$process = [System.Diagnostics.Process]::Start($psi)
	$process.StandardInput.NewLine = "`n"
	$process.StandardInput.Write($normalizedScript)
	$process.StandardInput.Close()

	$stdout = $process.StandardOutput.ReadToEnd()
	$stderr = $process.StandardError.ReadToEnd()
	$process.WaitForExit()

	$output = (($stdout, $stderr) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join "`n"
	return [pscustomobject]@{
		ExitCode = $process.ExitCode
		Output = $output.TrimEnd()
	}
}

function Explain-SshFailure {
	param([string]$Text)

	if ($Text -match "Connection timed out|No route to host|Network is unreachable|Could not resolve hostname") {
		return T "\u7f51\u7edc\u4e0d\u901a\uff1a\u8bf7\u68c0\u67e5 IP\u3001\u5c40\u57df\u7f51\u3001\u7f51\u7edc\u8fde\u63a5\u548c\u9632\u706b\u5899\u3002"
	}

	if ($Text -match "Connection refused") {
		return T "SSH \u670d\u52a1\u88ab\u62d2\u7edd\uff1a\u8bf7\u68c0\u67e5 Orange Pi \u4e0a ssh \u670d\u52a1\u662f\u5426\u5f00\u542f\uff0c\u7aef\u53e3\u662f\u5426\u6b63\u786e\u3002"
	}

	if ($Text -match "Permission denied") {
		return T "\u8ba4\u8bc1\u5931\u8d25\uff1a\u8bf7\u68c0\u67e5\u7528\u6237\u540d\u3001\u5bc6\u7801\u6216 SSH \u5bc6\u94a5\u3002"
	}

	if ($Text -match "Host key verification failed|REMOTE HOST IDENTIFICATION HAS CHANGED") {
		return T "Host key \u6821\u9a8c\u5931\u8d25\uff1a\u8bf7\u786e\u8ba4\u8bbe\u5907\u8eab\u4efd\uff0c\u5fc5\u8981\u65f6\u6267\u884c ssh-keygen -R ORANGE_PI_IP\u3002"
	}

	return T "SSH \u8fde\u63a5\u5931\u8d25\uff1a\u8bf7\u68c0\u67e5\u7f51\u7edc\u3001\u7aef\u53e3\u3001\u7528\u6237\u540d\u548c\u8ba4\u8bc1\u65b9\u5f0f\u3002"
}

function Show-RemoteBlock {
	param(
		[string]$Title,
		[string]$BashScript,
		[string[]]$Arguments = @(),
		[int]$TimeoutSeconds = 12
	)

	Write-Section $Title

	$syntax = Invoke-RemoteBash -BashScript $BashScript -Arguments $Arguments -SyntaxOnly -TimeoutSeconds $TimeoutSeconds
	if ($syntax.ExitCode -ne 0) {
		Write-Warn ((T "\u8fdc\u7a0b Bash \u8bed\u6cd5\u68c0\u67e5\u5931\u8d25\uff0c\u8df3\u8fc7\u672c\u533a\u5757\u3002\u8f93\u51fa\uff1a") + "`n" + $syntax.Output)
		return
	}

	$result = Invoke-RemoteBash -BashScript $BashScript -Arguments $Arguments -TimeoutSeconds $TimeoutSeconds
	if ($result.ExitCode -ne 0) {
		Write-Warn ((T "\u8fdc\u7a0b\u67e5\u8be2\u672a\u5b8c\u5168\u6210\u529f\uff0c\u7ee7\u7eed\u4e0b\u4e00\u9879\u3002\u8f93\u51fa\uff1a") + "`n" + $result.Output)
		return
	}

	if ([string]::IsNullOrWhiteSpace($result.Output)) {
		Write-Warn (T "\u6ca1\u6709\u8fd4\u56de\u5185\u5bb9\u3002")
	}
	else {
		Write-Host $result.Output
	}
}

Require-Config

Write-Section "\u0053\u0053\u0048\u0020\u8fde\u63a5"
if (-not (Test-SshCommand)) {
	exit 1
}

Write-Host ((T "\u76ee\u6807\uff1a") + "$($script:SshUser)@$($script:OrangePiIp):$($script:SshPort)")
$sshTest = Invoke-SshRaw -Command "echo STATUS_OK" -TimeoutSeconds 8

if ($sshTest.ExitCode -ne 0 -or $sshTest.Output -notmatch "STATUS_OK") {
	Write-Bad (Explain-SshFailure $sshTest.Output)
	if (-not [string]::IsNullOrWhiteSpace($sshTest.Output)) {
		Write-Host $sshTest.Output
	}
	exit 1
}

Write-Ok (T "SSH \u8fde\u63a5\u6210\u529f\u3002")

$systemScript = @'
set -u
export LC_ALL=C
export LANG=C

echo '[Hostname]'
hostname 2>/dev/null || echo 'unknown'

echo
echo '[System time]'
date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date

echo
echo '[Uptime]'
uptime -p 2>/dev/null || uptime 2>/dev/null || echo 'unknown'

echo
echo '[IP addresses]'
hostname -I 2>/dev/null || ip -4 addr show 2>/dev/null | awk '/inet / {print $2}'

echo
echo '[CPU load]'
cat /proc/loadavg 2>/dev/null || echo 'unknown'

echo
echo '[CPU temperature]'
if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
	temp=$(cat /sys/class/thermal/thermal_zone0/temp)
	awk -v t="$temp" 'BEGIN { printf "%.1f C\n", t / 1000 }'
else
	echo 'unavailable'
fi

echo
echo '[Memory]'
free -h 2>/dev/null || echo 'free unavailable'

echo
echo '[Root disk]'
df -h / 2>/dev/null || echo 'df unavailable'

echo
echo '[SSH service]'
systemctl is-active ssh 2>/dev/null || echo 'unknown'
'@
Show-RemoteBlock "\u7cfb\u7edf\u72b6\u6001" $systemScript

$visionScript = @'
set -u
export LC_ALL=C
export LANG=C

PROJECT_DIR="$1"
PID_FILE="$PROJECT_DIR/vision.pid"

echo '[Project directory]'
if [ -d "$PROJECT_DIR" ]; then
	echo "exists: $PROJECT_DIR"
else
	echo "missing: $PROJECT_DIR"
	exit 0
fi

echo
echo '[PID file]'
if [ ! -f "$PID_FILE" ]; then
	echo 'not found'
	exit 0
fi

pid=$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)
if [ -z "$pid" ]; then
	echo 'invalid: empty PID file'
	exit 0
fi

case "$pid" in
	*[!0-9]*)
		echo 'invalid: PID is not numeric'
		exit 0
		;;
esac

echo "PID: $pid"

if [ ! -d "/proc/$pid" ]; then
	echo 'warning: PID file is stale, process does not exist'
	exit 0
fi

echo
echo '[Process]'
ps -p "$pid" -o pid=,etime=,%cpu=,%mem=,args= 2>/dev/null || echo 'ps query failed'

cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)

echo
echo '[Verification]'
echo "cwd: $cwd"
echo "cmdline: $cmd"
case "$cmd" in
	*"$PROJECT_DIR"*|*main.py*) echo 'process looks related to this project' ;;
	*) echo 'warning: process command line does not clearly match this project' ;;
esac
'@
Show-RemoteBlock "\u89c6\u89c9\u7a0b\u5e8f\u72b6\u6001" $visionScript @($script:RemoteProjectDir)

$cameraScript = @'
set -u
export LC_ALL=C
export LANG=C

echo '[Video nodes]'
ls -l /dev/video* 2>/dev/null || echo 'no video nodes'

echo
echo '[/dev/video0]'
if [ -e /dev/video0 ]; then
	ls -l /dev/video0
else
	echo 'not found'
fi

echo
echo '[v4l2-ctl]'
if command -v v4l2-ctl >/dev/null 2>&1; then
	v4l2-ctl --list-devices
else
	echo 'v4l2-ctl not installed'
fi
'@
Show-RemoteBlock "\u6444\u50cf\u5934\u72b6\u6001" $cameraScript

$logScript = @'
set -u
export LC_ALL=C
export LANG=C

PROJECT_DIR="$1"
LOG_DIR="$PROJECT_DIR/logs"

echo '[Log directory]'
if [ ! -d "$LOG_DIR" ]; then
	echo 'no logs yet'
	exit 0
fi

latest=$(find "$LOG_DIR" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -z "$latest" ]; then
	echo 'no logs yet'
	exit 0
fi

echo '[Latest log]'
if command -v stat >/dev/null 2>&1; then
	stat -c 'file: %n%nsize: %s bytes%nmodified: %y' "$latest" 2>/dev/null || ls -lh "$latest"
else
	ls -lh "$latest"
fi

echo
echo '[Last 20 lines]'
tail -n 20 "$latest" 2>&1 || echo 'cannot read latest log'
'@
Show-RemoteBlock "\u6700\u8fd1\u65e5\u5fd7" $logScript @($script:RemoteProjectDir)
