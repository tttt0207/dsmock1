param(
	[switch]$Run,
	[switch]$Stop,
	[switch]$Restart
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $ProjectRoot "deploy_config.ps1"
$PidFileName = "vision.pid"

function Write-Info {
	param([string]$Message)
	Write-Host "[INFO] $Message"
}

function Write-Fail {
	param([string]$Message)
	Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Require-Command {
	param([string]$Name)

	if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
		throw "Command not found: $Name. Please install Windows OpenSSH Client and make sure $Name is in PATH."
	}
}

function Require-Config {
	if (-not (Test-Path -LiteralPath $ConfigPath)) {
		throw "Missing deploy_config.ps1. Copy deploy_config.example.ps1 to deploy_config.ps1, then edit your Orange Pi settings."
	}

	. $ConfigPath

	$script:OrangePiIp = $OrangePiIp
	$script:SshUser = $SshUser
	$script:RemoteProjectDir = $RemoteProjectDir
	$script:SshPort = $SshPort
	$script:RunAfterSync = $RunAfterSync

	if ([string]::IsNullOrWhiteSpace($script:OrangePiIp)) {
		throw "Missing config value: OrangePiIp."
	}

	if ([string]::IsNullOrWhiteSpace($script:SshUser)) {
		throw "Missing config value: SshUser."
	}

	if ([string]::IsNullOrWhiteSpace($script:RemoteProjectDir)) {
		throw "Missing config value: RemoteProjectDir."
	}

	if (-not $script:SshPort) {
		$script:SshPort = 22
	}

	if ($null -eq $script:RunAfterSync) {
		$script:RunAfterSync = $false
	}
}

function Shell-Quote {
	param([string]$Value)
	return "'" + ($Value -replace "'", "'\''") + "'"
}

function Invoke-Remote {
	param(
		[string]$Command,
		[string]$FailureMessage
	)

	$target = "$($script:SshUser)@$($script:OrangePiIp)"
	& ssh -p $script:SshPort $target $Command

	if ($LASTEXITCODE -ne 0) {
		throw $FailureMessage
	}
}

function Test-RemoteConnection {
	Write-Info "Testing SSH: $($script:SshUser)@$($script:OrangePiIp):$($script:SshPort)"
	Invoke-Remote "echo connected" "Cannot connect to Orange Pi. Check IP, SSH user, port, network, firewall, password, or SSH key."
}

function New-RemoteProjectDir {
	$remoteDir = Shell-Quote $script:RemoteProjectDir
	Invoke-Remote "mkdir -p $remoteDir" "Cannot create remote project directory: $($script:RemoteProjectDir)."
}

function Should-SkipPath {
	param([System.IO.FileSystemInfo]$Item)

	$excludedNames = @(
		".git",
		".github",
		".vscode",
		".venv",
		"venv",
		"__pycache__",
		"output",
		"outputs",
		"logs",
		"dataset",
		"datasets",
		"deploy_config.ps1",
		"增加球定点判断版本.py"
	)

	if ($excludedNames -contains $Item.Name) {
		return $true
	}

	if (-not $Item.PSIsContainer) {
		$extension = $Item.Extension.ToLowerInvariant()
		$excludedExtensions = @(
			".pyc",
			".pyo",
			".pid",
			".log",
			".tmp",
			".jpg",
			".jpeg",
			".png",
			".bmp",
			".gif",
			".tif",
			".tiff",
			".avi",
			".mp4",
			".mkv",
			".mov"
		)

		if ($excludedExtensions -contains $extension) {
			return $true
		}
	}

	return $false
}

function Copy-FilteredProject {
	param(
		[string]$SourceDir,
		[string]$TargetDir
	)

	New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

	foreach ($item in Get-ChildItem -Force -LiteralPath $SourceDir) {
		if (Should-SkipPath $item) {
			continue
		}

		$targetPath = Join-Path $TargetDir $item.Name

		if ($item.PSIsContainer) {
			Copy-FilteredProject $item.FullName $targetPath
		}
		else {
			Copy-Item -LiteralPath $item.FullName -Destination $targetPath -Force
		}
	}
}

function Sync-Project {
	$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("sort_vision_deploy_" + [guid]::NewGuid().ToString("N"))

	try {
		Write-Info "Preparing deploy staging directory: $stageRoot"
		Copy-FilteredProject $ProjectRoot $stageRoot

		New-RemoteProjectDir

		$destination = "$($script:SshUser)@$($script:OrangePiIp):$($script:RemoteProjectDir)/"
		Write-Info "Syncing project to: $destination"
		& scp -P $script:SshPort -r (Join-Path $stageRoot ".") $destination

		if ($LASTEXITCODE -ne 0) {
			throw "Sync failed. Check scp, SSH connection, and remote directory permissions."
		}

		$remoteDir = Shell-Quote $script:RemoteProjectDir
		Invoke-Remote "cd $remoteDir && chmod +x run.sh && mkdir -p logs" "Remote post-sync setup failed. Check run.sh and directory permissions."
		Write-Info "Sync complete."
	}
	finally {
		if (Test-Path -LiteralPath $stageRoot) {
			Remove-Item -LiteralPath $stageRoot -Recurse -Force
		}
	}
}

function Stop-RemoteProgram {
	$remoteDir = Shell-Quote $script:RemoteProjectDir
	$pidFile = Shell-Quote $PidFileName

	$command = @"
cd $remoteDir 2>/dev/null || { echo 'Remote project directory does not exist. Nothing to stop.'; exit 0; }
if [ ! -f $pidFile ]; then echo 'No vision.pid found. Nothing to stop.'; exit 0; fi
pid=`$(cat $pidFile)
case "`$pid" in
	''|*[!0-9]*) echo 'vision.pid does not contain a valid PID. Refusing to stop.'; exit 1 ;;
esac
if ! kill -0 "`$pid" 2>/dev/null; then
	echo 'Recorded PID is not running. Removing stale vision.pid.'
	rm -f $pidFile
	exit 0
fi
cwd=`$(readlink "/proc/`$pid/cwd" 2>/dev/null || true)
cmd=`$(tr '\0' ' ' < "/proc/`$pid/cmdline" 2>/dev/null || true)
expected=`$(pwd -P)
if [ "`$cwd" != "`$expected" ]; then
	echo "PID `$pid is not running from this project directory. Refusing to stop."
	exit 1
fi
case "`$cmd" in
	*main.py*) ;;
	*) echo "PID `$pid is not this project's main.py process. Refusing to stop."; exit 1 ;;
esac
kill "`$pid"
for i in 1 2 3 4 5 6 7 8 9 10; do
	if ! kill -0 "`$pid" 2>/dev/null; then
		rm -f $pidFile
		echo "Stopped PID `$pid."
		exit 0
	fi
	sleep 1
done
echo "Stop signal was sent, but PID `$pid is still running. Please check Orange Pi manually."
exit 1
"@

	Invoke-Remote $command "Failed to stop remote program."
}

function Start-RemoteProgram {
	$remoteDir = Shell-Quote $script:RemoteProjectDir
	$pidFile = Shell-Quote $PidFileName

	$command = @"
cd $remoteDir || { echo 'Remote project directory does not exist.'; exit 1; }
if [ ! -f main.py ]; then echo 'main.py is missing. Cannot start.'; exit 1; fi
if [ ! -f run.sh ]; then echo 'run.sh is missing. Cannot start.'; exit 1; fi
chmod +x run.sh
mkdir -p logs
if [ -f $pidFile ]; then
	oldpid=`$(cat $pidFile)
	case "`$oldpid" in
		''|*[!0-9]*) oldpid='' ;;
	esac
	if [ -n "`$oldpid" ] && kill -0 "`$oldpid" 2>/dev/null; then
		echo "Program already seems to be running. PID=`$oldpid. Use -Restart to restart."
		exit 0
	fi
fi
log_file="logs/vision_`$(date +%Y%m%d_%H%M%S).log"
nohup ./run.sh >> "`$log_file" 2>&1 < /dev/null &
pid=`$!
echo "`$pid" > $pidFile
sleep 1
if kill -0 "`$pid" 2>/dev/null; then
	echo "Started vision program. PID=`$pid, log=`$log_file"
	exit 0
fi
echo "Remote program failed to start. Check log: `$log_file"
exit 1
"@

	Invoke-Remote $command "Remote run failed. Check logs under the Orange Pi project directory."
}

try {
	Require-Command "ssh"
	Require-Command "scp"
	Require-Config

	if ($Run -and $Stop) {
		throw "-Run and -Stop cannot be used together. Use -Restart to stop, sync, and start."
	}

	Test-RemoteConnection

	if ($Stop) {
		Stop-RemoteProgram
		exit 0
	}

	if ($Restart) {
		Stop-RemoteProgram
		Sync-Project
		Start-RemoteProgram
		exit 0
	}

	Sync-Project

	if ($Run -or $script:RunAfterSync) {
		Start-RemoteProgram
	}
	else {
		Write-Info "Sync only. To sync and run, use: .\deploy.ps1 -Run"
	}
}
catch {
	Write-Fail $_.Exception.Message
	exit 1
}
