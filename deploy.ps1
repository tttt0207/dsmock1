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

function Get-SshArgumentText {
	param([string]$RemoteCommand)

	$target = "$($script:SshUser)@$($script:OrangePiIp)"
	$args = @(
		"-p", "$($script:SshPort)",
		$target,
		$RemoteCommand
	)

	return ($args | ForEach-Object { Quote-WindowsArgument $_ }) -join " "
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

function Invoke-RemoteBash {
	param(
		[string]$BashScript,
		[string[]]$Arguments = @(),
		[string]$FailureMessage
	)

	# Encode the remote script locally and decode it on Orange Pi. This avoids
	# BOM or terminal encoding artifacts being interpreted as bash commands.
	$normalizedScript = ($BashScript -replace "`r`n", "`n" -replace "`r", "`n")
	$normalizedScript = $normalizedScript.TrimStart([char]0xFEFF)

	if (-not $normalizedScript.EndsWith("`n")) {
		$normalizedScript += "`n"
	}

	$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
	$scriptBytes = $utf8NoBom.GetBytes($normalizedScript)
	$scriptBase64 = [System.Convert]::ToBase64String($scriptBytes)
	$remoteCommand = "printf %s " + (Quote-BashArgument $scriptBase64) + " | base64 -d | bash -s"

	if ($Arguments.Count -gt 0) {
		$remoteCommand += " --"
		foreach ($arg in $Arguments) {
			$remoteCommand += " " + (Quote-BashArgument $arg)
		}
	}

	$psi = [System.Diagnostics.ProcessStartInfo]::new()
	$psi.FileName = "ssh"
	$psi.Arguments = Get-SshArgumentText -RemoteCommand $remoteCommand
	$psi.RedirectStandardInput = $false
	$psi.RedirectStandardOutput = $true
	$psi.RedirectStandardError = $true
	$psi.UseShellExecute = $false
	$psi.CreateNoWindow = $true

	$process = [System.Diagnostics.Process]::Start($psi)
	$stdout = $process.StandardOutput.ReadToEnd()
	$stderr = $process.StandardError.ReadToEnd()
	$process.WaitForExit()

	if (-not [string]::IsNullOrWhiteSpace($stdout)) {
		Write-Host $stdout.TrimEnd()
	}

	if (-not [string]::IsNullOrWhiteSpace($stderr)) {
		Write-Host $stderr.TrimEnd()
	}

	if ($process.ExitCode -ne 0) {
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
		"deploy_config.ps1"
	)

	if ($excludedNames -contains $Item.Name) {
		return $true
	}

	if (-not $Item.PSIsContainer) {
		$extension = $Item.Extension.ToLowerInvariant()

		if ($extension -eq ".py") {
			foreach ($char in $Item.Name.ToCharArray()) {
				if ([int][char]$char -gt 127) {
					return $true
				}
			}
		}

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

			# Normalize Linux shell scripts to UTF-8 without BOM and LF.
			if ($item.Extension.ToLowerInvariant() -eq ".sh") {
				$text = [System.IO.File]::ReadAllText($targetPath)
				$text = $text.TrimStart([char]0xFEFF)
				$text = $text -replace "`r`n", "`n" -replace "`r", "`n"

				$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
				[System.IO.File]::WriteAllText($targetPath, $text, $utf8NoBom)
			}
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
	$bashScript = @'
PROJECT_DIR="$1"
PID_FILE="$PROJECT_DIR/vision.pid"

if [ ! -f "$PID_FILE" ]; then
	echo "Program is not running, or PID file does not exist."
	exit 0
fi

pid=$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)

if [ -z "$pid" ]; then
	echo "Warning: vision.pid is empty. Nothing was stopped."
	exit 0
fi

case "$pid" in
	*[!0-9]*)
		echo "Warning: vision.pid is not numeric. Nothing was stopped."
		exit 0
		;;
esac

if [ ! -d "/proc/$pid" ]; then
	echo "PID file is stale; process does not exist."
	rm -f "$PID_FILE"
	exit 0
fi

cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)

case "$cmd" in
	*main.py*|*"$PROJECT_DIR"*) ;;
	*)
		echo "Safety warning: PID $pid does not look like this project. Refusing to stop."
		echo "Command line: $cmd"
		exit 1
		;;
esac

echo "Stopping project process PID=$pid"
kill "$pid"

for i in 1 2 3 4 5 6 7 8 9 10; do
	if [ ! -d "/proc/$pid" ]; then
		rm -f "$PID_FILE"
		echo "Stopped PID $pid."
		exit 0
	fi
	sleep 0.5
done

echo "Stop signal was sent, but PID $pid is still running. Please check Orange Pi manually."
exit 1
'@

	Invoke-RemoteBash -BashScript $bashScript -Arguments @($script:RemoteProjectDir) -FailureMessage "Failed to stop remote program."
}

function Start-RemoteProgram {
	$bashScript = @'
PROJECT_DIR="$1"
PID_FILE="$PROJECT_DIR/vision.pid"

cd "$PROJECT_DIR" || { echo "Remote project directory does not exist: $PROJECT_DIR"; exit 1; }

if [ ! -f main.py ]; then
	echo "main.py is missing. Cannot start."
	exit 1
fi

if [ ! -f run.sh ]; then
	echo "run.sh is missing. Cannot start."
	exit 1
fi

chmod +x run.sh
mkdir -p logs

if [ -f "$PID_FILE" ]; then
	oldpid=$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)
	case "$oldpid" in
		''|*[!0-9]*) oldpid='' ;;
	esac

	if [ -n "$oldpid" ] && [ -d "/proc/$oldpid" ]; then
		echo "Program already seems to be running. PID=$oldpid. Use -Restart to restart."
		exit 0
	fi

	rm -f "$PID_FILE"
fi

log_file="logs/vision_$(date +%Y%m%d_%H%M%S).log"
nohup ./run.sh >> "$log_file" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 1

if [ -d "/proc/$pid" ]; then
	echo "Started vision program. PID=$pid, log=$log_file"
	exit 0
fi

echo "Remote program failed to start. Check log: $log_file"
exit 1
'@

	Invoke-RemoteBash -BashScript $bashScript -Arguments @($script:RemoteProjectDir) -FailureMessage "Remote run failed. Check logs under the Orange Pi project directory."
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
