# Copy this file to deploy_config.ps1, then edit the values for your Orange Pi.
# Do not save SSH passwords in this file. Use SSH key login or enter the password when prompted.

$OrangePiIp = "192.168.1.100"
$SshUser = "orangepi"
$RemoteProjectDir = "/home/orangepi/sort_vision"
$SshPort = 22
$RunAfterSync = $false
