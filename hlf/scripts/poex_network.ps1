Param(
  [string]$ChannelName = "poexchannel",
  [string]$ChaincodeName = "poex",
  [string]$ChaincodeVersion = "1.0",
  [int]$Sequence = 1
)

$ErrorActionPreference = "Stop"

function Invoke-DockerTools {
  param([string]$Cmd)
  docker run --rm -v "${PWD}:/fabric" -w /fabric hyperledger/fabric-tools:2.5 bash -lc $Cmd
}

Write-Host "== PoEx Fabric network helper ==" -ForegroundColor Cyan

if (-not (Test-Path "./crypto-config")) {
  Write-Host "Generating crypto material (cryptogen)..." -ForegroundColor Yellow
  Invoke-DockerTools "cryptogen generate --config=/fabric/config/crypto-config.yaml --output=/fabric/crypto-config"
}

if (-not (Test-Path "./channel-artifacts")) {
  New-Item -ItemType Directory -Force -Path "./channel-artifacts" | Out-Null
}

$ChannelBlock = "./channel-artifacts/${ChannelName}.block"
if (-not (Test-Path $ChannelBlock)) {
  Write-Host "Generating application channel config block (configtxgen outputBlock)..." -ForegroundColor Yellow
  Invoke-DockerTools "export FABRIC_CFG_PATH=/fabric/config; configtxgen -profile PoExChannel -channelID ${ChannelName} -outputBlock /fabric/channel-artifacts/${ChannelName}.block"
}

Write-Host "Starting Fabric containers..." -ForegroundColor Yellow
& docker compose -f docker-compose-hlf.yml up -d

Write-Host "Waiting a bit for containers..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "Joining orderer to channel (channel participation API)..." -ForegroundColor Yellow
& docker exec poex-cli bash -lc "osnadmin channel join --channelID ${ChannelName} --config-block /fabric/channel-artifacts/${ChannelName}.block -o orderer.example.com:7053 --ca-file /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt --client-cert /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/server.crt --client-key /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/server.key"

Write-Host "Peer channel join..." -ForegroundColor Yellow
& docker exec poex-cli bash -lc "peer channel join -b /fabric/channel-artifacts/${ChannelName}.block"

Write-Host "NOTE: Chaincode deployment is next (package/install/approve/commit)." -ForegroundColor Cyan
Write-Host "This script currently scaffolds network + channel only." -ForegroundColor Cyan
