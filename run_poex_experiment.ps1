Param(
  [string]$ChannelName = "poexchannel",
  [string]$ChaincodeName = "poex",
  [string]$RunId = "1",
  [int]$MaxRounds = 10,
  [string]$AggMethod = "fedavg",
  [int]$PoExEnabled = 1,
  [double]$PoExThreshold = 0.5,
  [string]$AttackType = "none",
  [string]$MaliciousClients = "",
  [double]$MaliciousRatio = 0.0,
  [double]$AttackSigma = 0.1,
  [int]$Reset = 1
)

$ErrorActionPreference = "Stop"

Write-Host "== PoEx Experiment Runner (sequential clients) ==" -ForegroundColor Cyan

if ($Reset -ne 0) {
  Write-Host "Resetting PoEx stack (including volumes)..." -ForegroundColor Yellow
  & docker compose -f docker-compose-poex.yml down -v --remove-orphans | Out-Host
}

function Invoke-DockerTools {
  param([string]$Cmd)
  docker run --rm -v "${PWD}\hlf:/fabric" -w /fabric hyperledger/fabric-tools:2.5 bash -lc $Cmd
}

# 1) Generate crypto + genesis + channel tx under ./hlf
Write-Host "[1/5] Preparing Fabric artifacts (crypto-config, genesis, channel tx)..." -ForegroundColor Yellow
if (-not (Test-Path "./hlf/crypto-config")) {
  Invoke-DockerTools "cryptogen generate --config=/fabric/config/crypto-config.yaml --output=/fabric/crypto-config"
}

if (-not (Test-Path "./hlf/channel-artifacts")) {
  New-Item -ItemType Directory -Force -Path "./hlf/channel-artifacts" | Out-Null
}

$GenesisBlockHost = "./hlf/channel-artifacts/genesis.block"
if (-not (Test-Path $GenesisBlockHost)) {
  Invoke-DockerTools "export FABRIC_CFG_PATH=/fabric/config; configtxgen -profile PoExOrdererGenesis -channelID system-channel -outputBlock /fabric/channel-artifacts/genesis.block"
}

$ChannelTxHost = "./hlf/channel-artifacts/${ChannelName}.tx"
if (-not (Test-Path $ChannelTxHost)) {
  Invoke-DockerTools "export FABRIC_CFG_PATH=/fabric/config; configtxgen -profile PoExChannel -outputCreateChannelTx /fabric/channel-artifacts/${ChannelName}.tx -channelID ${ChannelName}"
}

# 2) Start Fabric base services first
Write-Host "[2/5] Starting Fabric base services (orderer/peer/cli)..." -ForegroundColor Yellow
$env:CHANNEL_NAME = $ChannelName
$env:CHAINCODE_NAME = $ChaincodeName
$env:RUN_ID = $RunId
$env:MAX_ROUNDS = "$MaxRounds"
$env:AGG_METHOD = $AggMethod
$env:POEX_ENABLED = "$PoExEnabled"
$env:POEX_THRESHOLD = "$PoExThreshold"
$env:ATTACK_TYPE = $AttackType
$env:MALICIOUS_CLIENTS = $MaliciousClients
$env:MALICIOUS_RATIO = "$MaliciousRatio"
$env:ATTACK_SIGMA = "$AttackSigma"

& docker compose -f docker-compose-poex.yml up -d orderer.example.com peer0.org1.example.com cli

Write-Host "[3/5] Joining orderer+peer to channel..." -ForegroundColor Yellow
Start-Sleep -Seconds 10
$oldEap2 = $ErrorActionPreference
$ErrorActionPreference = "Continue"

$channelListText = (& docker exec poex-cli bash -lc "peer channel list" 2>&1 | Out-String)
if ($channelListText -match "(?m)^\s*${ChannelName}\s*$") {
  $ErrorActionPreference = $oldEap2
  Write-Host "Channel ${ChannelName} already exists on peer; skipping create/join." -ForegroundColor Yellow
} else {

$createText = (& docker exec poex-cli bash -lc "peer channel create -o orderer.example.com:7050 --tls --cafile /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt -c ${ChannelName} -f /fabric/channel-artifacts/${ChannelName}.tx --outputBlock /fabric/channel-artifacts/${ChannelName}.block; echo __RC:`$?" 2>&1 | Out-String)
$createText | Out-Host

$joinText = (& docker exec poex-cli bash -lc "peer channel join -b /fabric/channel-artifacts/${ChannelName}.block; echo __RC:`$?" 2>&1 | Out-String)
$ErrorActionPreference = $oldEap2
$joinText | Out-Host

$createOk = ($createText -match "__RC:0") -or ($createText -match "Received block: 0")
if (-not $createOk) {
  throw "Channel create failed for ${ChannelName}."
}

$joinOk = ($joinText -match "__RC:0") -or ($joinText -match "Successfully submitted proposal") -or ($joinText -match "ledger \[${ChannelName}\] already exists") -or ($joinText -match "already exists with state")
if (-not $joinOk) {
  throw "Peer failed to join channel ${ChannelName}."
}

}

Write-Host "[4/5] Deploying and initializing PoEx chaincode..." -ForegroundColor Yellow
$Label = "${ChaincodeName}_1.0"
$Package = "/fabric/channel-artifacts/${Label}.tar.gz"
$ccaasSrc = "/fabric/chaincode/poex/ccaas"

& docker exec poex-cli bash -lc "set -euo pipefail; rm -rf /tmp/ccaas_pkg; mkdir -p /tmp/ccaas_pkg; cp ${ccaasSrc}/metadata.json /tmp/ccaas_pkg/metadata.json; cd ${ccaasSrc}; tar -czf /tmp/ccaas_pkg/code.tar.gz connection.json; cd /tmp/ccaas_pkg; tar -czf ${Package} metadata.json code.tar.gz" | Out-Host

$installOut = & docker exec poex-cli bash -lc "peer lifecycle chaincode install ${Package}"
$installOut | Out-Host

$queryOut = & docker exec poex-cli bash -lc "peer lifecycle chaincode queryinstalled"
$queryOut | Out-Host
$packageId = ($queryOut | Select-String -Pattern "Package ID: (.*), Label: ${Label}" | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
if (-not $packageId) { throw "Failed to parse packageId for ${Label}" }

Write-Host "Starting external PoEx chaincode service (ccaas)..." -ForegroundColor Yellow
$env:CHAINCODE_ID = $packageId
& docker compose -f docker-compose-poex.yml up -d --build poex-chaincode | Out-Host

& docker exec poex-cli bash -lc "peer lifecycle chaincode approveformyorg -o orderer.example.com:7050 --tls --cafile /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt -C ${ChannelName} -n ${ChaincodeName} -v 1.0 --package-id ${packageId} --sequence 1 --signature-policy \"OR('Org1MSP.member')\"" | Out-Host
& docker exec poex-cli bash -lc "peer lifecycle chaincode commit -o orderer.example.com:7050 --tls --cafile /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt -C ${ChannelName} -n ${ChaincodeName} -v 1.0 --sequence 1 --signature-policy \"OR('Org1MSP.member')\" --peerAddresses peer0.org1.example.com:7051 --tlsRootCertFiles /fabric/crypto-config/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" | Out-Host

Write-Host "Starting PoEx gateway..." -ForegroundColor Yellow
& docker compose -f docker-compose-poex.yml up -d --build poex-gateway | Out-Host

Write-Host "Waiting for PoEx gateway..." -ForegroundColor Yellow
for ($i=0; $i -lt 60; $i++) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://localhost:8080/health"
    if ($r.StatusCode -eq 200) { break }
  } catch { }
  Start-Sleep -Seconds 2
}

Write-Host "Starting aggregator..." -ForegroundColor Yellow
& docker compose -f docker-compose-poex.yml up -d --build aggregator | Out-Host

Write-Host "Waiting for aggregator..." -ForegroundColor Yellow
for ($i=0; $i -lt 60; $i++) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://localhost:5001/health"
    if ($r.StatusCode -eq 200) { break }
  } catch { }
  Start-Sleep -Seconds 2
}

# 3) Run clients in parallel (background jobs)
Write-Host "[5/5] Running clients in parallel..." -ForegroundColor Yellow
$clients = @('client_0','client_1','client_2')

# Start all clients in background
$jobs = @()
foreach ($c in $clients) {
  Write-Host "Starting $c..." -ForegroundColor Cyan
  $job = Start-Job -ScriptBlock {
    param($composefile, $client)
    Set-Location $using:PWD
    & docker compose -f $composefile --profile clients run --rm $client
  } -ArgumentList "docker-compose-poex.yml", $c
  $jobs += $job
}

# Wait for all clients to finish
Write-Host "Waiting for all clients to complete..." -ForegroundColor Yellow
$jobs | Wait-Job | Out-Null

# Show job outputs
foreach ($job in $jobs) {
  Write-Host "`n--- Output from $($job.Name) ---" -ForegroundColor Gray
  Receive-Job $job | Out-Host
  Remove-Job $job
}

Write-Host "Done. Results written to ./results/poex_results.csv" -ForegroundColor Green
