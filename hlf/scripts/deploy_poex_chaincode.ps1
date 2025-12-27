Param(
  [string]$ChannelName = "poexchannel",
  [string]$ChaincodeName = "poex",
  [string]$ChaincodeVersion = "1.0",
  [int]$Sequence = 1
)

$ErrorActionPreference = "Stop"

$Label = "${ChaincodeName}_${ChaincodeVersion}"
$Package = "/fabric/channel-artifacts/${Label}.tar.gz"

Write-Host "== Deploying PoEx chaincode ==" -ForegroundColor Cyan

# Package
Write-Host "Packaging chaincode..." -ForegroundColor Yellow
& docker exec poex-cli bash -lc "mkdir -p /fabric/channel-artifacts; peer lifecycle chaincode package ${Package} --path /fabric/chaincode/poex/javascript --lang node --label ${Label}"

# Install
Write-Host "Installing chaincode on peer..." -ForegroundColor Yellow
$installOut = & docker exec poex-cli bash -lc "peer lifecycle chaincode install ${Package}"
Write-Host $installOut

# Query installed -> extract package ID
Write-Host "Query installed chaincodes..." -ForegroundColor Yellow
$queryOut = & docker exec poex-cli bash -lc "peer lifecycle chaincode queryinstalled"
Write-Host $queryOut

$packageId = ($queryOut | Select-String -Pattern "Package ID: (.*), Label: ${Label}" | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
if (-not $packageId) {
  throw "Failed to parse packageId for label ${Label}."
}
Write-Host "Package ID: $packageId" -ForegroundColor Green

# Approve for Org1
Write-Host "Approving chaincode definition for Org1..." -ForegroundColor Yellow
& docker exec poex-cli bash -lc "peer lifecycle chaincode approveformyorg -o orderer.example.com:7050 --tls --cafile /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt -C ${ChannelName} -n ${ChaincodeName} -v ${ChaincodeVersion} --package-id ${packageId} --sequence ${Sequence} --init-required"

# Commit
Write-Host "Committing chaincode definition..." -ForegroundColor Yellow
& docker exec poex-cli bash -lc "peer lifecycle chaincode commit -o orderer.example.com:7050 --tls --cafile /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt -C ${ChannelName} -n ${ChaincodeName} -v ${ChaincodeVersion} --sequence ${Sequence} --init-required --peerAddresses peer0.org1.example.com:7051 --tlsRootCertFiles /fabric/crypto-config/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"

# Init
Write-Host "Initializing chaincode (InitLedger)..." -ForegroundColor Yellow
& docker exec poex-cli bash -lc "peer chaincode invoke -o orderer.example.com:7050 --tls --cafile /fabric/crypto-config/ordererOrganizations/example.com/orderers/orderer.example.com/tls/ca.crt -C ${ChannelName} -n ${ChaincodeName} --isInit -c '{\"Args\":[\"InitLedger\"]}' --peerAddresses peer0.org1.example.com:7051 --tlsRootCertFiles /fabric/crypto-config/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"

Write-Host "Done." -ForegroundColor Green
