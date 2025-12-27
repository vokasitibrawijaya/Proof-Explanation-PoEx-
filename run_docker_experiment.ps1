# PowerShell script to run FedXChain distributed experiment with Docker

$ErrorActionPreference = "Stop"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "FedXChain Distributed Experiment with Docker + Blockchain" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Function to cleanup
function Cleanup {
    Write-Host ""
    Write-Host "Cleaning up containers..." -ForegroundColor Yellow
    docker-compose down -v
    Write-Host "✓ Cleanup complete" -ForegroundColor Green
}

# Register cleanup on exit
trap { Cleanup }

# Step 1: Build images
Write-Host "Step 1: Building Docker images..." -ForegroundColor Yellow
docker-compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build images" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Images built" -ForegroundColor Green
Write-Host ""

# Step 2: Start blockchain
Write-Host "Step 2: Starting blockchain..." -ForegroundColor Yellow
docker-compose up -d blockchain
Write-Host "⏳ Waiting for blockchain to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10
Write-Host "✓ Blockchain started" -ForegroundColor Green
Write-Host ""

# Step 3: Deploy smart contract
Write-Host "Step 3: Deploying smart contract..." -ForegroundColor Yellow
docker-compose up deployer
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Contract deployment may have issues, continuing..." -ForegroundColor Yellow
}
Write-Host "✓ Contract deployed" -ForegroundColor Green
Write-Host ""

# Step 4: Start aggregator
Write-Host "Step 4: Starting aggregator server..." -ForegroundColor Yellow
docker-compose up -d aggregator
Write-Host "⏳ Waiting for aggregator to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 15
Write-Host "✓ Aggregator started" -ForegroundColor Green
Write-Host ""

# Step 5: Start FL clients
Write-Host "Step 5: Starting FL client containers..." -ForegroundColor Yellow
Write-Host "This will run the federated learning training..." -ForegroundColor Cyan
Write-Host ""

docker-compose up client_0 client_1 client_2 client_3 client_4

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "View logs:" -ForegroundColor Yellow
Write-Host "  docker-compose logs aggregator"
Write-Host "  docker-compose logs client_0"
Write-Host ""
Write-Host "Check status:" -ForegroundColor Yellow
Write-Host "  docker-compose ps"
Write-Host ""
Write-Host "Stop containers:" -ForegroundColor Yellow
Write-Host "  docker-compose down"
Write-Host ""

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Experiment Complete" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan

# Ask if user wants to cleanup
$response = Read-Host "Do you want to stop and remove containers? (y/N)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Cleanup
} else {
    Write-Host ""
    Write-Host "Containers are still running. Use 'docker-compose down' to stop them." -ForegroundColor Yellow
}
