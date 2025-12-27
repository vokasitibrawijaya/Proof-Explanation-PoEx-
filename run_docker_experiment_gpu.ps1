# PowerShell script to run FedXChain with GPU support

$ErrorActionPreference = "Stop"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "FedXChain Distributed Experiment with Docker + GPU (PyTorch)" -ForegroundColor Cyan
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

# Check NVIDIA GPU
Write-Host "Checking NVIDIA GPU support..." -ForegroundColor Yellow
try {
    $gpuInfo = docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NVIDIA GPU detected and accessible" -ForegroundColor Green
    } else {
        Write-Host "⚠ WARNING: GPU not detected. Install NVIDIA Container Toolkit:" -ForegroundColor Yellow
        Write-Host "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" -ForegroundColor Yellow
        Write-Host ""
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne "y") {
            exit 1
        }
    }
} catch {
    Write-Host "⚠ WARNING: Could not check GPU. Continuing..." -ForegroundColor Yellow
}

Write-Host ""

# Function to cleanup
function Cleanup {
    Write-Host ""
    Write-Host "Cleaning up containers..." -ForegroundColor Yellow
    docker-compose -f docker-compose.gpu.yml down -v
    Write-Host "✓ Cleanup complete" -ForegroundColor Green
}

# Register cleanup on exit
trap { Cleanup }

# Step 1: Build images
Write-Host "Step 1: Building Docker images (GPU version)..." -ForegroundColor Yellow
docker-compose -f docker-compose.gpu.yml build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to build images" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Images built" -ForegroundColor Green
Write-Host ""

# Step 2: Start blockchain
Write-Host "Step 2: Starting blockchain..." -ForegroundColor Yellow
docker-compose -f docker-compose.gpu.yml up -d blockchain
Write-Host "⏳ Waiting for blockchain to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 20
Write-Host "✓ Blockchain started" -ForegroundColor Green
Write-Host ""

# Step 3: Deploy smart contract
Write-Host "Step 3: Deploying smart contract..." -ForegroundColor Yellow
docker-compose -f docker-compose.gpu.yml up deployer
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Deployer exited with warnings (this is normal)" -ForegroundColor Yellow
}
Write-Host "✓ Contract deployed" -ForegroundColor Green
Write-Host ""

# Step 4: Start aggregator
Write-Host "Step 4: Starting aggregator..." -ForegroundColor Yellow
docker-compose -f docker-compose.gpu.yml up -d aggregator
Start-Sleep -Seconds 5
Write-Host "✓ Aggregator started" -ForegroundColor Green
Write-Host ""

# Step 5: Start GPU clients
Write-Host "Step 5: Starting GPU clients..." -ForegroundColor Yellow
docker-compose -f docker-compose.gpu.yml up -d client_gpu_0 client_gpu_1 client_gpu_2 client_gpu_3 client_gpu_4
Write-Host "✓ Clients started" -ForegroundColor Green
Write-Host ""

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "FedXChain GPU Experiment Running!" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor logs:" -ForegroundColor Yellow
Write-Host "  docker logs fedxchain-client-gpu-0 -f" -ForegroundColor White
Write-Host "  docker logs fedxchain-aggregator-gpu -f" -ForegroundColor White
Write-Host ""
Write-Host "Check status:" -ForegroundColor Yellow
Write-Host "  docker ps" -ForegroundColor White
Write-Host ""
Write-Host "Stop experiment:" -ForegroundColor Yellow
Write-Host "  docker-compose -f docker-compose.gpu.yml down" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray

# Follow logs
docker logs fedxchain-client-gpu-0 -f
