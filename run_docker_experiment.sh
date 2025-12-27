#!/bin/bash
# Orchestration script to run FedXChain distributed experiment with Docker

set -e

echo "========================================================================"
echo "FedXChain Distributed Experiment with Docker + Blockchain"
echo "========================================================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Function to show logs
show_logs() {
    echo ""
    echo "========================================================================"
    echo "Container Logs"
    echo "========================================================================"
    docker-compose logs --tail=50 $1
}

# Function to cleanup
cleanup() {
    echo ""
    echo "Cleaning up containers..."
    docker-compose down -v
    echo "✓ Cleanup complete"
}

# Trap cleanup on exit
trap cleanup EXIT

# Parse arguments
DETACH=""
LOGS="aggregator"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detach)
            DETACH="-d"
            shift
            ;;
        --logs)
            LOGS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Build images
echo "Step 1: Building Docker images..."
docker-compose build
echo "✓ Images built"
echo ""

# Step 2: Start blockchain
echo "Step 2: Starting blockchain..."
docker-compose up -d blockchain
echo "⏳ Waiting for blockchain to be ready..."
sleep 10
echo "✓ Blockchain started"
echo ""

# Step 3: Deploy smart contract
echo "Step 3: Deploying smart contract..."
docker-compose up deployer
echo "✓ Contract deployed"
echo ""

# Step 4: Start aggregator
echo "Step 4: Starting aggregator server..."
docker-compose up -d aggregator
echo "⏳ Waiting for aggregator to initialize..."
sleep 15
echo "✓ Aggregator started"
echo ""

# Step 5: Start FL clients
echo "Step 5: Starting FL client containers..."
docker-compose up $DETACH client_0 client_1 client_2 client_3 client_4

if [ -z "$DETACH" ]; then
    echo ""
    echo "========================================================================"
    echo "Training Complete!"
    echo "========================================================================"
    echo ""
    echo "View logs:"
    echo "  docker-compose logs aggregator"
    echo "  docker-compose logs client_0"
    echo ""
    echo "Check status:"
    echo "  docker-compose ps"
    echo ""
fi

echo "========================================================================"
echo "Experiment Complete"
echo "========================================================================"
