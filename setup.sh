#!/bin/bash
# Setup script for FedXChain experiment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════════════════════"
echo "  FedXChain Experiment Setup"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p results logs hardhat/contracts hardhat/scripts
echo "✓ Directories created"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "To run the experiment:"
echo "  1. Without blockchain (simple mode):"
echo "     ./run_experiment.sh"
echo ""
echo "  2. With Docker (includes blockchain):"
echo "     docker-compose up"
echo ""
echo "  3. With local blockchain:"
echo "     ./run_with_blockchain.sh"
echo ""
