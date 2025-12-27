# FedXChain dengan Docker + Blockchain

Panduan lengkap untuk menjalankan FedXChain dengan arsitektur distributed menggunakan Docker containers dan blockchain yang aktif.

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │  Blockchain  │◄─────┤   Deployer   │                   │
│  │   (Hardhat)  │      │   (Deploy    │                   │
│  │   Port 8545  │      │   Contract)  │                   │
│  └──────┬───────┘      └──────────────┘                   │
│         │                                                   │
│         │ ┌────────────────────────────────────┐          │
│         ├─┤        Aggregator Server           │          │
│         │ │  - Coordinates FL rounds           │          │
│         │ │  - Blockchain integration          │          │
│         │ │  - PoEx consensus                  │          │
│         │ │  Port 5000                         │          │
│         │ └────────────┬───────────────────────┘          │
│         │              │                                   │
│         │    ┌─────────┴─────────┐                        │
│         │    │                   │                        │
│  ┌──────▼────▼──┐  ┌──────────┐  ┌──────────┐           │
│  │  Client 0    │  │ Client 1 │  │ Client 2 │  ...      │
│  │  - Training  │  │          │  │          │           │
│  │  - SHAP      │  │          │  │          │           │
│  │  - Submit    │  │          │  │          │           │
│  └──────────────┘  └──────────┘  └──────────┘           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

### Windows
1. **Docker Desktop** untuk Windows
   - Download: https://www.docker.com/products/docker-desktop
   - Pastikan WSL2 enabled
   - Minimal 8GB RAM allocated

2. **PowerShell** (sudah terinstall di Windows)

### Linux/Mac
1. **Docker** dan **Docker Compose**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   
   # Mac (menggunakan Homebrew)
   brew install docker docker-compose
   ```

## 🚀 Quick Start

### Opsi 1: PowerShell (Windows)

```powershell
# Jalankan eksperimen lengkap
.\run_docker_experiment.ps1
```

### Opsi 2: Bash (Linux/Mac/WSL)

```bash
# Beri permission execute
chmod +x run_docker_experiment.sh

# Jalankan eksperimen
./run_docker_experiment.sh
```

### Opsi 3: Manual dengan Docker Compose

```bash
# 1. Build images
docker-compose build

# 2. Start blockchain
docker-compose up -d blockchain

# 3. Deploy contract
docker-compose up deployer

# 4. Start aggregator
docker-compose up -d aggregator

# 5. Start clients (akan menjalankan training)
docker-compose up client_0 client_1 client_2 client_3 client_4
```

## 📊 Monitoring Eksperimen

### Cek Status Containers

```bash
docker-compose ps
```

Output:
```
NAME                    STATUS          PORTS
fedxchain-blockchain    Up 2 minutes    0.0.0.0:8545->8545/tcp
fedxchain-aggregator    Up 1 minute     0.0.0.0:5000->5000/tcp
fedxchain-client-0      Up 30 seconds
fedxchain-client-1      Up 30 seconds
...
```

### Lihat Logs Real-time

```bash
# Logs aggregator
docker-compose logs -f aggregator

# Logs client tertentu
docker-compose logs -f client_0

# Logs semua services
docker-compose logs -f
```

### Cek Blockchain

```bash
# Test koneksi ke blockchain
curl http://localhost:8545 -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
```

### Cek Aggregator Status

```bash
# Via curl
curl http://localhost:5000/status

# Via browser
# Buka: http://localhost:5000/status
```

Output:
```json
{
  "blockchain_connected": true,
  "current_round": 3,
  "max_rounds": 10,
  "n_clients": 5,
  "updates_received": 2
}
```

## 🔧 Konfigurasi

### Ubah Jumlah Clients

Edit `docker-compose.yml`:
- Tambah/kurangi service `client_X`
- Update `N_CLIENTS` di aggregator environment

### Ubah Parameter Eksperimen

Edit `configs/experiment_config.yaml`:
```yaml
rounds: 20              # Jumlah training rounds
local_epochs: 1         # Epochs per round
shap_samples: 10        # Samples untuk SHAP computation
trust_alpha: 0.4        # Weight untuk accuracy
trust_beta: 0.3         # Weight untuk XAI
trust_gamma: 0.3        # Weight untuk consistency
```

### Tambah Malicious Nodes

Edit environment di `docker-compose.yml` untuk client tertentu:
```yaml
client_0:
  environment:
    - NODE_TYPE=client
    - NODE_ID=0
    - IS_MALICIOUS=true          # ← Tambahkan
    - ATTACK_TYPE=label_flip      # ← Tambahkan
    - ATTACK_INTENSITY=0.3        # ← Tambahkan
```

## 🛠️ Troubleshooting

### Error: "Cannot connect to Docker daemon"
```bash
# Windows: Start Docker Desktop
# Linux: Start Docker service
sudo systemctl start docker
```

### Error: "Port 8545 already in use"
```bash
# Cek process yang menggunakan port
netstat -ano | findstr :8545    # Windows
lsof -i :8545                   # Linux/Mac

# Stop containers existing
docker-compose down
```

### Container Crash atau Exit

```bash
# Lihat logs untuk error details
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]

# Rebuild jika ada perubahan code
docker-compose up --build [service-name]
```

### Blockchain tidak respond

```bash
# Restart blockchain
docker-compose restart blockchain

# Cek health
docker inspect fedxchain-blockchain | grep Health -A 10
```

## 📈 Hasil Eksperimen

### Lokasi Output

Hasil eksperimen akan tersimpan di:
- `results/` - Metrics per round
- `logs/` - Training logs
- `hardhat/deployments/` - Contract deployment info

### Export Results

```bash
# Copy dari container ke host
docker cp fedxchain-aggregator:/app/results ./results_backup

# View dalam container
docker-compose exec aggregator ls -la /app/results
```

## 🧹 Cleanup

### Stop Containers

```bash
docker-compose stop
```

### Remove Containers & Networks

```bash
docker-compose down
```

### Remove Everything (termasuk volumes)

```bash
docker-compose down -v
```

### Remove Images

```bash
docker-compose down --rmi all
```

## 🔬 Eksperimen Lanjutan

### Run dengan Custom Config

1. Buat config baru: `configs/custom_config.yaml`
2. Mount di `docker-compose.yml`:
   ```yaml
   aggregator:
     volumes:
       - ./configs/custom_config.yaml:/app/configs/config.yaml
   ```

### Scaling Clients

```bash
# Scale ke 10 clients
docker-compose up --scale client=10
```

### Multiple Experiments

```bash
# Run eksperimen 1
docker-compose -p exp1 up

# Run eksperimen 2 (parallel)
docker-compose -p exp2 up
```

## 📝 Architecture Components

### 1. Blockchain Container
- **Image**: Node.js 18 + Hardhat
- **Purpose**: Local Ethereum blockchain untuk smart contracts
- **Port**: 8545
- **Network**: fedxchain-network

### 2. Deployer Container
- **Purpose**: Deploy smart contract ke blockchain
- **Lifecycle**: Runs once, exits after deployment
- **Output**: Contract address & ABI

### 3. Aggregator Container
- **Image**: Python 3.10 + Flask
- **Purpose**: 
  - Coordinate FL rounds
  - Aggregate model updates
  - Implement PoEx consensus
  - Blockchain integration
- **Port**: 5000
- **API Endpoints**:
  - `GET /health` - Health check
  - `POST /register` - Register client
  - `GET /get_model` - Get global model
  - `POST /submit_update` - Submit local update
  - `GET /status` - Get current status

### 4. Client Containers (x5)
- **Image**: Python 3.10 + scikit-learn + SHAP
- **Purpose**:
  - Local training
  - SHAP computation
  - Submit updates to aggregator
- **Environment**:
  - `NODE_ID`: Unique client identifier
  - `AGGREGATOR_URL`: Aggregator endpoint
  - `BLOCKCHAIN_RPC`: Blockchain endpoint

## 🎯 Comparison: Simulasi vs Docker

| Aspect | Simulasi (Local) | Docker (Distributed) |
|--------|------------------|---------------------|
| **Deployment** | Single Python process | Multiple containers |
| **Blockchain** | Mocked/Disabled | Real Hardhat node |
| **Network** | In-memory | TCP/IP via Docker network |
| **Isolation** | Single environment | Isolated containers |
| **Scalability** | Limited by RAM | Limited by Docker resources |
| **Realism** | Algorithm validation | Production-like setup |
| **Complexity** | Low | Medium-High |
| **IEEE Paper** | ✓ Sufficient | ✓✓ More comprehensive |

## 💡 Best Practices

1. **Resource Allocation**
   - Minimum: 8GB RAM, 4 CPU cores
   - Recommended: 16GB RAM, 8 CPU cores

2. **Development Workflow**
   - Test dengan simulasi lokal dulu
   - Validate dengan Docker untuk paper
   - Use Docker untuk demo dan deployment

3. **Logging**
   - Gunakan `-f` untuk real-time logs
   - Save logs: `docker-compose logs > experiment.log`

4. **Debugging**
   - Exec into container: `docker-compose exec aggregator bash`
   - Interactive Python: `docker-compose exec aggregator python`

## 📚 References

- Docker Compose: https://docs.docker.com/compose/
- Hardhat: https://hardhat.org/
- Flask: https://flask.palletsprojects.com/
- SHAP: https://shap.readthedocs.io/

---

**Status**: ✅ Ready for Deployment  
**Last Updated**: December 25, 2025  
**Tested On**: Docker Desktop 4.x, Docker Compose 2.x
