# FedXChain dengan GPU Support (PyTorch + CUDA)

Panduan untuk menjalankan FedXChain dengan GPU acceleration menggunakan PyTorch dan NVIDIA CUDA.

## 📋 Prerequisites

### 1. NVIDIA GPU
- NVIDIA GPU dengan CUDA support (GTX 1060 atau lebih tinggi)
- Driver NVIDIA terbaru ([Download](https://www.nvidia.com/download/index.aspx))

### 2. NVIDIA Container Toolkit (untuk Docker)

**Windows:**
```powershell
# Install WSL2 terlebih dahulu jika belum
wsl --install

# Install NVIDIA Container Toolkit di dalam WSL2
wsl
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Linux:**
```bash
# Tambah repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verifikasi GPU Support

```bash
# Test GPU di Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Jika berhasil, Anda akan melihat output GPU info.

## 🚀 Menjalankan Eksperimen GPU

### Windows PowerShell
```powershell
.\run_docker_experiment_gpu.ps1
```

### Linux/Mac
```bash
chmod +x run_docker_experiment_gpu.sh
./run_docker_experiment_gpu.sh
```

### Manual dengan Docker Compose
```bash
# 1. Build GPU images
docker-compose -f docker-compose.gpu.yml build

# 2. Start blockchain
docker-compose -f docker-compose.gpu.yml up -d blockchain

# 3. Wait for blockchain
sleep 20

# 4. Deploy contract
docker-compose -f docker-compose.gpu.yml up deployer

# 5. Start aggregator
docker-compose -f docker-compose.gpu.yml up -d aggregator

# 6. Start GPU clients
docker-compose -f docker-compose.gpu.yml up -d client_gpu_0 client_gpu_1 client_gpu_2 client_gpu_3 client_gpu_4
```

## 📊 Monitoring

### Lihat Logs GPU Client
```bash
docker logs fedxchain-client-gpu-0 -f
```

### Cek GPU Usage
```bash
# Di host
nvidia-smi

# Di dalam container
docker exec fedxchain-client-gpu-0 nvidia-smi
```

### Monitor semua containers
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## 🔧 Architecture GPU Version

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network (GPU)                      │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Blockchain  │◄─────┤   Deployer   │                    │
│  │   (Hardhat)  │      │   (Deploy    │                    │
│  │   Port 8545  │      │   Contract)  │                    │
│  └──────┬───────┘      └──────────────┘                    │
│         │                                                    │
│         │ ┌────────────────────────────────────┐           │
│         ├─┤        Aggregator Server           │           │
│         │ │  - Coordinates FL rounds           │           │
│         │ │  - Blockchain integration          │           │
│         │ │  - PoEx consensus                  │           │
│         │ │  Port 5000 (CPU)                   │           │
│         │ └────────────┬───────────────────────┘           │
│         │              │                                    │
│         │    ┌─────────┴─────────┐                         │
│         │    │                   │                         │
│  ┌──────▼────▼──┐  ┌──────────┐  ┌──────────┐            │
│  │ GPU Client 0 │  │GPU Client│  │GPU Client│            │
│  │ - PyTorch    │  │    1     │  │    2     │  ...       │
│  │ - CUDA 12.1  │  │          │  │          │            │
│  │ - SHAP       │  │          │  │          │            │
│  │ 🎮 GPU       │  │ 🎮 GPU   │  │ 🎮 GPU   │            │
│  └──────────────┘  └──────────┘  └──────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🆚 Perbedaan CPU vs GPU Version

| Feature | CPU Version | GPU Version |
|---------|------------|-------------|
| Model | Scikit-learn (Logistic Regression) | PyTorch Neural Network |
| Training Speed | ~1-2s per round | ~0.5s per round (with GPU) |
| Memory | ~500MB per client | ~1-2GB per client |
| Dependencies | sklearn, numpy | PyTorch, CUDA |
| GPU Required | ❌ No | ✅ Yes |
| Model Complexity | Linear | Deep Neural Network |

## 📈 Expected Performance

**GPU Speedup:**
- Training: 2-4x faster
- SHAP computation: 1.5-2x faster
- Overall: 2-3x faster per round

**GPU Memory Usage:**
- Per client: ~1-2GB VRAM
- 5 clients: ~5-10GB VRAM total
- Blockchain: ~100MB RAM

## 🐛 Troubleshooting

### Error: "could not select device driver"
```bash
# Pastikan NVIDIA Container Toolkit terinstall
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Jika error, reinstall toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Error: "CUDA out of memory"
```yaml
# Edit docker-compose.gpu.yml, batasi jumlah GPU per client
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # Gunakan 1 GPU
          capabilities: [gpu]
    limits:
      memory: 2G  # Batasi memory
```

### GPU tidak terdeteksi di Windows
1. Pastikan Docker Desktop menggunakan WSL2 backend
2. Install NVIDIA driver di Windows host
3. Install NVIDIA Container Toolkit di WSL2
4. Restart Docker Desktop

### PyTorch tidak menggunakan GPU
```python
# Test di dalam container
docker exec -it fedxchain-client-gpu-0 python -c "import torch; print(torch.cuda.is_available())"

# Harus output: True
```

## 📝 Notes

1. **Memory Requirements**: Setiap GPU client membutuhkan ~1-2GB VRAM
2. **Multi-GPU**: Jika punya multiple GPU, edit docker-compose.gpu.yml untuk assign GPU berbeda ke setiap client
3. **CPU Fallback**: Jika GPU tidak tersedia, PyTorch akan otomatis fallback ke CPU
4. **Performance**: GPU memberikan benefit signifikan untuk model neural network yang lebih besar

## 🔄 Switch Between CPU and GPU

```bash
# CPU version (current)
docker-compose up

# GPU version
docker-compose -f docker-compose.gpu.yml up
```

## 📚 References

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
