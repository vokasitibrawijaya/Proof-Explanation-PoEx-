Dokumen ini disusun sebagai panduan operasional riset untuk mengimplementasikan **Proof of Explanation (PoEx)** pada sistem **FedXChain**. Fokus utama adalah mensimulasikan lingkungan terdistribusi (Blockchain & FL) di dalam satu mesin server dengan manajemen sumber daya yang optimal.

---

## Dokumen Perencanaan Eksperimen: PoEx

### 1. Identitas Eksperimen

* **Judul:** Analisis Ketahanan Protokol Konsensus PoEx terhadap Serangan *Model Poisoning* pada *Federated Learning* berbasis Blockchain.
* **Objektif:** Membuktikan bahwa mekanisme konsensus berbasis XAI (SHAP) dapat mendeteksi dan menolak pembaruan model yang berbahaya/anomali sebelum masuk ke buku besar (*ledger*).
* **Metode XAI:** SHAP (*Shapley Additive Explanations*).
* **Platform Blockchain:** Hyperledger Fabric (HLF) v2.x.

### 2. Spesifikasi Infrastruktur (Single Server)

| Komponen | Spesifikasi | Alokasi Riset |
| --- | --- | --- |
| **CPU** | Ryzen 7 (8 Core / 16 Thread) | 2 Core untuk OS/Docker, 6 Core untuk Node FL/BC. |
| **GPU** | NVIDIA VRAM 16GB | Fokus pada proses *Training* dan *Inference* SHAP. |
| **RAM** | 32 GB DDR4/D5 | 8GB untuk HLF, 16GB untuk FL Clients, 8GB Buffer. |
| **Storage** | SSD NVMe | Digunakan untuk persistensi data Ledger Blockchain. |

---

## 3. Desain Arsitektur Sistem (Simulasi)

Meskipun menggunakan 1 PC, kita akan menggunakan **Docker Containers** untuk menciptakan isolasi jaringan.

* **Node 1 (Orderer):** Mengelola konsensus blok (HLF).
* **Node 2 (Peer0):** Menyimpan *Smart Contract* (Chaincode) PoEx.
* **Node 3-5 (FL Clients):** Mensimulasikan 3 perangkat berbeda yang melakukan pelatihan lokal.

---

## 4. Prosedur Eksperimen (Setting Kerja)

### Tahap 1: Setup Lingkungan

1. **Sistem Operasi:** Gunakan **Ubuntu 22.04 LTS** (lebih stabil untuk Docker dan NVIDIA Driver).
2. **Containerization:** Instal Docker dan Docker Compose.
3. **Library AI:** Python 3.10+, PyTorch/TensorFlow, Flower (`flwr`), dan `shap`.

### Tahap 2: Implementasi Logika PoEx (Chaincode)

PoEx bekerja dengan membandingkan vektor penjelasan (). Logika yang akan dimasukkan ke dalam *Smart Contract* adalah:

* Jika  (Threshold), transaksi **Valid**.
* Jika , transaksi **Ditolak**, dan *Trust Score* node dikurangi.

### Tahap 3: Skenario Pengujian

Bapak perlu membagi eksperimen menjadi dua kelompok besar:

1. **Baseline:** FL standar tanpa PoEx (hanya agregasi FedAvg).
2. **Proposed (FedXChain):** FL dengan konsensus PoEx aktif.

**Jenis Serangan yang Disimulasikan:**

* **Label Flipping:** Node jahat mengubah label data (misal: 'Normal' jadi 'Attack').
* **Gaussian Noise:** Node jahat menambahkan *noise* pada *weights* model untuk merusak akurasi global.

---

## 5. Metrik Evaluasi (Untuk Paper)

Untuk mendapatkan data yang kuat bagi jurnal **IEEE Access**, Bapak harus mencatat:

1. **Model Performance:** Accuracy, Precision, Recall, dan F1-Score.
2. **Security:** *Success Rate* dari serangan *poisoning* (seberapa sering serangan berhasil menembus blockchain).
3. **Efficiency:** Waktu komputasi yang dibutuhkan untuk menghasilkan penjelasan SHAP dan waktu validasi di Blockchain (*Latency*).
4. **XAI Integrity:** Visualisasi SHAP untuk membuktikan bahwa node jahat memang memiliki pola kontribusi fitur yang berbeda (anomali).

---

## 6. Konfigurasi Eksekusi (Optimization)

Agar RAM 32GB dan VRAM 16GB tidak *crash*:

* **Sequential Execution:** Jangan jalankan semua kontainer klien secara bersamaan. Gunakan skrip Python untuk menjalankan pelatihan satu per satu, simpan hasilnya ke Blockchain, baru lanjut ke klien berikutnya.
* **Quantization:** Gunakan presisi `float16` pada model AI untuk menghemat VRAM.
* **SHAP Background Data:** Gunakan sampel kecil (misal: 100 data) sebagai referensi SHAP untuk mempercepat proses pembuatan penjelasan.

---