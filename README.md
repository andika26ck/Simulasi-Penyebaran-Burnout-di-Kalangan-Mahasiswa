# 🧠 Simulasi Burnout Mahasiswa — ABM Dashboard

Dashboard interaktif simulasi penyebaran burnout mahasiswa menggunakan **Agent-Based Modeling (ABM)**.

## 📋 Deskripsi

Proyek ini merupakan Tugas Besar Pemodelan & Simulasi (UAS Minggu 16) dengan topik:

> *"Simulasi Penyebaran Burnout di Kalangan Mahasiswa Menggunakan Agent-Based Modeling: Evaluasi Efektivitas Intervensi Peer Support dan Self-Care"*

## 🚀 Cara Menjalankan

### Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy ke Streamlit Cloud
1. Fork / push repo ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → pilih `app.py` sebagai main file
4. Klik **Deploy** — selesai!

## 📁 Struktur File

```
burnout_dashboard/
├── app.py              # Dashboard Streamlit utama
├── simulation.py       # Core ABM engine (diekstrak dari v3)
├── requirements.txt    # Dependencies
├── .streamlit/
│   └── config.toml    # Tema & konfigurasi Streamlit
└── README.md
```

## 🔢 Model Matematika

```
B(t+1) = B(t) + S(t)·V − R·[α·PS + β·SC + δ·CBT] + γ·Cs·B̄ₙ
```

| Simbol | Deskripsi |
|--------|-----------|
| B(t) | Burnout Level (0–1) |
| S(t) | Stressor akademik |
| V | Vulnerability individual |
| R | Resilience (tumbuh dengan SC+CBT) |
| PS | Peer Support dari tetangga |
| SC | Self-Care effect |
| CBT | Efek program CBT/Mindfulness |
| γ | Laju kontagion burnout |

## 📊 Fitur Dashboard

| Tab | Isi |
|-----|-----|
| 📈 Hasil Simulasi | Trajektori burnout, komposisi populasi, KPI cards |
| 🗺️ Jaringan Sosial | Visualisasi interaktif jaringan antar agen |
| 🔬 Analitik Lanjut | Distribusi burnout, recovery time, profil agen |
| ⚖️ Komparasi Skenario | 5 skenario what-if dengan Monte Carlo |
| 📖 Tentang Model | Dokumentasi lengkap model & referensi |

## 🗂️ Versi Pengembangan

| Versi | Minggu | Fitur |
|-------|--------|-------|
| v1 | 6 | Base model |
| v2 | 10 | CBT + compassion fatigue |
| v3 | 12 | Monte Carlo 1000 iterasi |
| Dashboard | 16 | Streamlit interaktif |

## 📚 Referensi Utama

- Maslach & Leiter (2016) — Burnout theory
- Bakker et al. (2009) — Emotional contagion burnout
- Watts & Strogatz (1998) — Small-world networks
- Railsback & Grimm (2019) — Agent-Based Modeling
