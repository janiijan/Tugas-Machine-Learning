# Aplikasi Prediksi Customer Churn

Tugas Besar Machine Learning (Part 1 sampai Part 3).
Aplikasi web untuk memprediksi apakah pelanggan telekomunikasi berisiko berhenti berlangganan (churn).

## Alur Proyek
1. **Part 1** - Ide dan implementasi dengan kerangka CRISP-DM (prediksi churn, dataset Kaggle).
2. **Part 2** - Preprocessing, tuning 5 algoritma (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting), evaluasi, dan analisis good fit.
3. **Part 3** - Model terbaik diunduh dari Google Colab (`churn_model.pth`) lalu dipakai pada aplikasi web ini.

## Model
- Algoritma: **Random Forest** hasil tuning (GridSearchCV, 5-fold)
- Parameter: `n_estimators=300`, `max_depth=15`, `min_samples_leaf=10`
- Hasil pada data uji: F1 = 0,638 | Recall = 0,778 | ROC-AUC = 0,844
- Status fitting: good fit (selisih Train-Test F1 = 0,055)
- Dataset: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

File `churn_model.pth` berisi pipeline scikit-learn (preprocessing + model) yang disimpan dengan `joblib`.

## Struktur Folder
```
.
├── app.py              # API FastAPI + penyaji halaman web
├── index.html          # User Interface (Vue 3)
├── churn_model.pth     # Model hasil Google Colab
├── requirements.txt
└── README.md
```

## Cara Menjalankan
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Buka **http://127.0.0.1:8000** di browser, isi data pelanggan, lalu klik **Prediksi**.
Dokumentasi API otomatis tersedia di http://127.0.0.1:8000/docs.

> Versi scikit-learn harus sama dengan yang dipakai di Google Colab (1.6.1), karena model disimpan dengan versi tersebut.

## Endpoint
| Method | Path | Keterangan |
|---|---|---|
| GET | `/` | Halaman User Interface |
| GET | `/health` | Cek status aplikasi |
| POST | `/predict` | Prediksi churn dari data pelanggan (JSON) |

## Pengembang
- Nama: Muhammad Rafsyanjani
- NIM:  202343501742
