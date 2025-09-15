# 🌍 Translator MLOps Demo

A **mini-project** to demonstrate **MLOps practices** using [Hugging Face Transformers](https://huggingface.co/transformers/) and [MLflow](https://mlflow.org/).  
The project trains a small **English → Spanish translator** and shows the full lifecycle:

- ✅ Experiment tracking with MLflow  
- ✅ Model training with Hugging Face  
- ✅ Metrics logging (BLEU score)  
- 🚧 Model Registry + Serving (Week 2)  
- 🚧 Azure ML deployment (Week 3)  
- 🚧 Monitoring & Drift detection (Week 4)  

---

## 📦 Environment Setup

You can use either **Conda** or **Python venv**.

### Option A: Conda
```bash
conda create -n translator-mlops python=3.10 -y
conda activate translator-mlops
pip install -r requirements.txt
```

### Option B: venv
```bash
python -m venv .venv
# Activate:
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

Verify install:
```bash
python -c "import torch, transformers, datasets, mlflow; print('✅ Environment OK')"
```

---

## 🚀 How to Run

### 1. Start MLflow UI
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Open 👉 [http://localhost:5000](http://localhost:5000)

### 2. Train a small model
```bash
python src/train.py
```

This will:
- Log parameters and metrics (BLEU score) to MLflow  
- Save model checkpoints in `outputs/`  

---

## 📂 Project Structure
```
translator-mlops-demo/
├─ src/
│  ├─ data.py          # dataset + preprocessing
│  ├─ train.py         # training + MLflow logging
│  └─ utils.py         # helper functions
├─ configs/            # training configs (Week 2+)
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## 🛠 Tech Stack
- Python 3.10  
- PyTorch  
- Hugging Face Transformers + Datasets  
- MLflow  

---

## 📅 Roadmap
- [x] Week 1: Local training + MLflow logging  
- [ ] Week 2: Model Registry + Serving  
- [ ] Week 3: Deploy to Azure ML Online Endpoint  
- [ ] Week 4: Monitoring with Evidently + Azure Monitor  

---

## 📜 License
MIT License — free to use and adapt.
