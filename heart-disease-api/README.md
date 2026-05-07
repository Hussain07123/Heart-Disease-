# ❤️ Heart Disease Prediction API

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

A production-ready REST API that predicts the likelihood of heart disease based on patient clinical data, built with **FastAPI** and **XGBoost**.

---

## 🚀 Features

- **Single & batch predictions** via REST endpoints
- **Auto-generated Swagger UI** at `/docs`
- **Full input validation** with Pydantic schemas
- **Risk levels**: Low / Moderate / High with confidence scores
- **CORS enabled** — ready to connect to any frontend

---

## 📁 Project Structure

```
heart-disease-api/
├── main.py           # FastAPI app + all endpoints
├── train_model.py    # Train + save XGBoost model
├── schema.py         # Pydantic input/output schemas
├── model.pkl         # Saved trained model (generated)
├── scaler.pkl        # Saved feature scaler (generated)
├── requirements.txt
├── data/
│   ├── heart.csv            # Dataset
│   └── generate_data.py     # Sample data generator
└── README.md
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Hussain07123/heart-disease-api
cd heart-disease-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset (or add real heart.csv from Kaggle)
python data/generate_data.py

# 4. Train the model
python train_model.py

# 5. Start the API
uvicorn main:app --reload
```

Visit **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Model health check |
| POST | `/predict` | Single patient prediction |
| POST | `/predict/batch` | Batch predictions (up to 100) |

---

## 🧪 Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "sex": 1, "cp": 0,
    "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.1423,
  "risk_level": "Low",
  "message": "✅ Low likelihood of heart disease detected. Risk level: Low (14.2% confidence)."
}
```

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Classifier |
| Dataset | Cleveland Heart Disease Dataset |
| Features | 13 clinical attributes |
| Evaluation | Accuracy + ROC-AUC |

---

## ⚠️ Disclaimer

This API is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

**Made with ❤️ by [Hussain07123](https://github.com/Hussain07123)**
