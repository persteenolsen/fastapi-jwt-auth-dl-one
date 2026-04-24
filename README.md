# 🚀 FastAPI ML Inference API (XOR Neural Network)

Last updated:

- 24-04-2026

A secure machine learning inference API built with FastAPI, featuring a PyTorch-trained MLP model exported to ONNX, with JWT authentication and strict XOR input validation.

## 🚀 What this project demonstrates

- End-to-end ML pipeline
- Model training → export → inference
- Secure API with JWT authentication
- Production-style backend structure
- Input validation and error handling

---

## 🚀 Features

- FastAPI REST API
- PyTorch-trained MLP (XOR problem)
- ONNX Runtime inference
- JWT authentication (OAuth2 password flow)
- Pydantic input validation
- XOR-only input enforcement (0 or 1)
- Custom error handling

---

## 🧠 Machine Learning Model

- Framework: PyTorch  
- Architecture:
  - Input: 2 features  
  - Hidden layer: 4 neurons (ReLU)  
  - Output: 1 neuron (Sigmoid)  
- Task: XOR binary classification  
- Export: ONNX  

---

## 🔁 System Architecture

Client → JWT Login → Token → Predict Request → ONNX Runtime → Neural Network → Response

---

## 🔐 Authentication

### Step 1: Get token

POST /token

Send form data:
- username = admin
- password = password

Example request:

curl example:
    curl -X POST "http://localhost:8000/token" 
    -H "Content-Type: application/x-www-form-urlencoded" 
    -d "username=admin&password=password"

Response:

{
  "access_token": "your_token_here",
  "token_type": "bearer"
}

---

### Step 2: Use token

Add header:

Authorization: Bearer <token>

---

## 📊 Prediction Endpoint

### Request

POST /predict

Headers:
Authorization: Bearer <token>
Content-Type: application/json

Body:

{
  "x1": 0,
  "x2": 1
}

---

### Response

{
  "user": "admin",
  "prediction": 0.9989
}

---

## ⚠️ Input Rules

Only XOR-valid inputs:

| x1 | x2 | output |
|----|----|--------|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

Invalid inputs (letters, numbers outside 0/1) are rejected.

---

## 🧪 Use Cases

- ML API deployment learning project
- Neural network inference service
- FastAPI backend practice
- JWT authentication demo
- ONNX serving pipeline

---

## 🛠 Tech Stack

- FastAPI
- PyTorch
- ONNX Runtime
- NumPy
- Pydantic
- Python-JOSE (JWT)

---


## 📈 Notes

- XOR is a toy dataset
- Focus is ML system design, not real-world prediction accuracy

---

# 📁 Project Structure

```
project/
├── train.py              # Train + export ONNX
├── main.py               # FastAPI inference API (Vercel)
├── requirements/train.txt             # ML training dependencies
├── requirements/dev.txt               # local dev environment
├── requirements.txt      # Vercel production dependencies
├── model.onnx           # exported model
└── .env                 # local config (not deployed)
```

---

# 🧪 1. TRAINING ENVIRONMENT (train.txt)

Used ONLY for training the model locally.

torch>=2.2,<2.4
onnx>=1.16,<2.0
numpy>=1.26,<2.0

---

# 💻 2. LOCAL DEVELOPMENT ENVIRONMENT (dev.txt)

Used to run FastAPI locally + test full pipeline.

-r train.txt

fastapi>=0.110,<0.116
uvicorn>=0.29,<0.31
python-dotenv>=1.0,<2.0
python-jose[cryptography]>=3.3,<4.0
onnxruntime>=1.17,<1.20

python-multipart>=0.0.9

---

# 🚀 3. VERCEL PRODUCTION ENVIRONMENT (requirements.txt)

Used for deployment (NO PyTorch included).

fastapi>=0.110,<0.116
uvicorn>=0.29,<0.31
python-dotenv>=1.0,<2.0
python-jose[cryptography]>=3.3,<4.0
onnxruntime>=1.17,<1.20
numpy>=1.26,<2.0

python-multipart>=0.0.9

---

# 🧪 TRAINING PIPELINE

Install training dependencies:
pip install -r train.txt (included when installing dev.txt like below)

Run training:
python train.py

Output:
model.onnx

---

# 🚀 LOCAL DEVELOPMENT

Install dev environment:
pip install -r requirements/dev.txt

Run FastAPI:
uvicorn main:app --reload

Swagger UI:
http://127.0.0.1:8000/docs

---

# 🚀 DEPLOYMENT (VERCEL)

Vercel uses:
requirements.txt

No PyTorch required.

---

## 👨‍💻 Author

Learning project for understanding ML system deployment.