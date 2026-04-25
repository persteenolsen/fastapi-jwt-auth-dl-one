# 🚀 FastAPI ML Inference Service with ONNX Runtime and PyTorch-Trained Model

Last updated:
- 25-04-2026

A production-style machine learning inference API demonstrating model training in PyTorch, export to ONNX, and efficient inference using FastAPI and ONNX Runtime.

---

## 🚀 What this project demonstrates

- End-to-end ML workflow (training → export → inference)
- Model training in PyTorch with deployment in ONNX format
- Clear separation of training, development, and production environments
- Secure API design with JWT authentication
- Lightweight inference using ONNX Runtime
- Input validation and controlled inference behavior

---

## 🚀 Features

- FastAPI REST API
- PyTorch-trained neural network (XOR problem)
- ONNX Runtime inference engine
- JWT authentication (OAuth2 password flow)
- Pydantic request validation
- Strict XOR input enforcement (0 or 1 only)
- Custom error handling

---

## 📈 Notes

- XOR is a classic toy problem used to demonstrate why neural networks require non-linear layers
- Focus is system design, not model accuracy
- Demonstrates full ML pipeline: training → export → deployment
- Clear separation between training and production inference environments

---

## 🧠 Machine Learning Model

- Framework: PyTorch
- Architecture:
  - Input layer: 2 features
  - Hidden layer: 8 neurons (ReLU activation)
  - Output layer: 1 neuron (Sigmoid activation)
- Task: XOR binary classification
- Model export: ONNX format for inference

---

## 🔁 System Architecture

Client → JWT Authentication → Token → Prediction Request → ONNX Runtime → Response

---

## 🔐 Authentication

### Get token

```bash
curl -X POST "http://localhost:8000/token" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "username=admin&password=password"
```

### Response

```json id="auth_response"
{
  "access_token": "your_token_here",
  "token_type": "bearer"
}
```

### Use token

Authorization header:

Authorization: Bearer <token>

---

## 📊 Prediction Endpoint

### Request

POST /predict

Body:

```json id="predict_request"
{
  "x1": 0,
  "x2": 1
}
```

---

### Response

```json id="predict_response"
{
  "user": "admin",
  "prediction": 0.9989
}
```

---

## ⚠️ Input Rules

| x1 | x2 | output |
|----|----|--------|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

Only binary inputs (0 or 1) are accepted.

---

## 🧪 Tech Stack

- FastAPI  
- PyTorch  
- ONNX Runtime  
- NumPy  
- Pydantic  
- Python-JOSE (JWT)

---

## 📁 Project Structure

```
project/
├── train.py              # Train model and export ONNX
├── main.py               # FastAPI inference service
├── model.onnx            # Exported model
├── requirements/
│   ├── train.txt         # Training dependencies
│   ├── dev.txt           # Local development dependencies
├── requirements.txt      # Production (Vercel) dependencies
└── .env                  # Local configuration (not deployed)
```
---

## 🧪 TRAINING ENVIRONMENT (train.txt)

Used ONLY for training the model locally.

torch>=2.2,<2.4
onnx>=1.16,<2.0
numpy>=1.26,<2.0

---

## 💻 LOCAL DEVELOPMENT ENVIRONMENT (dev.txt)

Used to run FastAPI locally and test full pipeline.

-r train.txt
  
fastapi>=0.110,<0.116
uvicorn>=0.29,<0.31
python-dotenv>=1.0,<2.0
python-jose[cryptography]>=3.3,<4.0
onnxruntime>=1.17,<1.20
numpy>=1.26,<2.0
python-multipart>=0.0.9

---

## 🚀 PRODUCTION ENVIRONMENT (requirements.txt)

Used for deployment (no PyTorch included).

fastapi>=0.110,<0.116  
uvicorn>=0.29,<0.31  
onnxruntime>=1.17,<1.20  
numpy>=1.26,<2.0  
python-dotenv>=1.0,<2.0  
python-jose[cryptography]>=3.3,<4.0  
python-multipart>=0.0.9  

---

## 🧪 TRAINING PIPELINE

Install training dependencies:

```bash
pip install -r requirements/train.txt
```

Run training:

```bash
python train.py
```

Output:
model.onnx

---

## 🚀 LOCAL DEVELOPMENT

Install dependencies:

```bash
pip install -r requirements/dev.txt
```

Run API:

```bash
uvicorn main:app --reload
```

Swagger UI:
http://127.0.0.1:8000/docs

---

## 🚀 DEPLOYMENT (VERCEL)

Production uses:

requirements.txt

No PyTorch included in deployment.

---

## 👨‍💻 Author

Learning project for understanding machine learning system deployment and inference architecture.