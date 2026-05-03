# 🚀 FastAPI ML Inference Service with ONNX Runtime and PyTorch-Trained Model

Last updated:
- 03-05-2026

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
  - Hidden layer: 4 neurons (Tanh activation)
  - Output layer: 1 neuron (Sigmoid activation)
- Task: XOR binary classification
- Model export: ONNX format for inference

---

## 🔧 Model Tuning

During development, the model was tuned to improve stability and realism of predictions.

Key tuning changes:

- Added Tanh function instead of ReLu
- Reduced the hidden layer size (8 → 4 neurons)
- Kept learning rate (0.01)
- Kept training epochs (2000)

Result:

- Smoother and faster model
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

Learning project for understanding machine learning system deployment and inference architecture

---

## 📈 Model Parameter Adjustments Guide

This guide outlines how common neural network parameters influence the training process and results. It also provides recommendations for adjusting them to optimize model performance.

## 1. Number of Hidden Layers
- **Increase:** Adds complexity, enabling the model to capture more features. However, it also increases the risk of overfitting. This is suitable for complex datasets.
- **Decrease:** Results in a simpler model, which may underfit complex tasks but can work well for small or simple datasets.

**Recommendation:**  
- Start with 1-2 hidden layers.  
- Use 3-5 layers for more complexity if the dataset requires it.

---

## 2. Number of Neurons (per Layer)
- **Increase:** More neurons allow the model to learn more complex patterns. However, too many neurons increase the risk of overfitting if the data is limited.
- **Decrease:** A simpler model with fewer neurons reduces overfitting but may fail to capture complex patterns.

**Recommendation:**  
- Start with 4-16 neurons per layer.  
- Increase the number if the model is not capturing enough complexity.

---

## 3. Learning Rate (lr)
- **Increase:** Speeds up learning but can cause the model to overshoot the optimal weights, potentially leading to poor convergence or instability.
- **Decrease:** Results in slower, more stable convergence, but may get stuck in local minima or take too long to converge.

**Recommendation:**  
- Start with a learning rate between `0.001` and `0.01`.  
- For faster convergence on simpler tasks, try between `0.1` and `0.5`.

---

## 4. Epochs
- **Increase:** More epochs allow the model to learn longer, but too many can lead to overfitting (where the model memorizes the training data).
- **Decrease:** Fewer epochs may cause underfitting, where the model hasn't learned enough.

**Recommendation:**  
- Start with 100-300 epochs.  
- Monitor for overfitting and stop earlier if needed.

---

## 5. Weight Decay (Regularization)
- **Increase:** Helps prevent overfitting by penalizing large weights. However, excessive decay can lead to underfitting.
- **Decrease:** Provides more flexibility but increases the risk of overfitting.

**Recommendation:**  
- Start with weight decay values between `1e-4` and `1e-5`.  
- Increase for larger networks or more complex datasets.

---

## Additional Tips:
- **Hidden Layers and Neurons:** Start with simpler architectures and increase complexity only when necessary. Use cross-validation to test performance.
- **Learning Rate:** Consider using adaptive optimizers like **Adam** for better learning rate control.
- **Epochs:** Monitor validation loss and implement early stopping to prevent overfitting, especially with large datasets.
- **Weight Decay:** Use weight decay to prevent overfitting, but be cautious not to overdo it, as this could lead to underfitting.

---

## Key Takeaways:
- **Start simple:** Use 1-2 layers and 4-16 neurons.
- **Adjust learning rate:** Reduce it if the model oscillates or diverges.
- **Monitor overfitting:** Use early stopping or adjust epochs accordingly.
- **Regularize with care:** Apply weight decay to prevent overfitting but avoid excessive regularization.