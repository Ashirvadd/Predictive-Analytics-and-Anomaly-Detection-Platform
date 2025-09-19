# Predictive Analytics & Anomaly Detection Suite

A research-grade AI system that integrates advanced **time series forecasting** with an **ensemble anomaly detection framework** and a **domain-specific large language model** for human-readable anomaly explanations. This project leverages state-of-the-art deep learning architectures, hyperparameter optimization, and efficient deployment techniques to deliver sub-second inference and high accuracy.

---
<img width="963" height="544" alt="Image" src="https://github.com/user-attachments/assets/6f204a88-5dd1-4a81-a29e-ed6c64676301" />
<img width="1038" height="704" alt="Image" src="https://github.com/user-attachments/assets/c911ebb7-a2f8-46be-b940-60c62d129b55" />
<img width="1752" height="736" alt="Image" src="https://github.com/user-attachments/assets/57114726-e8df-4ea0-a891-38f52b07bfa0" />
<img width="1799" height="629" alt="Image" src="https://github.com/user-attachments/assets/d2129a60-f8d2-455f-9d68-21c4daa5caed" />
<img width="752" height="464" alt="Image" src="https://github.com/user-attachments/assets/7409f93c-1663-4365-ae6a-f56ac5635483" />
<img width="729" height="499" alt="Image" src="https://github.com/user-attachments/assets/20c1f29f-ec74-4598-9d05-f81e0586a919" />

## 🚀 Features

### 1. **Time Series Forecasting (TimesNet)**
- **Architecture**: Implements TimesNet with multi-scale convolutional kernels and temporal feature embeddings.
- **Performance**: Achieves R² = 0.967, MAPE = 3.2%, and MAE = 0.074 on 1.75M+ time series records.
- **Forecast Horizon**: 96-hour input → 24-hour output.
- **Optimized Training**: Optuna Bayesian optimization, mixed precision (FP16), and early stopping.

### 2. **Ensemble Anomaly Detection**
- Combines **Reconstruction Error**, **Z-Score**, and **Isolation Forest** methods.
- **Accuracy**: 94.6% detection accuracy with <4% false positive rate.
- **Speed**: Sub-second anomaly detection per sequence.
- **Robustness**: Consensus-based voting for high reliability.

### 3. **Anomaly Explanation with DistilGPT-2**
- **Model**: Fine-tuned DistilGPT-2 (82M parameters) on 18K+ domain-specific prompt–response pairs.
- **Output**: Generates clear, context-aware explanations for detected anomalies.
- **Performance**: 89.7% human evaluator approval, 2.8× faster convergence vs. GPT-2.
- **Inference Speed**: ~47ms average response time.

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Frameworks**: PyTorch, Hugging Face Transformers, TimesNet
- **Optimization**: Optuna, CUDA Acceleration
- **Anomaly Detection**: Reconstruction Error, Z-Score, Isolation Forest
- **Interface**: Gradio Web App
- **Deployment**: REST API Endpoints, Batch Processing Support

---

---

## 📌 Research Impact
- **Predictive Maintenance**: Enables early fault detection.
- **Operational Efficiency**: Cuts analysis time by 60%.
- **Scalability**: Supports 1000+ concurrent sequences with low latency.
- **Generality**: Easily adaptable to multiple domains beyond the original dataset.

---
