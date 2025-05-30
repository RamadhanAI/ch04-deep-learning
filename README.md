# Chapter 4: Deep Learning with TensorFlow & PyTorch
this repository contains the full code and notebooks to build, train, and deploy deep learning models covering CNNs, RNNs, and Transformers for real-world AI tasks.

This repo includes:
- CNN and Transformer models
- Vision and NLP projects
- Deployment with FastAPI
- GradCAM & SHAP explanations

Train and deploy deep learning models with real-world architectures (CNN, DistilBERT), interpretability (GradCAM), and production APIs.

[![FastAPI Docs](https://img.shields.io/badge/docs-FastAPI-green?logo=fastapi)](http://127.0.0.1:8000/docs)
[![Run Notebook](https://img.shields.io/badge/Notebook-MNIST--CNN-blue?logo=jupyter)](notebooks/cnn_mnist_c.ipynb)
[![Run Notebook](https://img.shields.io/badge/Notebook-DistilBERT--Tickets-purple?logo=jupyter)](notebooks/bert_ticket_classification.ipynb)

---

## 📦 Contents

- `notebooks/`: Jupyter notebooks for CNN, DistilBERT, GradCAM
- `src/`: Modular training pipelines
- `deployment/`: FastAPI for serving models
- `Dockerfile`: Containerized deployment
- `requirements.txt`: All dependencies

Setup Instructions
Clone the repo:
git clone https://github.com/RamadhanAI/ch04-deep-learning.git
cd ch04-deep-learning
Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Contents
ch04_cnn_baseline.ipynb: CNN training and evaluation on MNIST-C
ch04_ticket_distilbert.ipynb: DistilBERT fine-tuning for ticket triage
gradcam.py: Utilities for CNN interpretability visualization
deploy.py: Example FastAPI server for model inference
Running Notebooks
Open notebooks in Google Colab or locally to train models and visualize results.

Deployment
Use deploy.py to launch a FastAPI inference server:

uvicorn deploy:app --reload
Troubleshooting
Ensure CUDA and drivers are compatible for GPU acceleration.
Use pip install --upgrade if dependencies fail.
Report issues in the GitHub repo issues tab.
