# ML Platform Installation Guide

## 🚀 Quick Start (Essential Dependencies)

For minimal setup with core ML functionality:

```bash
# Install essential requirements only
pip install -r requirements-essential.txt
```

This includes:
- Flask web framework
- Core ML libraries (scikit-learn, pandas, numpy)
- High-performance models (XGBoost, LightGBM, CatBoost)
- Google AI Studio integration
- Basic visualization tools

## 📦 Full Installation (All Models)

For complete ML platform with all advanced features:

```bash
# Install all requirements (comprehensive ML suite)
pip install -r requirements.txt
```

This includes everything in essential plus:
- Deep learning frameworks (TensorFlow, PyTorch)
- Advanced clustering and dimensionality reduction
- Time series forecasting
- Natural language processing
- Computer vision capabilities
- AutoML libraries
- Model interpretation tools
- GPU acceleration (if available)

## ⚡ Performance Recommendations

### For Production Deployment:
```bash
# Essential + Performance optimizations
pip install -r requirements-essential.txt
pip install tensorflow>=2.13.0 torch>=2.0.0  # If you need deep learning
pip install numba>=0.57.0  # For numerical acceleration
```

### For Development:
```bash
# Full suite for experimentation
pip install -r requirements.txt
```

## 🔧 GPU Support (Optional)

If you have NVIDIA GPU and want GPU acceleration:

```bash
# For CUDA 11.x
pip install cupy-cuda11x>=12.0.0
pip install cudf-cu11>=23.0.0
```

## 📊 Model Coverage by Installation

### Essential Installation:
- ✅ All regression models (Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost)
- ✅ All classification models (Logistic, SVM, Random Forest, XGBoost, LightGBM, CatBoost)
- ✅ Basic clustering (K-Means, DBSCAN, Hierarchical)
- ✅ Dimensionality reduction (PCA, t-SNE, UMAP)
- ✅ Google AI Studio recommendations

### Full Installation:
- ✅ Everything from Essential +
- ✅ Deep learning models (Neural Networks, CNNs, RNNs)
- ✅ Time series forecasting
- ✅ AutoML capabilities
- ✅ Advanced ensemble methods
- ✅ Model interpretation tools
- ✅ GPU acceleration

## 🚦 Installation Verification

After installation, test the setup:

```python
# Test core functionality
from core_ml import ml_core
print("✅ ML Core loaded successfully")

# Test model availability
import xgboost, lightgbm, catboost
print("✅ High-performance models available")

# Test Google AI integration
import google.generativeai as genai
print("✅ Google AI Studio ready")
```

## 🐛 Troubleshooting

### Common Issues:

1. **XGBoost/LightGBM Installation Fails:**
   ```bash
   # Try conda instead
   conda install -c conda-forge xgboost lightgbm catboost
   ```

2. **TensorFlow GPU Issues:**
   ```bash
   # Install CPU version only
   pip install tensorflow-cpu>=2.13.0
   ```

3. **Memory Issues:**
   ```bash
   # Install lighter versions
   pip install scikit-learn pandas numpy xgboost
   ```

## 📋 Environment Setup

1. Create virtual environment:
   ```bash
   python -m venv ml_platform_env
   source ml_platform_env/bin/activate  # Linux/Mac
   ml_platform_env\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements-essential.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Google AI API key
   ```

## 🎯 Recommendation

- **For your Walmart sales data**: Use **essential installation** - it provides all needed models with excellent performance
- **For production**: Essential installation is recommended for stability
- **For experimentation**: Full installation gives you access to cutting-edge ML techniques

Choose the installation method that best fits your needs! 🚀