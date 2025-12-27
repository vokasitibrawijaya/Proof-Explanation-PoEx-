"""Test SHAP speed: KernelExplainer vs LinearExplainer"""
import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
import shap

# Generate test data
X, y = make_classification(n_samples=100, n_features=30, random_state=42)
model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
model.fit(X, y)

X_sample = X[:10]

print("\n🐢 Testing KernelExplainer (SLOW)...")
t0 = time.time()
try:
    explainer_kernel = shap.KernelExplainer(model.predict_proba, X_sample)
    shap_kernel = explainer_kernel.shap_values(X_sample)
    time_kernel = time.time() - t0
    print(f"   Time: {time_kernel:.2f}s")
except Exception as e:
    print(f"   Error: {e}")
    time_kernel = 999

print("\n🚀 Testing LinearExplainer (FAST)...")
t0 = time.time()
try:
    explainer_linear = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
    shap_linear = explainer_linear.shap_values(X_sample)
    time_linear = time.time() - t0
    print(f"   Time: {time_linear:.3f}s")
    print(f"\n⚡ Speedup: {time_kernel/time_linear:.0f}x faster!")
except Exception as e:
    print(f"   Error: {e}")
