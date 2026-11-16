# AI Assignment (2026-2027): Iris Flower Classification

This repository contains a complete Artificial Intelligence & Machine Learning project for classifying Iris flower species (**Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**) based on sepal and petal dimensions.

---

## 📁 Project Structure

```text
IRIS_CLASSIFICATION/
├── data/
│   └── iris.csv                      # Local CSV copy of Iris dataset (generated on run)
├── models/
│   ├── best_model.joblib             # Serialized top-performing model
│   └── scaler.joblib                 # Serialized StandardScaler object
├── outputs/
│   ├── feature_distributions.png     # Histograms & KDE of features per species
│   ├── correlation_heatmap.png       # Correlation matrix between attributes
│   ├── pairwise_features_plot.png    # Pairwise scatter & KDE plots
│   ├── confusion_matrices.png        # Confusion matrices for all 6 models
│   ├── model_comparison_chart.png    # 5-fold CV vs test accuracy bar chart
│   └── model_comparison_results.csv # Exported metric comparison table
├── iris_classification.ipynb         # Interactive Jupyter Notebook with rich visuals
├── main.py                           # End-to-end standalone executable pipeline script
├── gui_app.py                        # Interactive Desktop Graphical User Interface (GUI)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation and guide
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Install Dependencies
Open your command prompt or terminal in this directory and run:
```bash
pip install -r requirements.txt
```

### 3. Run the Standalone Pipeline
To execute data loading, EDA, training, cross-validation, plot generation, and inference in one go:
```bash
python main.py
```
All visualization artifacts (`.png`) and performance metrics (`.csv`) will be automatically saved inside the `outputs/` folder.

### 4. Launch the Executive Desktop GUI (No extra install required)
To open the modern high-DPI desktop interface with live anatomical visualizer:
```bash
python gui_app.py
```

### 5. Launch the Modern Browser Web Dashboard (Streamlit)
To open the rich interactive web application with 3D spatial scatter projections and interactive probability charts:
```bash
pip install streamlit plotly
streamlit run app.py
```

### 6. Interactive Jupyter Notebook
To run the interactive notebook in VS Code or Jupyter:
```bash
jupyter notebook iris_classification.ipynb
```

---

## 🧠 Machine Learning Algorithms Implemented

1. **Logistic Regression**: Linear multiclass classification with L2 regularization.
2. **K-Nearest Neighbors (KNN)**: Non-parametric distance-based classification ($k=5$).
3. **Support Vector Machine (SVM)**: Radial Basis Function (RBF) kernel mapping into higher-dimensional feature space.
4. **Decision Tree Classifier**: Rule-based partitioning using Gini impurity criterion.
5. **Random Forest Classifier**: Ensemble of 100 bagged decision trees with feature sub-sampling.
6. **Multi-Layer Perceptron (MLP)**: Feedforward Artificial Neural Network with hidden layers `(16, 8)` and ReLU activations.

---

## 📊 Evaluation Methodology

- **Stratified Split**: 80% train / 20% test split preserving class proportions.
- **Feature Scaling**: Standardized via `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) for scale-sensitive models.
- **5-Fold Stratified Cross-Validation**: Ensures models generalize well across different training subsets.
- **Metrics Tracked**: Accuracy, Precision (Macro), Recall (Macro), F1-Score (Macro), and Confusion Matrices.

---

## 🔮 Custom Prediction Example
The trained pipeline supports instant inference on arbitrary flower measurements:
```python
import joblib

# Load saved pipeline
payload = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")

model = payload["model"]
species_labels = payload["target_names"]

# Sample: [sepal length, sepal width, petal length, petal width]
new_flower = [[5.1, 3.5, 1.4, 0.2]]
features = scaler.transform(new_flower) if payload["use_scaled"] else new_flower

predicted_class = model.predict(features)[0]
confidence = model.predict_proba(features)[0][predicted_class] * 100

print(f"Predicted Species: {species_labels[predicted_class]} ({confidence:.2f}%)")
```
