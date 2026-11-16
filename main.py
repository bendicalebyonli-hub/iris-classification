"""
=============================================================================
AI ASSIGNMENT 2026-2027: IRIS FLOWER CLASSIFICATION
=============================================================================
This script provides an end-to-end Machine Learning pipeline:
  1. Data Loading & Local Persistence (Iris dataset)
  2. Exploratory Data Analysis (EDA) & Visualizations
  3. Data Preprocessing (Stratified Split & Feature Scaling)
  4. Model Training & 5-Fold Cross-Validation:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Classifier (SVM)
     - Decision Tree Classifier
     - Random Forest Classifier
     - Multi-Layer Perceptron (Neural Network - MLP)
  5. Evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrices)
  6. Visualizations & Model Comparison Plots (saved to 'outputs/')
  7. Best Model Serialization (saved to 'models/')
  8. Interactive / Custom Sample Inference
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup directories
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def load_and_prepare_data():
    """Loads the Iris dataset, saves a CSV backup, and returns DataFrame and raw X, y."""
    print("\n" + "=" * 60)
    print(" STEP 1: LOADING DATASET")
    print("=" * 60)
    
    iris_raw = load_iris()
    feature_names = [name.replace(" (cm)", "").replace(" ", "_") for name in iris_raw.feature_names]
    target_names = list(iris_raw.target_names)
    
    df = pd.DataFrame(data=iris_raw.data, columns=feature_names)
    df["target"] = iris_raw.target
    df["species"] = df["target"].map({i: name for i, name in enumerate(target_names)})
    
    # Save CSV locally for independent reference
    csv_path = os.path.join(DATA_DIR, "iris.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Iris dataset loaded successfully. Total samples: {len(df)}")
    print(f"[+] Dataset saved locally to: {csv_path}")
    print(f"[+] Target species: {target_names}")
    
    return df, feature_names, target_names


def perform_eda(df, feature_names):
    """Performs Exploratory Data Analysis (EDA) and saves plots."""
    print("\n" + "=" * 60)
    print(" STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Dataset Summary & Missing Values ---")
    print(df.info())
    print("\n--- Missing Value Count ---")
    print(df.isnull().sum())
    
    print("\n--- Descriptive Statistics ---")
    print(df.describe().round(3))
    
    print("\n--- Class Distribution ---")
    print(df["species"].value_counts())
    
    # Visualizations
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Feature Distribution Histogram / KDE
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, feature in enumerate(feature_names):
        ax = axes[idx // 2, idx % 2]
        sns.histplot(data=df, x=feature, hue="species", kde=True, ax=ax, palette="Set2", alpha=0.6)
        ax.set_title(f"Distribution of {feature.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    hist_path = os.path.join(OUTPUTS_DIR, "feature_distributions.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[+] Saved feature distributions to: {hist_path}")
    
    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[feature_names].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Heatmap of Iris Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[+] Saved correlation heatmap to: {heatmap_path}")
    
    # 3. Pairplot
    pairplot = sns.pairplot(df[feature_names + ["species"]], hue="species", palette="Dark2", corner=False)
    pairplot.fig.subplots_adjust(top=0.94)
    pairplot.fig.suptitle("Pairwise Relationships of Iris Features by Species", fontsize=13, fontweight="bold")
    pairplot_path = os.path.join(OUTPUTS_DIR, "pairwise_features_plot.png")
    pairplot.savefig(pairplot_path, dpi=300)
    plt.close()
    print(f"[+] Saved pairwise relationships plot to: {pairplot_path}")


def preprocess_data(df, feature_names):
    """Splits data into stratified train/test sets and scales features."""
    print("\n" + "=" * 60)
    print(" STEP 3: PREPROCESSING & FEATURE SCALING")
    print("=" * 60)
    
    X = df[feature_names].values
    y = df["target"].values
    
    # 80% Train, 20% Test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"[+] Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the fitted scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[+] StandardScaler fitted and saved to: {scaler_path}")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_models():
    """Defines and returns candidate models dictionary."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
            "use_scaled": True
        },
        "K-Nearest Neighbors (KNN)": {
            "model": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
            "use_scaled": True
        },
        "Support Vector Machine (SVM)": {
            "model": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
            "use_scaled": True
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=RANDOM_STATE),
            "use_scaled": False
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE),
            "use_scaled": False
        },
        "Multi-Layer Perceptron (Neural Net)": {
            "model": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=600, random_state=RANDOM_STATE),
            "use_scaled": True
        }
    }
    return models


def train_and_evaluate(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, target_names):
    """Trains all models, evaluates cross-validation & test metrics, and plots comparisons."""
    print("\n" + "=" * 60)
    print(" STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    results = []
    trained_models = {}
    confusion_matrices = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for name, config in models.items():
        clf = config["model"]
        x_tr = X_train_scaled if config["use_scaled"] else X_train
        x_te = X_test_scaled if config["use_scaled"] else X_test
        
        # 5-Fold Cross Validation
        cv_scores = cross_val_score(clf, x_tr, y_train, cv=cv, scoring="accuracy")
        
        # Train on full training split
        clf.fit(x_tr, y_train)
        y_pred = clf.predict(x_te)
        
        # Test Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        trained_models[name] = clf
        confusion_matrices[name] = cm
        
        results.append({
            "Model": name,
            "CV_Mean_Acc": cv_scores.mean(),
            "CV_Std": cv_scores.std(),
            "Test_Accuracy": acc,
            "Precision_Macro": prec,
            "Recall_Macro": rec,
            "F1_Macro": f1
        })
        
        print(f"\n[{name}]")
        print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test Accuracy:     {acc:.4f}")
        print(f"  Precision (Macro): {prec:.4f}")
        print(f"  Recall (Macro):    {rec:.4f}")
        print(f"  F1-Score (Macro):  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
    
    results_df = pd.DataFrame(results).sort_values(by="Test_Accuracy", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 60)
    print(" SUMMARY COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    summary_path = os.path.join(OUTPUTS_DIR, "model_comparison_results.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n[+] Results saved to: {summary_path}")
    
    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    x_indices = np.arange(len(results_df))
    bar_width = 0.35
    
    plt.bar(x_indices - bar_width / 2, results_df["CV_Mean_Acc"], width=bar_width, label="5-Fold CV Mean Acc", color="#4C72B0")
    plt.bar(x_indices + bar_width / 2, results_df["Test_Accuracy"], width=bar_width, label="Test Set Accuracy", color="#55A868")
    
    plt.xlabel("Models", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy Score", fontsize=12, fontweight="bold")
    plt.title("Model Performance Comparison (CV vs Test Accuracy)", fontsize=14, fontweight="bold")
    plt.xticks(x_indices, results_df["Model"], rotation=20, ha="right")
    plt.ylim(0.85, 1.05)
    plt.legend()
    plt.tight_layout()
    comparison_chart_path = os.path.join(OUTPUTS_DIR, "model_comparison_chart.png")
    plt.savefig(comparison_chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparison chart to: {comparison_chart_path}")
    
    # Plot 2: Confusion Matrices Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=target_names, yticklabels=target_names)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrices to: {cm_path}")
    
    # Identify Best Model (by CV Mean + Test Accuracy)
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    
    print("\n" + "=" * 60)
    print(f" BEST PERFORMING MODEL: {best_model_name}")
    print(f" Test Accuracy: {best_row['Test_Accuracy']:.4f} | F1-Score: {best_row['F1_Macro']:.4f}")
    print("=" * 60)
    
    # Save Best Model
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump({
        "model_name": best_model_name,
        "model": best_model,
        "use_scaled": models[best_model_name]["use_scaled"],
        "target_names": target_names
    }, best_model_path)
    print(f"[+] Best model exported to: {best_model_path}")
    
    return best_model, models[best_model_name]["use_scaled"]


def demonstrate_inference(scaler, target_names):
    """Demonstrates how to use the saved best model for predicting new custom inputs."""
    print("\n" + "=" * 60)
    print(" STEP 5: SAMPLE INFERENCE DEMONSTRATION")
    print("=" * 60)
    
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    saved_payload = joblib.load(best_model_path)
    model = saved_payload["model"]
    use_scaled = saved_payload["use_scaled"]
    
    # Test cases representing typical measurements for Setosa, Versicolor, and Virginica
    samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Expected: setosa
        [6.0, 2.7, 4.2, 1.3],  # Expected: versicolor
        [7.2, 3.6, 6.1, 2.5]   # Expected: virginica
    ])
    
    samples_processed = scaler.transform(samples) if use_scaled else samples
    predictions = model.predict(samples_processed)
    
    # Check if predict_proba is supported
    probabilities = model.predict_proba(samples_processed) if hasattr(model, "predict_proba") else None
    
    for i, sample in enumerate(samples):
        pred_label = target_names[predictions[i]]
        confidence_str = ""
        if probabilities is not None:
            conf = probabilities[i][predictions[i]] * 100
            confidence_str = f" (Confidence: {conf:.2f}%)"
        print(f"Sample {i+1}: Features {sample.tolist()} --> Predicted Species: {pred_label.upper()}{confidence_str}")
        
    print("\n[+] All steps completed successfully!")


def main():
    df, feature_names, target_names = load_and_prepare_data()
    perform_eda(df, feature_names)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df, feature_names)
    models = get_models()
    train_and_evaluate(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, target_names)
    demonstrate_inference(scaler, target_names)


if __name__ == "__main__":
    main()
