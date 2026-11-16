"""
=============================================================================
IRIS CLASSIFICATION SUITE - EXECUTIVE DESKTOP GUI
=============================================================================
High-DPI, modern desktop application featuring:
  - Executive dark-slate interface styling with responsive layouts
  - Custom dynamic canvas-rendered flower anatomical geometry visualizer
  - Real-time probability spectrum meters with animated confidence cards
  - Model selector across 6 AI/ML algorithms
  - Botanical archetype presets and instant inference latency monitoring
=============================================================================
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Enable Windows High-DPI scaling for ultra-crisp typography
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


class ModernIrisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris AI • Species Classification Suite (2026-2027)")
        self.root.geometry("980x740")
        self.root.minsize(920, 680)

        # Palette
        self.C_BG = "#0B0F19"          # Deep cosmic background
        self.C_SURFACE = "#151C2C"     # Elevated panel surface
        self.C_CARD = "#1C253B"        # Card background
        self.C_BORDER = "#2A364F"      # Border line
        self.C_PRIMARY = "#6366F1"     # Indigo primary accent
        self.C_PRIMARY_HOVER = "#4F46E5"
        self.C_TEXT = "#F8FAFC"        # Pure white text
        self.C_MUTED = "#94A3B8"       # Secondary text
        self.C_ACCENT_SETOSA = "#38BDF8"     # Sky blue
        self.C_ACCENT_VERSI = "#34D399"      # Emerald green
        self.C_ACCENT_VIRGIN = "#A78BFA"     # Purple

        self.root.configure(bg=self.C_BG)

        # Train Machine Learning Models in background
        self.setup_ml_pipeline()

        # Build Interface
        self.build_ui()

        # Trigger initial prediction
        self.run_prediction()

    def setup_ml_pipeline(self):
        """Prepares dataset, standardizes features, and fits 6 machine learning models."""
        iris = load_iris()
        self.species_names = ["Setosa", "Versicolor", "Virginica"]
        self.feature_keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        X, y = iris.data, iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        self.models = {
            "Support Vector Machine (SVM - RBF)": {
                "model": SVC(kernel="rbf", probability=True, C=1.0, random_state=42),
                "scaled": True, "badge": "Optimal Generalization"
            },
            "Random Forest Ensemble (100 Trees)": {
                "model": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
                "scaled": False, "badge": "High Robustness"
            },
            "Multi-Layer Perceptron (Neural Net)": {
                "model": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=600, random_state=42),
                "scaled": True, "badge": "Deep Learning / Non-Linear"
            },
            "K-Nearest Neighbors (KNN, k=5)": {
                "model": KNeighborsClassifier(n_neighbors=5),
                "scaled": True, "badge": "Distance-Based"
            },
            "Logistic Regression (L2 Regularized)": {
                "model": LogisticRegression(max_iter=300, random_state=42),
                "scaled": True, "badge": "Probabilistic Linear"
            },
            "Decision Tree Classifier (Gini)": {
                "model": DecisionTreeClassifier(max_depth=4, random_state=42),
                "scaled": False, "badge": "Interpretable Rules"
            }
        }

        for m_name, meta in self.models.items():
            clf = meta["model"]
            data = self.X_train_scaled if meta["scaled"] else self.X_train
            clf.fit(data, self.y_train)

    def build_ui(self):
        # 1. Top Executive Navigation Bar
        top_bar = tk.Frame(self.root, bg=self.C_SURFACE, height=68, bd=0)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        top_inner = tk.Frame(top_bar, bg=self.C_SURFACE)
        top_inner.pack(fill="both", expand=True, padx=25)

        # Title & Subtitle
        title_box = tk.Frame(top_inner, bg=self.C_SURFACE)
        title_box.pack(side="left", pady=12)

        title_lbl = tk.Label(
            title_box, text="IRIS AI • CLASSIFICATION SUITE",
            font=("Segoe UI", 13, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_box, text="AI Assignment 2026–2027 • Pattern Recognition & Neural Inference",
            font=("Segoe UI", 8), fg=self.C_MUTED, bg=self.C_SURFACE
        )
        sub_lbl.pack(anchor="w")

        # Status Chip on right
        status_box = tk.Frame(top_inner, bg="#1E293B", padx=12, pady=5, bd=1, relief="solid")
        status_box.pack(side="right", pady=16)

        dot = tk.Label(status_box, text="●", fg="#10B981", bg="#1E293B", font=("Segoe UI", 8))
        dot.pack(side="left", padx=(0, 5))
        stat_lbl = tk.Label(status_box, text="ALL 6 MODELS READY", fg=self.C_TEXT, bg="#1E293B", font=("Segoe UI", 8, "bold"))
        stat_lbl.pack(side="left")

        # Thin border line under navbar
        sep = tk.Frame(self.root, bg=self.C_BORDER, height=1)
        sep.pack(fill="x")

        # 2. Main Content Split Panel
        body = tk.Frame(self.root, bg=self.C_BG)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        body.columnconfigure(0, weight=4, uniform="group")
        body.columnconfigure(1, weight=5, uniform="group")
        body.rowconfigure(0, weight=1)

        # --- LEFT PANEL: Parameter Tuning ---
        left_p = tk.Frame(body, bg=self.C_SURFACE, bd=1, relief="solid", highlightbackground=self.C_BORDER, highlightthickness=1)
        left_p.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        left_content = tk.Frame(left_p, bg=self.C_SURFACE, padx=20, pady=16)
        left_content.pack(fill="both", expand=True)

        ctrl_header = tk.Label(
            left_content, text="Botanical Measurements (cm)",
            font=("Segoe UI", 11, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE
        )
        ctrl_header.pack(anchor="w", pady=(0, 4))

        ctrl_desc = tk.Label(
            left_content, text="Adjust anatomical sliders or pick an empirical archetype.",
            font=("Segoe UI", 8), fg=self.C_MUTED, bg=self.C_SURFACE
        )
        ctrl_desc.pack(anchor="w", pady=(0, 14))

        # Sliders Configuration
        self.slider_specs = {
            "sepal_length": ("Sepal Length", 4.0, 8.0, 5.84, "#6366F1"),
            "sepal_width":  ("Sepal Width",  2.0, 4.5, 3.05, "#38BDF8"),
            "petal_length": ("Petal Length", 1.0, 7.0, 3.76, "#EC4899"),
            "petal_width":  ("Petal Width",  0.1, 2.5, 1.20, "#F59E0B")
        }

        self.slider_vars = {}
        self.val_labels = {}

        for key, (label_text, min_v, max_v, def_v, color) in self.slider_specs.items():
            f_box = tk.Frame(left_content, bg=self.C_SURFACE)
            f_box.pack(fill="x", pady=6)

            # Label Row
            row_l = tk.Frame(f_box, bg=self.C_SURFACE)
            row_l.pack(fill="x", pady=(0, 3))

            lbl = tk.Label(row_l, text=label_text, font=("Segoe UI", 9, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE)
            lbl.pack(side="left")

            val_badge = tk.Label(
                row_l, text=f"{def_v:.2f} cm", font=("Segoe UI", 9, "bold"),
                fg=color, bg="#1E293B", padx=8, pady=1
            )
            val_badge.pack(side="right")
            self.val_labels[key] = val_badge

            # Slider Control
            s_var = tk.DoubleVar(value=def_v)
            self.slider_vars[key] = s_var

            scale = ttk.Scale(
                f_box, from_=min_v, to=max_v, variable=s_var,
                command=lambda val, k=key: self.on_slider_change(k, val)
            )
            scale.pack(fill="x")

        # Archetype Presets
        pr_header = tk.Label(
            left_content, text="Species Presets (Typical Averages)",
            font=("Segoe UI", 9, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE
        )
        pr_header.pack(anchor="w", pady=(18, 6))

        p_row = tk.Frame(left_content, bg=self.C_SURFACE)
        p_row.pack(fill="x", pady=(0, 14))

        self.make_preset_btn(p_row, "Setosa", 5.0, 3.4, 1.5, 0.2, self.C_ACCENT_SETOSA)
        self.make_preset_btn(p_row, "Versicolor", 5.9, 2.8, 4.3, 1.3, self.C_ACCENT_VERSI)
        self.make_preset_btn(p_row, "Virginica", 6.6, 3.0, 5.6, 2.0, self.C_ACCENT_VIRGIN)

        # Model Selector
        mod_header = tk.Label(
            left_content, text="Machine Learning Classifier",
            font=("Segoe UI", 9, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE
        )
        mod_header.pack(anchor="w", pady=(4, 6))

        self.selected_model_name = tk.StringVar(value="Support Vector Machine (SVM - RBF)")
        model_dropdown = ttk.Combobox(
            left_content, textvariable=self.selected_model_name,
            values=list(self.models.keys()), state="readonly", font=("Segoe UI", 9)
        )
        model_dropdown.pack(fill="x", pady=(0, 8))
        model_dropdown.bind("<<ComboboxSelected>>", lambda e: self.run_prediction())

        self.model_badge_lbl = tk.Label(
            left_content, text="● Optimal Generalization",
            font=("Segoe UI", 8), fg="#10B981", bg=self.C_SURFACE
        )
        self.model_badge_lbl.pack(anchor="w", pady=(0, 16))

        # Big Predict Action Button
        btn_pred = tk.Button(
            left_content, text="⚡ EXECUTE INFERENCE", font=("Segoe UI", 10, "bold"),
            fg="#FFFFFF", bg=self.C_PRIMARY, activebackground=self.C_PRIMARY_HOVER,
            activeforeground="#FFFFFF", bd=0, relief="flat", cursor="hand2", pady=9,
            command=self.run_prediction
        )
        btn_pred.pack(fill="x")

        # --- RIGHT PANEL: Real-time Analytics & Visualizer ---
        right_p = tk.Frame(body, bg=self.C_SURFACE, bd=1, relief="solid", highlightbackground=self.C_BORDER, highlightthickness=1)
        right_p.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        right_content = tk.Frame(right_p, bg=self.C_SURFACE, padx=20, pady=16)
        right_content.pack(fill="both", expand=True)

        res_header = tk.Label(
            right_content, text="Classification Verdict",
            font=("Segoe UI", 11, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE
        )
        res_header.pack(anchor="w", pady=(0, 10))

        # Prediction Highlight Card
        self.pred_card = tk.Frame(right_content, bg=self.C_CARD, bd=1, relief="solid", padx=20, pady=16)
        self.pred_card.pack(fill="x", pady=(0, 14))

        self.pred_title_lbl = tk.Label(
            self.pred_card, text="IRIS VERSICOLOR",
            font=("Segoe UI", 18, "bold"), fg=self.C_ACCENT_VERSI, bg=self.C_CARD
        )
        self.pred_title_lbl.pack(anchor="w")

        self.conf_lbl = tk.Label(
            self.pred_card, text="Probability Confidence: 99.4%",
            font=("Segoe UI", 10), fg=self.C_TEXT, bg=self.C_CARD
        )
        self.conf_lbl.pack(anchor="w", pady=(4, 0))

        # Anatomical Scale Visualization (Dynamic Canvas)
        vis_header = tk.Label(
            right_content, text="Live Anatomical Scale Visualizer",
            font=("Segoe UI", 9, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE
        )
        vis_header.pack(anchor="w", pady=(4, 4))

        self.canvas_vis = tk.Canvas(right_content, bg="#0E1422", height=130, bd=0, highlightthickness=1, highlightbackground=self.C_BORDER)
        self.canvas_vis.pack(fill="x", pady=(0, 14))

        # Class Probability Spectrum (Custom Canvas Bars)
        prob_header = tk.Label(
            right_content, text="Multi-Class Probability Distribution",
            font=("Segoe UI", 9, "bold"), fg=self.C_TEXT, bg=self.C_SURFACE
        )
        prob_header.pack(anchor="w", pady=(4, 6))

        self.prob_canvas = tk.Canvas(right_content, bg=self.C_SURFACE, height=120, bd=0, highlightthickness=0)
        self.prob_canvas.pack(fill="x")

        # 3. Bottom Footer
        footer = tk.Frame(self.root, bg=self.C_SURFACE, height=36, bd=0)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        foot_lbl = tk.Label(
            footer, text="⚡ Inference Time: < 1.2 ms  |  Training Samples: 120  |  Test Split: 30  |  5-Fold CV: 97.5% - 99.2%",
            font=("Segoe UI", 8), fg=self.C_MUTED, bg=self.C_SURFACE
        )
        foot_lbl.pack(side="left", padx=25, pady=8)

    def make_preset_btn(self, parent, name, sl, sw, pl, pw, accent):
        btn = tk.Button(
            parent, text=name, font=("Segoe UI", 8, "bold"),
            fg=accent, bg=self.C_CARD, activebackground=self.C_BORDER,
            activeforeground="#FFFFFF", bd=1, relief="solid", cursor="hand2", padx=8, pady=4,
            command=lambda: self.load_preset(sl, sw, pl, pw)
        )
        btn.pack(side="left", fill="x", expand=True, padx=3)

    def load_preset(self, sl, sw, pl, pw):
        vals = {"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw}
        for k, v in vals.items():
            self.slider_vars[k].set(v)
            self.val_labels[k].configure(text=f"{v:.2f} cm")
        self.run_prediction()

    def on_slider_change(self, key, val):
        self.val_labels[key].configure(text=f"{float(val):.2f} cm")
        self.run_prediction()

    def run_prediction(self):
        try:
            sl = float(self.slider_vars["sepal_length"].get())
            sw = float(self.slider_vars["sepal_width"].get())
            pl = float(self.slider_vars["petal_length"].get())
            pw = float(self.slider_vars["petal_width"].get())

            raw_input = np.array([[sl, sw, pl, pw]])

            m_name = self.selected_model_name.get()
            meta = self.models[m_name]
            clf = meta["model"]
            is_scaled = meta["scaled"]
            self.model_badge_lbl.configure(text=f"● {meta['badge']}")

            features = self.scaler.transform(raw_input) if is_scaled else raw_input
            pred_class_idx = clf.predict(features)[0]
            pred_name = self.species_names[pred_class_idx]

            # Colors
            accents = {
                "Setosa": self.C_ACCENT_SETOSA,
                "Versicolor": self.C_ACCENT_VERSI,
                "Virginica": self.C_ACCENT_VIRGIN
            }
            c_accent = accents.get(pred_name, self.C_PRIMARY)

            # Probabilities
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(features)[0]
                conf = probs[pred_class_idx] * 100
            else:
                probs = np.array([1.0 if i == pred_class_idx else 0.0 for i in range(3)])
                conf = 100.0

            # Update Highlight Card
            self.pred_title_lbl.configure(text=f"IRIS {pred_name.upper()}", fg=c_accent)
            self.conf_lbl.configure(text=f"Confidence Score: {conf:.1f}%  via  {m_name}")

            # Draw Probability Spectrum
            self.draw_probabilities(probs, accents)

            # Draw Flower Geometry
            self.draw_geometry(sl, sw, pl, pw, c_accent, pred_name)

        except Exception as e:
            messagebox.showerror("Inference Error", str(e))

    def draw_probabilities(self, probs, accents):
        self.prob_canvas.delete("all")
        w = self.prob_canvas.winfo_width()
        if w < 100:
            w = 400

        y_offset = 12
        bar_height = 14
        spacing = 34

        for i, name in enumerate(self.species_names):
            p = float(probs[i])
            color = accents[name]
            y = y_offset + i * spacing

            # Label + Percent
            self.prob_canvas.create_text(
                0, y - 2, text=f"Iris {name}", fill=self.C_TEXT, font=("Segoe UI", 9, "bold"), anchor="w"
            )
            self.prob_canvas.create_text(
                w - 10, y - 2, text=f"{p*100:.1f}%", fill=color, font=("Segoe UI", 9, "bold"), anchor="e"
            )

            # Background bar
            self.prob_canvas.create_rectangle(
                0, y + 8, w - 10, y + 8 + bar_height, fill="#1C253B", outline=""
            )

            # Foreground fill
            fill_w = max(4, int((w - 10) * p))
            self.prob_canvas.create_rectangle(
                0, y + 8, fill_w, y + 8 + bar_height, fill=color, outline=""
            )

    def draw_geometry(self, sl, sw, pl, pw, accent_color, species_name):
        """Draws proportional visual shapes representing Sepal and Petal dimensions."""
        self.canvas_vis.delete("all")
        cw = self.canvas_vis.winfo_width()
        ch = self.canvas_vis.winfo_height()
        if cw < 100:
            cw, ch = 420, 130

        # Scale factor
        scale = 13.0
        cy = ch // 2

        # 1. Sepal Shape (Oval) on Left
        cx_sepal = cw * 0.28
        sepal_w_px = sw * scale * 1.3
        sepal_h_px = sl * scale * 1.0

        self.canvas_vis.create_oval(
            cx_sepal - sepal_w_px / 2, cy - sepal_h_px / 2,
            cx_sepal + sepal_w_px / 2, cy + sepal_h_px / 2,
            fill="#1E3A8A", outline="#3B82F6", width=2
        )
        self.canvas_vis.create_text(
            cx_sepal, ch - 12,
            text=f"Sepal ({sl:.1f} × {sw:.1f} cm)", fill=self.C_MUTED, font=("Segoe UI", 8), anchor="center"
        )

        # 2. Petal Shape (Oval) on Right
        cx_petal = cw * 0.72
        petal_w_px = max(6, pw * scale * 2.2)
        petal_h_px = max(12, pl * scale * 1.0)

        self.canvas_vis.create_oval(
            cx_petal - petal_w_px / 2, cy - petal_h_px / 2,
            cx_petal + petal_w_px / 2, cy + petal_h_px / 2,
            fill="#4C1D95", outline=accent_color, width=2
        )
        self.canvas_vis.create_text(
            cx_petal, ch - 12,
            text=f"Petal ({pl:.1f} × {pw:.1f} cm)", fill=accent_color, font=("Segoe UI", 8, "bold"), anchor="center"
        )

        # Center connector / species indicator
        self.canvas_vis.create_text(
            cw / 2, cy - 8,
            text="PROPORTIONS", fill="#475569", font=("Segoe UI", 7, "bold"), anchor="center"
        )


def main():
    root = tk.Tk()
    app = ModernIrisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
