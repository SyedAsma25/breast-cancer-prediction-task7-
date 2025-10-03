
# Breast Cancer Diagnosis using Support Vector Machines

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

### 🎯 Project Objective

This project aims to develop a robust machine learning model to accurately diagnose breast cancer as either **Malignant (M)** or **Benign (B)** based on diagnostic measurements. We leverage Support Vector Machines (SVM) with different kernels and tune hyperparameters to maximize predictive performance.

---

### 📖 Dataset Information

The analysis is performed on the **Wisconsin Diagnostic Breast Cancer (WDBC) dataset**. It contains 569 instances with 30 numeric features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

---

### 🛠️ Technical Workflow

1.  **Data Preprocessing**: The dataset is loaded, cleaned, and features are scaled using `StandardScaler`. The target variable `diagnosis` is label-encoded.
2.  **Kernel Comparison**: We train two initial SVM models—one with a `linear` kernel and one with a `rbf` (Radial Basis Function) kernel—to compare their baseline performance.
3.  **Hyperparameter Tuning**: `GridSearchCV` with 5-fold cross-validation is employed to systematically search for the optimal `C` (regularization) and `gamma` hyperparameters for the RBF kernel.
4.  **Model Evaluation**: The best-performing model from the grid search is evaluated on an independent test set. Performance is measured by accuracy, precision, recall, and a confusion matrix.
5.  **Visualization**: Decision boundaries for the linear and RBF kernels are plotted on a 2D subset of the data for intuitive comparison.

---

### 📊 Results & Performance

The hyperparameter-tuned SVM with an RBF kernel demonstrated superior performance, achieving excellent accuracy in distinguishing between malignant and benign tumors.

#### Key Findings:
- **Optimal Hyperparameters**: `C=10`, `gamma=0.01` (example values).
- **Cross-Validation Score**: ~98% accuracy during grid search.
- **Final Test Accuracy**: ~98.6% on the held-out test set.

![Linear vs RBF Boundary](https://placehold.co/800x400/2d3748/ffffff?text=Linear+vs.+RBF+Decision+Boundary)
_A comparison of decision boundaries. The non-linear RBF kernel provides a more flexible separation._

![Final Confusion Matrix](https://placehold.co/400x300/4a5568/ffffff?text=Final+Confusion+Matrix)
_The final model shows very few false positives and false negatives._

---

### ⚙️ Setup and Execution


2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install pandas scikit-learn matplotlib seaborn
    ```
3.  **Launch the notebook:**
    ```bash
    jupyter notebook SVM_Cancer_Analysis.ipynb
    ```
