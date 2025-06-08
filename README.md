# ElevateLabs_Task7

## ✅ Steps Performed

### 1. Load and Prepare Dataset
- Loaded the dataset using `pandas`.
- Converted categorical labels (`B`, `M`) into binary values (0 and 1).
- Selected two numeric features for simplified 2D visualization.
- Normalized the feature values using `StandardScaler`.

### 2. Train SVM with Linear and RBF Kernels
- Trained two SVM models using:
  - **Linear kernel**
  - **RBF (Radial Basis Function) kernel**
- Used `sklearn.svm.SVC` to fit models.

### 3. Visualize Decision Boundaries
- Plotted decision boundaries for both kernels using `mlxtend.plotting.plot_decision_regions`.
- Helps visualize how the model separates malignant from benign cases.

### 4. Hyperparameter Tuning
- Used `GridSearchCV` to find the best values of:
  - `C` (regularization)
  - `gamma` (kernel coefficient for RBF)
- Evaluated combinations using 5-fold cross-validation.

### 5. Cross-Validation Performance
- Evaluated the best-performing model using 10-fold cross-validation.
- Calculated and printed accuracy scores and mean accuracy.

---

## 🧠 Key Concepts

- **SVM (Support Vector Machine)** is a powerful classifier that finds the optimal hyperplane to separate data into two categories.
- **Linear kernel** works well for linearly separable data.
- **RBF kernel** maps input space into higher dimensions for non-linear separation.
- **C (Regularization Parameter)** controls trade-off between margin and misclassification.
- **Gamma** defines influence of a single training example — higher values make model more sensitive.

---

## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MLxtend

---
