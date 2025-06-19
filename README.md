
## 🌿 Plant Disease Classification using Machine Learning

### 🔍 Introduction

This project focuses on identifying plant leaf diseases using traditional machine learning techniques. Given the importance of early detection in agriculture, we built a system that classifies leaf images into four categories: **Healthy**, **Rust**, **Scab**, and **Multiple Diseases**. The goal is to assist farmers and researchers by providing an efficient and lightweight solution without needing deep learning models.

Using color histograms as handcrafted features and classifiers like Random Forest, SVM, and Gradient Boosting, we achieved high accuracy even with imbalanced classes. The entire pipeline—from image preprocessing to prediction—is built using Python and scikit-learn.

---

### 💡 What This Project Does

* Reads plant leaf images from the **Plant Pathology 2020 - FGVC7** dataset
* Extracts color histogram features using OpenCV
* Handles class imbalance using **SMOTE**
* Trains three ML models (Random Forest, SVM, and GBM)
* Performs hyperparameter tuning using GridSearchCV
* Evaluates model performance using accuracy and classification report
* Saves the model and preprocessing tools (like LabelEncoder and Scaler) using `joblib` for deployment

---

### 📌 Why This Project is Important

Plant diseases, if not detected early, can lead to major crop losses. Deep learning solutions are powerful but often require high-end hardware. This project offers an alternative by using **traditional ML with handcrafted features**, which is faster and works on lower-end systems.


