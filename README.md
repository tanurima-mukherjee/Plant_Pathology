# 🌿 Plant Disease Classification using Traditional Machine Learning

## 🔍 Overview

This project focuses on classifying plant leaf images into four categories — **Healthy**, **Rust**, **Scab**, and **Multiple Diseases** — using **traditional machine learning methods**. Instead of using deep learning, we extract **handcrafted features** (color histograms) from images and train models like **Random Forest**, **Support Vector Machine (SVM)**, and **Gradient Boosting Machines (GBM)**.

By using efficient techniques and handling class imbalance with RandomOverSampler(Manual), the final models are accurate, interpretable, and lightweight — making them suitable for real-world agricultural applications on low-resource devices.

---

## ✨ Features

- 📁 Reads and processes `.jpg` plant leaf images
- 🧠 Extracts handcrafted features using `OpenCV` (color histograms)
- 🧪 Handles **class imbalance** using **RandomOverSampler(Manual)**
- 🔧 Trains models: `Random Forest`, `SVM`, and `GBM`
- 🛠️ Hyperparameter tuning using `GridSearchCV`
- 🧾 Model evaluation with `accuracy`, `precision`, `recall`, `F1-score`
- 💾 Saves models, encoders, and scalers using `joblib` for deployment

---

## 📚 Dataset

We use the **Plant Pathology 2020 - FGVC7** dataset from Kaggle:  
🔗 [Dataset Link](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data)

- 1,821 images of plant leaves  
- Multi-class labels: `healthy`, `rust`, `scab`, `multiple_diseases`

---

## 🛠️ Technologies Used

- `Python 3`  
- `Scikit-learn`  
- `OpenCV`  
- `NumPy` & `Pandas`  
- `Matplotlib` & `Seaborn`  
- `imbalanced-learn` (`RandomOverSampler`(Manual))  
- `joblib`
---
### ⚙️ Process

The project follows a step-by-step traditional machine learning pipeline:

#### 1. **Data Preprocessing**
- **CSV Reading**: Loaded image IDs and one-hot encoded labels from the `train.csv`.
- **Label Encoding**: Converted multi-class columns (`healthy`, `rust`, `scab`, `multiple_diseases`) into a single label column.
- **Train-Test Split**: Divided the dataset into `80%` training and `20%` testing.

#### 2. **Feature Extraction**
- **Color Histogram**: Used `OpenCV` to extract `HSV` color histograms from leaf images.
- **Fixed Size**: All images resized to `128x128` to maintain consistency.
- **Histogram Binning**: Used `8x8x8` bins for H, S, and V channels and flattened the histogram into a `1D feature vector`.

#### 3. **Class Imbalance Handling**
- **Issue**: Dataset was imbalanced with very few samples for `multiple_diseases`.
- **Solution**: Applied **RandomOverSampler(Manual)** to generate synthetic examples for minority classes and balance the training set.

#### 4. **Feature Scaling**
- Used `StandardScaler` to normalize feature values before applying `SVM` or other distance-based models.

#### 5. **Model Training**
- Trained three traditional ML models:
  - **Random Forest Classifier**
  - **Support Vector Machine (SVM)**
  - **Gradient Boosting Classifier (GBM)**
- Used `GridSearchCV` to tune hyperparameters for each model.

#### 6. **Model Evaluation**
- Evaluated models using:
  - **Accuracy**
  - **Precision, Recall, F1-score**
  - **Confusion Matrix**
- **Random Forest** achieved the highest accuracy.

#### 7. **Model Saving**
- Saved the trained model, label encoder, and scaler using `joblib` for future use and deployment.

---
### 🤝 Contributing
Contributions are welcome! If you find bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

---

### 📄 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
### 📬 Contact
- 📧 **Email**: tanurimamukherjee2@gmail.com  
- 💻 **GitHub**: [tanurima-mukherjee](https://github.com/tanurima-mukherjee)
  
---
