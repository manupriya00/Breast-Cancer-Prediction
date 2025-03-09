# 🩺 Breast Cancer Prediction using Machine Learning

## 📖 Project Overview
Breast cancer is one of the most common and dangerous diseases affecting millions of women worldwide. **Early detection is crucial** for improving survival rates. This project implements **Machine Learning models** to classify tumors as **benign (B)** or **malignant (M)** using a dataset of medical diagnostic features.  

This project explores **multiple classification algorithms**, compares their accuracy, and provides a robust solution for predicting breast cancer.

✅ **Best Accuracy Achieved:** **97% (Logistic Regression)**  
✅ **Dataset Used:** Wisconsin Breast Cancer Dataset (UCI ML Repository)  
✅ **Key Features Used:** Tumor cell properties like **radius, texture, perimeter, and symmetry**  
✅ **Goal:** Automate early breast cancer detection using ML  

---

## 📊 Dataset Information
The dataset used in this project is the **Wisconsin Breast Cancer Dataset**, containing **569 samples** with **30 numeric features** extracted from cell nuclei images. These features help in **distinguishing malignant from benign tumors**.

📌 **Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  

### 🔹 **Dataset Features**
The dataset includes **10 key characteristics** extracted from cell nuclei images:

- **Diagnosis**: `M` (Malignant) or `B` (Benign)  
- **Cell characteristics (mean, standard error, worst case for each):**  
  - **Radius** (mean distance from center to perimeter)  
  - **Texture** (variation in grayscale)  
  - **Perimeter, Area, Smoothness, Compactness**  
  - **Concavity & Concave Points** (severity of concave portions)  
  - **Symmetry & Fractal Dimension** (shape irregularity)  

---

## 🏗️ Project Workflow
This project follows a structured **Machine Learning pipeline**:

1️⃣ **Data Preprocessing:**  
   - Handle missing values  
   - Normalize feature values  
   - Drop unnecessary columns (`ID`, etc.)  

2️⃣ **Exploratory Data Analysis (EDA):**  
   - Visualize class distribution  
   - Check feature correlation using heatmaps  
   - Identify most important predictors  

📌 **Feature Correlation Heatmap:**  
![Heatmap](https://github.com/manupriya00/Breast-Cancer-Prediction/blob/main/heatmap.PNG)  

📌 **Pair Plot for Feature Relationships:**  
![Pair Plot](https://github.com/manupriya00/Breast-Cancer-Prediction/blob/main/pair%20plot.PNG)  

3️⃣ **Model Training & Optimization:**  
   - Train multiple ML models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN  
   - Use **Grid Search & Cross-Validation** for optimization  

4️⃣ **Model Evaluation:**  
   - Compare models based on **accuracy, precision, recall, and F1-score**  
   - Select the best-performing model  

📌 **Breast Cancer Prediction Visualization:**  
![Prediction](https://github.com/manupriya00/Breast-Cancer-Prediction/blob/main/breast%20cancer%20prediction%20image.jpg)  

📌 **Breast Cancer Prediction Using ML Flow:**  
![Workflow](https://github.com/manupriya00/Breast-Cancer-Prediction/blob/main/breast_cancer%20prediction%20using%20ML.png)  

---

## ⚡ Machine Learning Models Used
This project evaluates multiple classification models:

| Model                | Accuracy  |
|----------------------|-----------|
| **Logistic Regression** | **97%** |
| **Decision Tree**       | 93.5%   |
| **Random Forest**      | 96%     |
| **KNN**                | 89.6%   |
| **SVM**                | 91.2%   |

📌 **Conclusion:** **Logistic Regression performed best, balancing accuracy and efficiency.**

---

## 🖥️ How to Run the Project
Follow these steps to **run the project** on your local machine:

### **1️⃣ Install Required Libraries**
Ensure all dependencies are installed before running the project.

```sh
pip install numpy pandas matplotlib seaborn scikit-learn
