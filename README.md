# 🩺 Breast Cancer Prediction using Machine Learning

## 📖 Overview
Breast cancer is one of the most common and life-threatening diseases among women globally. Early detection plays a crucial role in increasing survival rates. This project aims to predict whether a breast mass is **benign (B)** or **malignant (M)** using machine learning algorithms applied to digitized **Fine Needle Aspirate (FNA) images**.

This project utilizes **various classification models** to analyze tumor characteristics, identify key risk factors, and assist in early diagnosis. The dataset used consists of **multiple cell nucleus features**, providing a quantitative approach to cancer detection.

---

## 📊 Dataset Description
The dataset used in this project is based on **fine needle aspirate (FNA) cytology tests**, which extract features from breast cell nuclei. These features help in distinguishing between malignant and benign tumors.

### 🔹 **Dataset Features:**
- **Diagnosis:** Malignant (M) or Benign (B) (Target variable)
- **Ten real-valued characteristics** of the cell nucleus:
  - **Radius:** Mean distance from the center to points on the perimeter.
  - **Texture:** Standard deviation of gray-scale values.
  - **Perimeter & Area:** Size-related measurements.
  - **Smoothness & Compactness:** Uniformity and structure of the nucleus.
  - **Concavity & Concave Points:** Severity of concave portions.
  - **Symmetry & Fractal Dimension:** Shape irregularity.

📌 **Data Source:** The dataset was taken from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

---

## 🏗️ Methodology
The project involves various machine learning models to classify tumors. The steps include:

1️⃣ **Data Preprocessing:**  
   - Handling missing values.
   - Dropping unnecessary columns.
   - Normalizing the data for better model performance.

2️⃣ **Exploratory Data Analysis (EDA):**  
   - Heatmaps and correlation matrices to understand feature relationships.
   - Visualizing class distributions.

3️⃣ **Feature Selection & Engineering:**  
   - Identifying the most important features for prediction.

4️⃣ **Model Selection & Training:**  
   - Testing various classification models.
   - Hyperparameter tuning for optimization.

5️⃣ **Model Evaluation:**  
   - Comparing different algorithms using accuracy, precision, recall, and F1-score.

---

## ⚡ Machine Learning Algorithms Used
This project implements multiple ML models to predict breast cancer:

✔ **Logistic Regression**  
✔ **Decision Tree Classifier**  
✔ **Random Forest Classifier**  
✔ **K-Nearest Neighbors (KNN)**  
✔ **Support Vector Machine (SVM)**  

📌 **Best Performing Model:** **Logistic Regression** achieved the highest accuracy of **97%**.

---

## 🖥️ How to Run the Project
Follow these steps to run the project on your local machine:

### **1️⃣ Install Required Libraries**
```sh
pip install numpy pandas matplotlib seaborn scikit-learn
