# 🩺 Breast Cancer Prediction using Machine Learning (PROJECT BY KAKI MANISHA PRIYANKA)

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
# Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data.csv")

# Drop unnecessary columns
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Check for missing values
null_counts = df.isnull().sum()
print(null_counts)

# Display dataset information
df.info()

# Display first few rows
df.head()

## 📂 Dataset Preview

Once the dataset is loaded, here’s a preview of the first few rows:

📌 **Dataset Head (First Few Rows):**  
![Dataset Head](https://github.com/manupriya00/Breast-Cancer-Prediction/blob/main/head.PNG)


# Statistical summary
df.describe()

# Data visualization - Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Encode the target variable
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Define feature variables (X) and target variable (y)
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize machine learning models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC()
}

# Train and evaluate each model
model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_results[name] = accuracy
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    ## 📊 Model Evaluation Results

After training, we evaluate the models using accuracy, precision, recall, and F1-score.

📌 **Model Performance Summary:**  
![Model Evaluation](https://github.com/manupriya00/Breast-Cancer-Prediction/blob/main/model%20evaluation.PNG)


# Display accuracy comparison
accuracy_df = pd.DataFrame(list(model_results.items()), columns=["Model", "Accuracy"])
print("\nModel Performance:")
print(accuracy_df)

# Visualization of model accuracy
plt.figure(figsize=(8, 5))
sns.barplot(x="Accuracy", y="Model", data=accuracy_df.sort_values(by="Accuracy", ascending=False), palette="Blues_r")
plt.xlabel("Accuracy Score")
plt.ylabel("Machine Learning Model")
plt.title("Model Accuracy Comparison")
plt.show()


## 📊 Model Performance Results

Below are the accuracy scores of the different machine learning models used in this project:

| Model                  | Accuracy  |
|------------------------|-----------|
| **Logistic Regression** | **97.0%** |
| **Decision Tree**       | 93.5%     |
| **Random Forest**       | 96.0%     |
| **K-Nearest Neighbors** | 89.6%     |
| **Support Vector Machine** | 91.2%  |

From the table above, **Logistic Regression performed the best**, achieving an **accuracy of 97%**.

---

## 📌 Insights & Key Findings

- **Logistic Regression** was the most effective model for this dataset, achieving **97% accuracy**.
- **Random Forest** performed well, but **Decision Tree had a slightly lower performance** due to overfitting.
- **K-Nearest Neighbors (KNN) performed the worst**, suggesting that distance-based methods may not work well for this dataset.
- **Support Vector Machine (SVM) provided decent results**, but it requires more hyperparameter tuning for improvements.

---
## 🚀 Future Enhancements

This project can be further improved with the following enhancements:

✅ **Deep Learning Implementation:**  
   - Use **Convolutional Neural Networks (CNNs)** for image-based classification of mammograms.
   - Experiment with **Recurrent Neural Networks (RNNs)** for feature extraction.

✅ **Hyperparameter Optimization:**  
   - Implement **Grid Search and Randomized Search** for improved model tuning.
   - Use **automated ML tools like AutoML** for better parameter selection.

✅ **Deploy the Model as a Web App:**  
   - Create a **Flask or FastAPI-based application** to allow users to upload patient data and get predictions.
   - Deploy the model on **cloud platforms like AWS, Google Cloud, or Heroku**.

✅ **Expand the Dataset:**  
   - Train on larger, real-world datasets for better generalization.
   - Use synthetic data augmentation techniques to balance class distributions.

---
## ❤️ Contributing

Contributions are always welcome! Follow these steps to contribute:

1️⃣ **Fork the repository**  
2️⃣ **Create a new branch** (`git checkout -b feature-branch`)  
3️⃣ **Commit your changes** (`git commit -m "Added new feature"`)  
4️⃣ **Push to the branch** (`git push origin feature-branch`)  
5️⃣ **Submit a Pull Request** for review  

If you find any **issues or bugs**, feel free to open an **issue** in the repository.

---
## 📜 References

- [Breast Cancer Research Foundation](https://www.bcrf.org/)
- [National Cancer Institute](https://www.cancer.gov/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)

---
## ⭐!

If you found this project useful, please consider **starring the repository** ⭐.  
This helps others discover it and improves community contributions!
PROJECT BY KAKI MANISHA PRIYANKA
---
