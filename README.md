# **CardioPredict: A Machine Learning Approach**

[View Project on Kaggle](https://www.kaggle.com/code/muhammadwaqas630/cardiopredict-a-machine-learning-approach/notebook)

## **Overview**

This project develops a machine learning model to predict the severity of heart disease in patients based on a range of health-related features. Using the UCI Machine Learning Repository's heart disease dataset, we aim to build an effective predictive model, evaluate its performance, and optimize it for real-world applications in healthcare.

---

## **Table of Contents**
1. [Project Introduction](#project-introduction)
2. [Dataset Information](#dataset-information)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Model Performance](#model-performance)
7. [Conclusion and Future Work](#conclusion-and-future-work)
8. [Technologies Used](#technologies-used)
9. [Contact](#contact)

---

## **Project Introduction**

Heart disease is a leading cause of death worldwide, making early prediction and diagnosis essential for improving patient outcomes. This project utilizes machine learning techniques to predict heart disease presence based on several health indicators.

### **Objectives:**
- Conduct exploratory data analysis (EDA) to understand data trends.
- Preprocess data by handling missing values and scaling features.
- Build and evaluate various machine learning models.
- Fine-tune models to optimize predictive accuracy for healthcare use.

---

## **Dataset Information**

### **Source:**
- **Dataset:** [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

### **Attributes:**
The dataset includes 14 attributes detailing health metrics, including:

| **Feature**      | **Description**                                                                 |
|------------------|---------------------------------------------------------------------------------|
| Age              | Age of the patient                                                              |
| Sex              | 1 = Male, 0 = Female                                                            |
| CP               | Chest pain type (1-4 categories)                                                |
| Trestbps         | Resting blood pressure (in mm Hg)                                               |
| Chol             | Serum cholesterol (in mg/dl)                                                    |
| Fbs              | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)                           |
| Restecg          | Resting electrocardiographic results (values 0, 1, 2)                           |
| Thalach          | Maximum heart rate achieved                                                     |
| Exang            | Exercise-induced angina (1 = Yes, 0 = No)                                       |
| Oldpeak          | ST depression induced by exercise relative to rest                              |
| Slope            | Slope of the peak exercise ST segment (0-2 categories)                          |
| Ca               | Number of major vessels (0-3) colored by fluoroscopy                            |
| Thal             | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect                             |
| Target           | 1 = Disease, 0 = No Disease (Dependent variable for prediction)                 |

---

## **Exploratory Data Analysis**

During the EDA phase, insights were gathered to guide the modeling process:

- **Distribution of Features:** Visualizations of age, cholesterol, and max heart rate helped reveal key patterns.
- **Correlation Analysis:** Correlation heatmaps highlighted relationships between variables, notably **CP** and the target.
- **Heart Disease Distribution:** Class distribution analysis helped assess balance in target classes.

### **Key Insights:**
- **Age** and **Cholesterol** trends were linked to heart disease likelihood.
- **Chest Pain Type (CP)** strongly correlated with disease presence.
- A slight imbalance in target classes was noted but was not significant enough to require resampling.

---

## **Data Preprocessing**

- **Missing Values:** Rows with missing values in **Ca** and **Thal** were removed due to sufficient dataset size.
- **Categorical Encoding:** Variables like **Sex**, **CP**, and **Thal** were encoded using one-hot encoding.
- **Feature Scaling:** Standardization applied to numerical features like **Age** and **Cholesterol** for better model performance.

---

## **Modeling**

Several machine learning models were evaluated to predict heart disease:

1. **Logistic Regression**: Simple and interpretable; provided moderate accuracy.
2. **K-Nearest Neighbors (KNN)**: Captured complex patterns but was computationally intensive.
3. **Support Vector Machine (SVM)**: Effective in high-dimensional data, especially with RBF kernel.
4. **Random Forest**: Balanced bias and variance, performing robustly across metrics.
5. **Decision Tree**: High accuracy, especially for simpler, interpretable decision-making tasks.

---

## **Model Performance**

Each model was tested using cross-validation and evaluated for accuracy on the test set. Below are the performance metrics:

| **Model**                  | **Cross-validation Accuracy** | **Test Accuracy**            |
|----------------------------|------------------------------|------------------------------|
| Logistic Regression        | 0.5134                       | 0.6739                       |
| SVM                        | 0.5527                       | 0.6413                       |
| Decision Tree Classifier   | 0.4764                       | **0.9130**                   |
| Random Forest Classifier   | 0.5298                       | 0.7500                       |
| K-Nearest Neighbors        | 0.5538                       | 0.6250                       |
| Gradient Boosting Classifier| 0.4623                      | 0.7554                       |
| XGBClassifier              | 0.4633                       | 0.8750                       |
| AdaBoost Classifier        | 0.5287                       | 0.6739                       |

### **Best Model:**
The **Decision Tree Classifier** achieved the highest test accuracy (91.3%), making it the most effective model for this dataset.

### **CPU Performance**
- **Total CPU time:** 7 minutes 8 seconds
- **Wall time:** 4 minutes 42 seconds

### **ROC and AUC:**
The **Decision Tree** model also demonstrated strong classification capability with high AUC.

---

## **Conclusion and Future Work**

### **Conclusion:**
- The **Decision Tree** model offered the best predictive accuracy, making it ideal for healthcare applications in heart disease prediction.
- While robust, further testing is needed to confirm reliability before potential deployment.

### **Future Work:**
- **Hyperparameter Tuning:** Consider RandomizedSearchCV for further optimization.
- **Feature Selection:** Use advanced selection methods for interpretability and dimensionality reduction.
- **Deployment:** Integrate the model into an accessible app or platform for real-time healthcare applications.

---

## **Technologies Used**

- **Programming Language:** Python 3.12
- **Libraries:**
  - Pandas (data manipulation)
  - Numpy (numerical computations)
  - Scikit-learn (machine learning models)
  - Matplotlib & Seaborn (data visualization)
  - Jupyter Notebook (development environment)

---

## **Contact**

For questions or collaboration opportunities, reach out:

[![Email](https://img.shields.io/badge/Email-waqasliaqat630%40gmail.com-red)](mailto:waqasliaqat630@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Waqas%20Liaqat-blue)](https://www.linkedin.com/in/muhammad-waqas-liaqat/)  
[![GitHub](https://img.shields.io/badge/GitHub-Waqas%20Liaqat-black)](https://github.com/waqas-liaqat)  
[![Kaggle](https://img.shields.io/badge/Kaggle-Muhammad%20Waqas-brightblue)](https://www.kaggle.com/muhammadwaqas630)

---
