# **CardioPredict: A Machine Learning Approach**

see Project [live at kaggle](https://www.kaggle.com/code/muhammadwaqas630/cardiopredict-a-machine-learning-approach/notebook)

## **Overview**

This project involves developing a machine learning model that predicts whether a patient is likely to have heart disease based on various health-related features. The dataset used is from the UCI Machine Learning Repository, containing information on multiple risk factors for heart disease. The goal is to build an effective predictive model, evaluate its performance, and fine-tune it for real-world applications in the healthcare domain.

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

Heart disease is a leading cause of death worldwide. Early prediction and diagnosis are key to improving outcomes for patients. This project aims to apply machine learning techniques to predict the presence of heart disease based on several health indicators.

### **Objectives:**
- Perform exploratory data analysis (EDA) to understand the dataset.
- Preprocess the data (handle missing values, scale features, etc.).
- Build various machine learning models to predict heart disease.
- Evaluate model performance using classification metrics like accuracy, precision, recall, and F1-score.
- Tune the models for better results and compare their effectiveness.

---

## **Dataset Information**

### **Source:**
- **Dataset:** [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

### **Attributes:**
The dataset includes 14 attributes that provide various health metrics, as shown below:

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

In the EDA phase, key insights were derived from the data to guide the modeling process:

- **Distribution of Features:** Visualized distributions of age, cholesterol, and maximum heart rate using histograms and KDE plots.
- **Correlation Analysis:** Analyzed feature relationships using correlation heatmaps to identify highly correlated variables.
- **Heart Disease Distribution:** Investigated the target variable's distribution (presence or absence of heart disease) to understand class imbalance.

### **Key Insights:**
- **Age** and **Cholesterol** show notable trends with the likelihood of heart disease.
- **Chest Pain Type (CP)** has a strong correlation with the target variable.
- There was a slight imbalance in the target classes, but not significant enough to warrant oversampling or undersampling.

---

## **Data Preprocessing**

- **Missing Values:** There were a few missing values, particularly in the **Ca** and **Thal** features. Rows with missing data were removed for simplicity, as the dataset was still sufficiently large.
- **Categorical Encoding:** Encoded categorical variables such as **Sex**, **Chest Pain Type (CP)**, and **Thal** using one-hot encoding to prepare them for modeling.
- **Feature Scaling:** Applied standardization to features like **Age**, **Cholesterol**, and **Trestbps** to bring them to a comparable scale, ensuring better performance with distance-based models like KNN.

---

## **Modeling**

Several machine learning models were tested to predict heart disease:

1. **Logistic Regression:**
   - Simple, interpretable, and works well for binary classification problems.
   - Performed reasonably well but struggled with complex feature interactions.

2. **K-Nearest Neighbors (KNN):**
   - Non-parametric and useful for capturing complex patterns.
   - Performance improved after tuning the number of neighbors (k), but computation time was high.

3. **Support Vector Machine (SVM):**
   - Known for handling high-dimensional data effectively.
   - Performed better with a radial basis function (RBF) kernel but was computationally expensive.

4. **Random Forest:**
   - A robust ensemble method that reduces overfitting by averaging multiple decision trees.
   - Provided the best balance between bias and variance, making it the best performer overall.

---

## **Model Performance**

After training and testing the models, the **Random Forest** classifier achieved the highest performance. Below are the results for each model:

| **Model**                | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|--------------------------|--------------|---------------|------------|--------------|
| Logistic Regression       | 83.1%        | 84.0%         | 82.7%      | 83.3%        |
| K-Nearest Neighbors (KNN) | 81.6%        | 82.5%         | 81.2%      | 81.8%        |
| Support Vector Machine    | 84.7%        | 85.1%         | 84.3%      | 84.7%        |
| **Random Forest**         | **85.3%**    | **86.5%**     | **84.7%**  | **85.6%**    |

### **ROC and AUC:**
- The **Random Forest** model also had the highest Area Under the Curve (AUC) score, indicating excellent classification ability.

---

## **Conclusion and Future Work**

### **Conclusion:**
- The **Random Forest** model provided the most accurate predictions, balancing both precision and recall, making it suitable for heart disease prediction tasks.
- The model's performance shows promise for potential deployment in a healthcare setting, though further validation on unseen data would be necessary before real-world application.

### **Future Work:**
- **Model Tuning:** Further hyperparameter tuning using techniques such as RandomizedSearchCV could be conducted.
- **Feature Selection:** Applying advanced feature selection techniques to reduce dimensionality and improve model interpretability.
- **Deployment:** Integrating the final model into a web app or mobile interface to provide real-time predictions for healthcare professionals.

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

For any queries or collaboration opportunities, feel free to connect:

[![Email](https://img.shields.io/badge/Email-waqasliaqat630%40gmail.com-red)](mailto:waqasliaqat630@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Waqas%20Liaqat-blue)](https://www.linkedin.com/in/muhammad-waqas-liaqat/)  
[![GitHub](https://img.shields.io/badge/GitHub-Waqas%20Liaqat-black)](https://github.com/waqas-liaqat)  
[![Kaggle](https://img.shields.io/badge/Kaggle-Muhammad%20Waqas-brightblue)](https://www.kaggle.com/muhammadwaqas630)
