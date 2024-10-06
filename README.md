# Wine Quality Detection using Machine Learning

## Overview
This project aims to predict the quality of wine using various physicochemical properties through machine learning models. By leveraging a dataset that includes metrics like acidity, sugar content, pH level, and others, this project demonstrates the application of multiple machine learning algorithms to classify wine quality effectively.

## Dataset
The dataset used in this project is available in the file named `Wine_quality.docx`. It includes data for several attributes such as:

- **Fixed Acidity**
- **Volatile Acidity**
- **Citric Acid**
- **Residual Sugar**
- **Chlorides**
- **Free Sulfur Dioxide**
- **Total Sulfur Dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**
- **Quality (Target Variable)**

The target variable `Quality` ranges from 0 to 10, where higher values represent better wine quality.

## Project Structure
- **Data Preprocessing**: Cleaning and normalizing data to prepare it for training.
- **Exploratory Data Analysis (EDA)**: Analysis of features, correlations, and data distribution.
- **Modeling**: Implementation of various machine learning models like Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM).
- **

# Clone the repository:
git clone https://github.com/Lavin-Valechha/Wine-Quality-Detection-using-ML.git

# Navigate to the project directory:
cd Wine-Quality-Detection-using-ML

# Open the Jupyter Notebook:
jupyter notebook

Run the Wine_Quality_Detection.ipynb file to explore the data, train models, and evaluate their performance.

## Model Performance
The performance of different models is evaluated and compared using metrics like:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

The model that best balances these metrics is chosen for the final deployment.

### Results
| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 78%      | 77%       | 76%    | 76%      |
| Decision Tree          | 82%      | 80%       | 81%    | 80%      |
| **Random Forest**      | **85%**  | **84%**   | **83%**| **83%**  |
| Support Vector Machine | 80%      | 78%       | 79%    | 78%      |

**Best Model**: Random Forest Classifier with an accuracy of 85%.

## Usage
Once the notebook is run and the model is trained, you can use it to predict the quality of a wine sample. Here's how:

1. Input the following features of a wine sample:
   - Fixed Acidity
   - Volatile Acidity
   - Citric Acid
   - Residual Sugar
   - Chlorides
   - Free Sulfur Dioxide
   - Total Sulfur Dioxide
   - Density
   - pH
   - Sulphates
   - Alcohol

2. Run the prediction function with your input values:

   ```python
   # Example input data for prediction
   sample_data = {
       'fixed_acidity': 7.4,
       'volatile_acidity': 0.7,
       'citric_acid': 0.0,
       'residual_sugar': 1.9,
       'chlorides': 0.076,
       'free_sulfur_dioxide': 11.0,
       'total_sulfur_dioxide': 34.0,
       'density': 0.9978,
       'pH': 3.51,
       'sulphates': 0.56,
       'alcohol': 9.4
   }

   # Predict quality using the trained model
   predicted_quality = model.predict([list(sample_data.values())])
   print(f"Predicted Quality: {predicted_quality}")

