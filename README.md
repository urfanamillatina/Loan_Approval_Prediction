<p align="center">
<h1 align="center">Loan Approval Prediction Model
</h1>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-XGBoost-FF6600?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-Random%20Forest-228B22?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-Decision%20Tree-4682B4?logo=python&logoColor=white" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Framework-Flask-000000?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/API-Postman-FF6C37?logo=postman&logoColor=white" />
  <img src="https://img.shields.io/badge/Cloud-Heroku-430098?logo=heroku&logoColor=white" />
</p>


<p align="center">
  <img src="./assets/loan_gif.gif" alt="Watch the demo" width="800"/><br/>
  <em>Demo: Loan Approval Prediction Model </em>
</p>


<p align="center">
The deployment of live demo of Loan Approval Prediction Model is here:<br>
🚀 <a href=" https://loan-approval-prediction-model-f2e70fd9ef88.herokuapp.com/"><b>Live Demo</b></a> <br>
</p>


## What's this model about?
The Machine Learning model for Loan Approval Prediction is a supervised learning using classification algorithms. This model was trained using the variables from the dataset to determine which variables were the driving force of good loans and bad loans. If the model predicts an applicant to be a good loan, it means the loan can be approved; and if the model predicts a bad loan, then the application from this applicant should be rejected. 

## The Development Process

<p align="center">
  <img src="./assets/ML_workflow.png" alt="MLworkflow" width="800"/><br/>
  <em>Machine Learning Workflow </em>
</p>

## Tech Stack

### Core Libraries
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge&logo=tree&logoColor=white)
![SVR](https://img.shields.io/badge/Decision%20Tree-4682B4?style=for-the-badge&logo=python&logoColor=white)

### Tools
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![Postman](https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=postman&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Heroku](https://img.shields.io/badge/Heroku-430098?style=for-the-badge&logo=heroku&logoColor=white)

### IDE
![VS Code](https://img.shields.io/badge/VS%20Code-0078d7?style=for-the-badge&logo=visual-studio-code&logoColor=white)

## Business Understanding
LendingClub (LC) was a marketplace for a peer-to-peer lending company where investors (individuals/ businesses) could lend out loans to borrowers (individuals/ businesses). Like any lending institution, the company had risks of loans wouldn’t be paid by the borrowers (credit loss). Thus, the company needed to assess risks, especially for loan applications from new applicants. If the company can identify potential bad loans, it can reduce the risks of credit loss. Since the risks of credit losses is bigger compared to the revenue from good loans, the company preferred to focus more on finding the bad loans, which could lead to more stringent loan approvals. 

 The purpose of this project is to build a predictive machine learning model to predict whether an applicant can be a potential good or bad loan that will lead to its approval or rejection. The problem is, the company wanted more stringent loan approvals, and at the same time, the company needed to approve loans to generate revenue. Thus, this ML model needs to balance out the rejections vs the approvals, but still leans toward conservative approvals. 

 ## The Dataset
This model is using dataset from LendingClub which has 75 feature columns and 400+k input rows.

## Exploratory Data Analysis (EDA)

- From 75 features, 57 features were dropped due to: significant missing values, data leakage, irrelevant, uninformative for classification, high correlation >90%.
- The final features: 17 predictive features & 1 target feature (loan_status)

## Feature Engineering

- Feature transformation - converting categorical to numerical data
- One-hot-encoding to convert categorical feature to numerical feature.
- Label encoding to transformation by labelling categorical value as numeric label

## Modelling

### Target Variable: 
loan_status: Good loan (0), Bad loan (1)

### Algorithms:

This model is trained with Supervised learning, classification algorithms:
- Decision Tree Classification
- Random Forest Classification
- XGBoost Classification

### Hyperparameter Tuning:
- RandomizedSearch
- Cross-validation

### Final Model:
- Stacking model: XGBoost + DecisionTree as estimators, and RandomForest as final estimator
- Blended model: XGBoost + DecisionTree + RandomForest + Stacking Model. 
  Blended model is weighted to ensure the best performance


<p align="center">
  <img src="./assets/stacktoblend.png" alt="Stack to Blend" width="800"/><br/>
  <em>Stacked Model Combined with Three Algorithms for the Final Blended Model </em>
</p>

## Model Evaluation

### Performance metrics: 
- Recall 
- Precision 
- f1
- accuracy
- ROC-AUC
- Precision-Recall

Focus on Recall 1 (bad loan) & 0 (good loan) and Precision-Recall
Blended model is weighted to ensure the best performance

## Model Deployment
- Pickle the Blended model as the best model.
- POST API via Postman.
- Commit and push saved files to GitHub.
- Integrate with the Herokuapp to deploy the model
- Model is ready to be used here: https://loan-approval-prediction-model-f2e70fd9ef88.herokuapp.com/

<p align="center">
  <img src="./assets/deployment.png" alt="deploymentflow" width="400"/><br/>
  <em>Deployment Flow and Tools </em>
</p>




