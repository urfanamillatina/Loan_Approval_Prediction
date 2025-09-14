# Loan_Approval_Prediction
Machine Learning Model which predicts whether the company should approve loan application for the new applicants with deployment here: https://loan-approval-prediction-model-f2e70fd9ef88.herokuapp.com/

![Loan Approval Prediction Model](./assets/loan_gif.gif)


## Software and Tools
1. [GithubAccount](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [VSCodeIDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

### Create a New Environment


```
conda create -p ./venv python=3.7 -y
```


The data is big. The concern is with the processing time in training and computational cost. Thus, using simpler algorithms like logistig regression is not feasible. The dataset is imbalanced and skewed, thus training the model with algorithms that insensitive to data imbalanced and skewed, like decision tree, random forest, and XGboost which later hyperparameter tuned. Then, these models stacked into one model to get better performance. Later, the stacked model, decision tree, random forest and XGBoost are blended into one model which they will be weighted to get the better performance in predicting loan approval. From the business perspective, it's impartial that the model is more favourable in predicting the potential of bad loans and reject the applications rather than predicting the potential of good loans. It means that there is a possibility that the model is rejecting a potential of good loan which means the loss of potential revenue. Thus, for this model, the performance metrics to focus on would be recall which has similar score between prediction bad loan and good loan with bad loan have higher score result than the good loan. By doing this, the company is still able to reject potential bad loan and or misclassed good loan. 






