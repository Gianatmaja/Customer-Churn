# Customer-Churn
This project involves analysing churn dataset to identify customers who will potentially churn. We will begin by performing EDA on the dataset. Then, we 
will train different models on it and perform hyperparameter tuning, before concluding our findings.

### Summary
- Obtained dataset from [kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).
- Performed EDA & identified several features potentially leading to churn.
- Trained different models using regularised logistic regression, svm, and xgboost.
- Achieved 91% recall in identifying customers who churned.

Quick Links:
- [Read project online](https://nbviewer.jupyter.org/github/Gianatmaja/Customer-Churn/blob/main/Predicting%20Customer%20Churns%20.ipynb) 
*Recommended for viewing

Alternatively, these files are also available to view/ download in the repo.
- [Python file](https://github.com/Gianatmaja/Customer-Churn/blob/main/Predicting%20Customer%20Churns%20.py)
- [Jupyter notebook version](https://github.com/Gianatmaja/Customer-Churn/blob/main/Predicting%20Customer%20Churns%20.ipynb)

### Results
Some snapshots from the project can be found below:
![res1](https://github.com/Gianatmaja/Customer-Churn/blob/main/images/Screenshot%202022-10-11%20at%2010.38.18%20AM.png)

![res2](https://github.com/Gianatmaja/Customer-Churn/blob/main/images/Screenshot%202022-10-11%20at%2010.38.42%20AM.png)

### Conclusion
The problem analysed here was to identify customers with higher churn probabilities. After preprocessing and undersampling our data, we looked at 3 different models, the logistic regression, linear svm, and xgb classifier. Opting to use recall as our metric, the xgb model eventually performed best, achieving a recall of 0.91. This implies that out of all customers that are expected to churn, we can identify approximately 91% of them. This will truly come helpful to a company looking to launch a marketing campaign to retain its customers.

