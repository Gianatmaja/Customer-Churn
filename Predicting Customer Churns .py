#!/usr/bin/env python
# coding: utf-8

# # Predicting Churn Rates

# ## Introduction

# Churn rates, in a business sense, provide a measure of how many customers stop doing business/ stop subscribing to a product. As this would mean the inability to retain revenue, companies would want to minimise the churn rate. One way of doing so would be to identify customers with a high potential to churn, and try to retain them by providing promos, discounts, etc.
# 
# Today, we're going to look at this [dataset](https://www.kaggle.com/blastchar/telco-customer-churn) from kaggle and see if we can apply machine learning to identify customers with a potential to churn.

# ### Importing the data

# In[1]:


import pandas as pd
import numpy as np

Data = pd.read_csv('TelcoChurn.csv')
Data.head(5)


# The dataset has 7043 rows and 21 columns, with the following names & types.

# In[2]:


Data.columns


# In[3]:


Data.dtypes


# Some description of the dataset can be found below:
# - customerID: ID of customer
# - gender: Gender of customer
# - SeniorCitizen: Whether the customer is a senior citizen (1 for yes and 0 for no)
# - Partner: Whether the customer has a partner (Yes or No)
# - Dependents: Whether the customer has dependent(s) (Yes or No)
# - tenure: Customer's tenure in months
# - PhoneService: Whether the customer has a phone service (Yes or No)
# - MultipleLines: Whether the customer has multiple phone lines (Yes, No, or No phone service)
# - InternetService: The customer's internet service (DSL, Fiber optic, or no)
# - OnlineSecurity: Whether the customer has online security (Yes, No, or No internet service)
# - OnlineBackup: Whether the customer has online backup (Yes, No, or No internet service)
# - DeviceProtection: Whether the customer has device protection (Yes, No, or No internet service)
# - TechSupport: Whether the customer has tech support (Yes, No, or No internet service)
# - StreamingTV: Whether the customer has streaming TV (Yes, No, or No internet service)
# - StreamingMovies: Whether the customer streams movie (Yes, No, or No internet service)
# - Contract: Contract term of the customer (Month-to-month, One year, Two year)
# - PaperlessBilling: Whether the customer has paperless billing (Yes or No)
# - PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# - MonthlyCharges: The customer's monthly charges
# - TotalCharges: The customer's total charges
# - Churn: Whether the customer churned (Yes or No)

# Interestingly, we observe that the 'TotalCharges' column has the type 'object'. This seems inappropriate as they are numerical values. So, we'll convert it.

# In[4]:


Data['TotalCharges'] = pd.to_numeric(Data['TotalCharges'], errors = 'coerce')


# Then, we check if there are any missing values in the dataset.

# In[5]:


Data.isna().sum()


# As there are 7043 rows and only 11 missing values, we can simply drop those values and use the other complete rows.

# In[6]:


Data.dropna(inplace = True)


# Now, let's do some checkings on our dataset. The 'customerID' column must contain unique values. Let's check if there are any duplicates.

# In[7]:


Unique_vals = ['customerID']

for features in Unique_vals:
    Col = Data[features]
    Length = len(Col)
    Unique = len(set(Col))
    if (Length/Unique) == 1:
        print('Values in the column {} are unique. No duplicates found.\n'.format(features))
    else:
        s = Train[features].value_counts()
        print('Values in the column {} are not unique. Duplicates and frequencies:'.format(features))
        print(s.where(s>1).dropna())
        print('\n')


# Then, we'll observe the other columns.

# In[8]:


Categorical_vals = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 
                    'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod', 'Churn']

for features in Categorical_vals:
    print('Values for column {}: '.format(features), set(Data[features]),', Length: ',len(set(Data[features])),'\n')


# In[9]:


Integer_vals = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

print('Summary for numerical features: \n', Data[Integer_vals].describe()) 


# ## Exploratory Data Analysis

# Now, let's perform some exploratory data analysis. We'll first look at the numerical columns.

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

colors = ['springgreen', 'violet', 'orange']

i = 0
for feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    
    fig = plt.figure(figsize = (30,5))
    
    ax = fig.add_subplot(1, 3, 1)
    sns.distplot(Data[feature], color = colors[i])
    plt.grid()
    plt.title('Histogram of {}'.format(feature))
    
    ax = fig.add_subplot(1,3,2)
    sns.boxplot(Data[feature], color = colors[i])
    plt.grid()
    plt.title('Boxplot of {}'.format(feature))
    
    Churn_df = Data[Data['Churn'] == 'Yes']
    No_churn_df = Data[Data['Churn'] == 'No']

    ax = fig.add_subplot(1,3,3)
    sns.distplot(No_churn_df[feature], color = 'skyblue', label = 'No churn')
    sns.distplot(Churn_df[feature], color = 'red', label = 'churn')
    plt.title('Histogram of {} wrt Churn'.format(feature))
    plt.grid()
    plt.legend()
    
    i = i + 1
    
    plt.show()


# Observations:
# - Customers who have stayed longer were less likely to churn.
# - Customers with higher monthly charges are more likely to churn. But interestingly, customers who paid higher total charges were less likely to churn.

# In[11]:


plt.figure(figsize = (10, 20))

sns.pairplot(Data, vars = ['tenure', 'MonthlyCharges', 'TotalCharges'], hue = 'Churn')
plt.show()


# Observations:
# - There is some correlation between tenure and total charges.
# - There is some correlation between total charges and monthly charges.

# Now, let's take a look at the categorical variables.

# In[12]:


Cats = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling','PaymentMethod', 'Churn']

for features in Cats:
    fig = plt.figure(figsize = (15,4))
    
    ax = fig.add_subplot(1, 2, 1)
    sns.countplot(x = features, data = Data, palette = 'pastel')
    plt.title('Barplot of {}'.format(features))
    plt.grid()
    
    ax = fig.add_subplot(1, 2, 2)
    sns.countplot(x = features, hue = 'Churn', data = Data, palette = 'Set2')
    plt.title('Barplot of {}, wrt Churn'.format(features))
    plt.grid()
    plt.show()


# Observations:
# - Male and female customers were equally likely to churn.
# - Senior citizens, and customers without partner or dependents appear to be more likely to churn.
# - Among other internet service, fiber optic customers were most likely to churn.
# - Customers without added internet features (online security/ backup, device protection, tech support) were more likely to churn.
# - Customers on month to month contract were more likely to churn.
# - Customers paying with electronic check were more likely to churn.
# - The dataset is imbalanced.

# Finally, let's do some violinplots to see if we can observe any other trends.

# In[13]:


fig = plt.figure(figsize=(12, 7))
sns.violinplot(x = 'OnlineSecurity', y = 'TotalCharges', hue = 'SeniorCitizen',
               data = Data, split = True, palette = 'mako')
plt.grid()
plt.show()


# In[14]:


fig = plt.figure(figsize=(12, 7))
sns.violinplot(x = 'Partner', y = 'TotalCharges', hue = 'Dependents',
               data = Data, split = True, palette = 'mako')
plt.grid()
plt.show()


# In[15]:


fig = plt.figure(figsize=(12, 7))
sns.violinplot(x = 'InternetService', y = 'TotalCharges', hue = 'Churn',
               data = Data, split = True, palette = 'mako')
plt.grid()
plt.show()


# In[16]:


fig = plt.figure(figsize=(12, 7))
sns.violinplot(x = 'StreamingMovies', y = 'TotalCharges', hue = 'Churn',
               data = Data, split = True, palette = 'mako')
plt.grid()
plt.show()


# In[17]:


fig = plt.figure(figsize=(12, 7))
sns.violinplot(x = 'PaymentMethod', y = 'TotalCharges', hue = 'Churn',
               data = Data, split = True, palette = 'mako')
plt.grid()
plt.show()


# Some more findings:
# - Customers paying for additional features like streaming movies on a higher fee were more likely to churn.
# - Customers with higher total charges prefer bank transfer or credit card payment.

# ## Modelling

# Now, we're done with the EDA and can move to the next step, which is the modelling part. First, we'll drop the customer ID column and preprocess the data.

# In[18]:


Data.drop('customerID', axis = 1, inplace = True)


# For categorical columns with only 2 input options, we'll use label encoder to preprocess them. Whereas for those with more than 2 options, we'll use one hot encoding. The reason for this is because we wouldn't want to create a misinterpretation where two inputs are more similar than the others.

# In[19]:


from sklearn.preprocessing import LabelEncoder

LE_Features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for feature in LE_Features:
    Data[feature] = LabelEncoder().fit_transform(Data[feature])
    
#gender: female = 0, male = 1
#Partner: no = 0, yes = 1
#Dependents: no = 0, yes = 1
#Phone service: no = 0, yes = 1
#PaperlessBilling: no = 0, yes = 1
#Churn: No = 0, yes = 1


# In[20]:


from sklearn.preprocessing import OneHotEncoder

OHE_Features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

for feature in OHE_Features:
    Dummy = pd.get_dummies(Data[feature], prefix = feature)
    Data = pd.concat([Data, Dummy], axis = 1)
    Data.drop(feature, axis = 1, inplace = True)

Data.head(5)


# Done with the categorical columns, we'll now seperate the target values (in this case the churn column), from the other features. Next, we'll split the data into training and test set and standardise the numerical columns with respect to the values in the training set.

# In[21]:


y = Data['Churn']
X = Data.drop('Churn', axis = 1)


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)


# In[23]:


pd.options.mode.chained_assignment = None

Standardised_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

for feature in Standardised_features:
    Min = X_train[feature].min()
    Max = X_train[feature].max()
    
    X_train[feature] = (X_train[feature] - Min)/(Max - Min)
    X_test[feature] = (X_test[feature] - Min)/(Max - Min)


# In[24]:


X_train.head(5)


# At this point, all the columns have been preprocessed and the data is ready for model fitting. But before that, we just have one more thing to do. Recall that previously, we observed that our data is imbalanced. To prevent having a model that would be biased towards the majority class, we should undersample/ oversample our dataset. In reality, oversampling might result in a better performance since all information is retained. However, here, we've decided to undersample our data instead to reduce the computation time.

# In[25]:


from imblearn.under_sampling import RandomUnderSampler

X_res, y_res = RandomUnderSampler(random_state = 1).fit_resample(X_train, y_train)


# We'll first use a logistic regression on the data. It is a simple and in most cases, a realiable algorithm.

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Log_model = LogisticRegression(max_iter = 1000).fit(X_res, y_res)
y_pred_lr = Log_model.predict(X_test)
accuracy_score(y_test, y_pred_lr)


# Looking at the accuracy, one might think that this model performs quite well. However, recalling the imbalanced dataset problem, we should reconsider whether accuracy should be our go-to scoring metric in this case. Perhaps, in this case, recall or precision might be a better metric. We'll first use recall.

# In[27]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_lr))


# Fortunately, it does seem that the logistic regression model performs quite well, scoring a recall of 0.84. Now let's see if we can further improve the model by tuning the hyperparameter.

# ### Hyperparameter tuning for recall

# In[28]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold

Log_model = LogisticRegression(max_iter = 1000)

grid_values = {'C':[0.001,0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty': ['l2']}

cross_validation = StratifiedKFold(n_splits = 5)

grid_log_model = GridSearchCV(Log_model, param_grid = grid_values, scoring = 'recall', cv = cross_validation)
grid_log_model.fit(X_res, y_res)


# In[29]:


grid_log_model.best_params_


# In[30]:


from sklearn.metrics import classification_report

y_pred_lr = grid_log_model.predict(X_test)
print(classification_report(y_test, y_pred_lr))


# Using hyperparameter tuning, we managed to improve recall to 0.85 by using a c parameter (regularisation) of 0.001 with an l2 penalty (ridge).
# 
# Now, let's see if we can get an even better model using other classifiers.

# In[31]:


# Your code here
from sklearn.svm import SVC

Svm_model = SVC()

grid_values2 = {'kernel': ['linear'], 'C':[0.001, 0.01, 0.1, 1, 10, 100]}
cross_validation = StratifiedKFold(n_splits = 5)

grid_svm_model = GridSearchCV(Svm_model, param_grid = grid_values2, scoring = 'recall', cv = cross_validation)
grid_svm_model.fit(X_res, y_res)

res = np.array(grid_svm_model.cv_results_['mean_test_score'].reshape(6,1))


# In[32]:


y_pred_svm = grid_svm_model.predict(X_test)
print(classification_report(y_test, y_pred_svm))


# Great! Using a linear svm, we managed to further improve the recall to 0.89. This means that we're able to identify 89% of the customers with a potential to churn, and can launch a marketing campaign to try and retain them.
# 
# Finally, let's try one more algorithm, a gradient boosted model using XGBoost.

# In[33]:


from xgboost import XGBClassifier

grid_values3 = {'max_depth': [1, 2, 3, 4, 5, 6],
                'n_estimators' : [2, 3, 5, 10, 15, 20]}

gbm = XGBClassifier()

cross_validation = StratifiedKFold(n_splits = 5)

grid_xgb_model = GridSearchCV(gbm, param_grid = grid_values3,scoring = "recall", cv = cross_validation)
grid_xgb_model.fit(X_res, y_res)

print("Best parameters found: ", grid_xgb_model.best_params_)


# In[34]:


from sklearn.metrics import classification_report

y_pred_xgb = grid_xgb_model.predict(X_test)
print(classification_report(y_test, y_pred_xgb))


# This model performs even better, scoring a recall of 0.91. Clearly, this model outperforms the other 2 models and should be used to find customers with a potential to churn.

# ### The case for precision

# Previously, we used recall as the scoring metric for our learning algorithms. This is fine, if the company wants to find as many customers with the potential to churn. However, if the company wants to be as accurate as possible when identifying churn-potential customers, then the above steps can be repeated, with the scoring metric changed to "precision". 

# ## Conclusion

# The problem analysed here was to identify customers with higher churn probabilities. After preprocessing and undersampling our data, we looked at 3 different models, the logistic regression, linear svm, and xgb classifier. Opting to use recall as our metric, the xgb model eventually performed best, achieving a recall of 0.91. This implies that out of all customers that are expected to churn, we can identify approximately 91% of them. This will truly come helpful to a company looking to launch a marketing campaign to retain its customers.
