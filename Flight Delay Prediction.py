#!/usr/bin/env python
# coding: utf-8

# ## Import required packages and load data

# In[1]:


# Installing imbalanced-learn
get_ipython().system(' pip install -U imbalanced-learn')


# In[2]:


conda install -c conda-forge imbalanced-learn


# In[1]:


#Regular EDA (exploratory data analysis) and plotting libraries
import math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#Package for splitting the dataset to training set and test set
from sklearn.model_selection import train_test_split

#Package for Logistic Regression model
from sklearn.linear_model import LogisticRegression

#Package for Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

#Package for Decision Tree model
from sklearn.tree import DecisionTreeClassifier

#Package to handle imbalanced dataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#Package for model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
from dmba import classificationSummary
from dmba import AIC_score


# In[2]:


# Loading the data
raw_data = pd.read_csv('FlightDelays.csv')


# ## Data preprocessing

# In[3]:


# Viewing dataframe structure
raw_data.shape


# 2201 observations of 13 columms

# In[4]:


# Running the first 10 rows
raw_data.head(10)


# In[5]:


#Counting the number of values in each column
raw_data.count()


# In[6]:


# Checking for null values
raw_data.isnull().sum()


# In[7]:


#Plotting null values in our dataset by using heatmap
sns.heatmap(raw_data.isnull())
plt.title("Empty Data")


# No missing value in our dataset

# In[8]:


# Checking datatype
raw_data.info()


# We have 6 string variables, and 7 numerical variables

# ## Dimensional reduction

# In[9]:


# Investigating all the elements whithin each Feature 

for column in raw_data: #create a loop to go through all columns in our dataset
    unique_values = np.unique(raw_data[column]) #take out the unique values
    nr_values = len(unique_values) #number of unique values
    if nr_values <= 10: #if clause to print the outcomes
        print("The number of values for feature {} is: {} -- {}".format(column, nr_values, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, nr_values))


# Because the dataset is in only in 1 month January 2014 and we have the DAY_OF_MONTH variable, we can consider removing FL_DATE. The 2 variables FL_NUM and TAIL_NUM do not seem like having any impact on our prediction models. Additionally, we do not need DISTANCE because we already have ORIGIN and DEST (we can use them to calculate distance if required). Furthermore, we will be creating a new dummy variable DELAY_DEP_TIME if the DEP_TIME (actual departure time) - CRS_DEP_TIME > 0 (YES) then assign 1, else (NO) give 0 accordingly.

# In[10]:


#Creating new DELAY_DEP_TIME column
raw_data['DELAY_DEP_TIME'] = raw_data['DEP_TIME'] - raw_data['CRS_DEP_TIME']
raw_data.loc[raw_data['DELAY_DEP_TIME'] > 0, 'DELAY_DEP_TIME'] = 1
raw_data.loc[raw_data['DELAY_DEP_TIME'] <= 0, 'DELAY_DEP_TIME'] = 0


# In[11]:


#Droping unnecessary columns FL_DATE, FL_NUM, TAIL_NUM,DEP_TIME in the dataset
raw_data.drop(['FL_DATE', 'FL_NUM', 'TAIL_NUM','DEP_TIME','DISTANCE'], axis=1, inplace=True)


# In[12]:


# Viewing dataset
raw_data


# In[13]:


#Renaming column names
raw_data.rename(columns={'Weather': 'WEATHER', 'Flight Status': 'FLIGHT_STATUS', 'DAY_OF_MONTH': 'DAY_MONTH'}, inplace=True)


# In[14]:


# Viewing dataset
raw_data


# In[15]:


#Listing column names
raw_data.columns


# In[16]:


#Creating hourly bins departure time (original data has 100's of categories) so bining is a musthave to buildup prediction models
raw_data.CRS_DEP_TIME = [round(t / 100) for t in raw_data.CRS_DEP_TIME]


# In[17]:


# Viewing dataset
raw_data


# In[18]:


#Rearranging column order
raw_data = raw_data[['CRS_DEP_TIME','DELAY_DEP_TIME', 'CARRIER', 'DEST', 'ORIGIN', 'WEATHER', 'DAY_WEEK',
       'DAY_MONTH', 'FLIGHT_STATUS']]


# In[19]:


# Viewing dataset
raw_data


# In[20]:


#Exporting to csv file
raw_data.to_csv(r'E:\Downloads\FlightDelaysTrainingData.csv', index=False)


# ## Data Exploration

# In[21]:


# Investigating all the elements whithin each Feature 

for column in raw_data:
    unique_values = np.unique(raw_data[column])
    nr_values = len(unique_values)
    if nr_values <= 10:
        print("The number of values for feature {} is: {} -- {}".format(column, nr_values, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, nr_values))


# In[22]:


# Investigating the distribution of outcome variable FLIGHT_STATUS
sns.countplot(x = 'FLIGHT_STATUS', data = raw_data, palette = 'Set1')


# We can see that the outcome is definitely imbalanced between 'ontime' and 'delay'. Class lable 'ontime' has abnormally high number of observations comparing to class lable 'delayed' (around 5 times). We're gonna solve this problem later to better the models performance.

# In[23]:


# Looping through all the features by our outcome variable - see if there is a relationship between predictors and outcome

features = ['CRS_DEP_TIME','DELAY_DEP_TIME', 'CARRIER', 'DEST', 'ORIGIN', 'WEATHER', 'DAY_WEEK', 'DAY_MONTH']

for f in features:
    sns.countplot(x = f, data = raw_data, palette = 'Set1', hue = 'FLIGHT_STATUS')
    plt.show()


# When we compare the countplot of each feature with the distribution of outcome variable FLIGHT_STATUS. According to the shape of the distribution, we can guess that 'CRS_DEP_TIME', 'CARRIER', 'DEST', 'ORIGIN', 'DAY_WEEK' can have greater impact on flight delay prediction.

# In[24]:


# Compare FLIGHT_STATUS with DAY_WEEK
pd.crosstab(raw_data.DAY_WEEK, raw_data.FLIGHT_STATUS)


# Monday and Friday have the most flilghts delayed, on the other hand, Saturday has the least

# In[25]:





# In[26]:


# Compare FLIGHT_STATUS with WEATHER
pd.crosstab(raw_data.WEATHER, raw_data.FLIGHT_STATUS)


# In[27]:


# Compare FLIGHT_STATUS with CRS_DEP_TIME
pd.crosstab(raw_data.CRS_DEP_TIME, raw_data.FLIGHT_STATUS)


# In[28]:


# Compare FLIGHT_STATUS with ORIGIN
pd.crosstab(raw_data.ORIGIN, raw_data.FLIGHT_STATUS)


# In[29]:


# Compare FLIGHT_STATUS with DEST
pd.crosstab(raw_data.DEST, raw_data.FLIGHT_STATUS)


# ## Data transformation

# In[30]:


# Converting categorical variables into numeric variables
new_raw_data = pd.get_dummies(raw_data, columns = features)


# In[31]:


raw_data.shape, new_raw_data.shape


# In[32]:


# Converting outcome variable to binary type
new_raw_data['FLIGHT_STATUS'][new_raw_data['FLIGHT_STATUS'] == 'delayed'] = 1
new_raw_data['FLIGHT_STATUS'][new_raw_data['FLIGHT_STATUS'] == 'ontime'] = 0


# In[33]:


# Viewing the dataset
new_raw_data


# In[34]:


new_raw_data.info()


# ## Conduct feature importance

# In[35]:


#Creating X and y data matrices (X = predictor variables, y = outcome variable)
X=new_raw_data.drop(labels=['FLIGHT_STATUS'], axis=1)
y=new_raw_data['FLIGHT_STATUS']


# In[36]:


#Ensuring int datatype of X and y 
y = y.astype(int)
X = X.astype(int)


# In[37]:


X.shape, y.shape


# In[38]:


#Handleing imbalanced data by using RandomOverSampler
ros = RandomOverSampler(sampling_strategy=1, random_state = 1) # sampling_strategy=1 means 50% for each class
X_res, y_res = ros.fit_resample(X, y)


# In[39]:


#Plotting the outcome of RandomOverSampler
ax = y_res.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling")


# In[40]:


X_res.shape, y_res.shape


# 3546 rows of 73 columns

# In[41]:


# Running a Tree-based estimators (i.e. decision trees & random forests)
dt = DecisionTreeClassifier(random_state=1, criterion = 'entropy', max_depth = 10)
dt.fit(X_res,y_res)


# In[42]:


# Running Feature Importance

fi_col = []
fi = []

for i,column in enumerate(new_raw_data.drop('FLIGHT_STATUS', axis = 1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])


# In[43]:


# Creating a Dataframe for Feature Importance
fi_col
fi

fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns = ['Feature','Feature_Importance'])
fi_df


# In[44]:


# Ordering the feature importance data
fi_df = fi_df.sort_values('Feature_Importance', ascending = False).reset_index()


# In[45]:


#Viewing the feature importance dataframe
fi_df


# In[46]:


#Filtering only feature_importance >0
fi_df = fi_df[fi_df['Feature_Importance'] > 0]


# In[47]:


# Creating list of columns to build up the prediction model
columns_to_keep = fi_df['Feature']


# In[48]:


columns_to_keep


# ## Data partition

# In[49]:


#Creating new X and y data matrices based on list of columns from feature importance 
#(X = predictor variables, y = outcome variable)
X=X_res[columns_to_keep]
y=y_res


# In[50]:


#Ensuring int datatype of X and y 
y = y.astype(int)
X = X.astype(int)


# In[51]:


X.shape,y.shape


# In[52]:


#Splitting the dataset into training set and test set, size = 0.4
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)


# In[53]:


train_X.shape, valid_X.shape, train_y.shape, valid_y.shape


# In[54]:


train_X.columns


# ## Logistic Regression model

# In[55]:


# Fitting a logistic regression model
model1 = LogisticRegression(random_state=1,solver = 'lbfgs')
model1.fit(train_X, train_y)


# In[56]:


#Printing model's coefficients model1
print('Intercept:', model1.intercept_)
print(pd.DataFrame({'Predictor': train_X.columns, 'Coefficients': model1.coef_[0]}))


# In[57]:


# Accuracy on Train
print("The Training Accuracy is: ", model1.score(train_X, train_y))

# Accuracy on Test
print("The Testing Accuracy is: ", model1.score(valid_X, valid_y))


# In[58]:


# training confusion matrix
classificationSummary(train_y, model1.predict(train_X))


# In[59]:


# validation confusion matrix
classificationSummary(valid_y, model1.predict(valid_X))


# In[60]:


# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Prediction')


# In[61]:


# Plotting Confusion Matrix

cm1 = confusion_matrix(valid_y, model1.predict(valid_X))
cm1_norm = cm1 / cm1.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm1_norm, classes = model1.classes_, title='Confusion matrix')


# In[62]:


cm1


# In[98]:


cm1.sum(axis=0)


# In[99]:


cm1.sum(axis=1)


# In[100]:


np.diag(cm1)


# In[101]:


cm1.sum()


# In[63]:


# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)
FP1 = cm1.sum(axis=0) - np.diag(cm1)
FN1 = cm1.sum(axis=1) - np.diag(cm1)
TP1 = np.diag(cm1)
TN1 = cm1.sum() - (FP1 + FN1 + TP1)


# In[64]:


FP1, FN1, TP1, TN1


# In[65]:


# True positive rate
TPR1 = TP1 / (TP1 + FN1)
print("The True Positive Rate is:", TPR1)

# Precision rate
PPV1 = TP1 / (TP1 + FP1)
print("The Precision is:", PPV1)

# False positive rate
FPR1 = FP1 / (FP1 + TN1)
print("The False positive rate is:", FPR1)


# ## Naïve Bayes model

# In[66]:


# Fitting a Naïve Bayes model
model2 = MultinomialNB(alpha=0.01)
model2.fit(train_X, train_y)


# In[67]:


# Accuracy on Train
print("The Training Accuracy is: ", model2.score(train_X, train_y))

# Accuracy on Test
print("The Testing Accuracy is: ", model2.score(valid_X, valid_y))


# In[68]:


# training confusion matrix
classificationSummary(train_y, model2.predict(train_X))


# In[69]:


# validation confusion matrix
classificationSummary(valid_y, model2.predict(valid_X))


# In[70]:


# Plotting Confusion Matrix

cm2 = confusion_matrix(valid_y, model2.predict(valid_X))
cm2_norm = cm2 / cm2.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm2_norm, classes = model2.classes_, title='Confusion matrix')


# In[71]:


cm2


# In[72]:


# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)
FP2 = cm2.sum(axis=0) - np.diag(cm2)
FN2 = cm2.sum(axis=1) - np.diag(cm2)
TP2 = np.diag(cm2)
TN2 = cm2.sum() - (FP2 + FN2 + TP2)


# In[73]:


FP2, FN2, TP2, TN2


# In[74]:


# True positive rate
TPR2 = TP2 / (TP2 + FN2)
print("The True Positive Rate is:", TPR2)

# Precision rate
PPV2 = TP2 / (TP2 + FP2)
print("The Precision is:", PPV2)

# False positive rate
FPR2 = FP2 / (FP2 + TN2)
print("The False positive rate is:", FPR2)


# ## Decision Tree model

# In[75]:


# Fitting a decision tree model
model3 = DecisionTreeClassifier(random_state=1, max_depth=1, criterion='gini')
model3.fit(train_X, train_y)


# In[76]:


# Accuracy on Train
print("The Training Accuracy is: ", model3.score(train_X, train_y))

# Accuracy on Test
print("The Testing Accuracy is: ", model3.score(valid_X, valid_y))


# In[77]:


# training confusion matrix
classificationSummary(train_y, model3.predict(train_X))


# In[78]:


# validation confusion matrix
classificationSummary(valid_y, model3.predict(valid_X))


# In[79]:


# Plotting Confusion Matrix

cm3 = confusion_matrix(valid_y, model3.predict(valid_X))
cm3_norm = cm3 / cm3.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm3_norm, classes = model3.classes_, title='Confusion matrix')


# In[80]:


cm3


# In[81]:


# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)
FP3 = cm3.sum(axis=0) - np.diag(cm3)
FN3 = cm3.sum(axis=1) - np.diag(cm3)
TP3 = np.diag(cm3)
TN3 = cm3.sum() - (FP3 + FN3 + TP3)


# In[82]:


FP3, FN3,TP3, TN3


# In[83]:


# True positive rate
TPR3 = TP3 / (TP3 + FN3)
print("The True Positive Rate is:", TPR3)

# Precision rate
PPV3 = TP3 / (TP3 + FP3)
print("The Precision is:", PPV3)

# False positive rate
FPR3 = FP3 / (FP3 + TN3)
print("The False positive rate is:", FPR3)


# ## Compare model performance

# Based on confusion matrix and True positive rate, Precision rate, False positive rate, model1 which is built by using logistic regression model is the optimal model for flight delay prediction.

# ## Use Testing Data

# In[84]:


# Loading the test data
test_data = pd.read_csv('FlightDelaysTestingData.csv')


# In[85]:


#Viewing dataset
test_data


# In[86]:


#Transforming test_data variables to a dataframe of dummy variables
new_test_data = pd.get_dummies(test_data, columns = features)


# In[87]:


#Removing outcome variable
new_test_data.drop(['FLIGHT_STATUS'], axis=1, inplace=True)


# In[88]:


#Viewing dataset
test_data


# In[89]:


#Merging X_test_data to valid_X
X_test_data = pd.DataFrame(valid_X.append(new_test_data))


# In[90]:


#Viewing dataset
X_test_data


# In[91]:


#Viewing column names
X_test_data.columns


# In[92]:


# Dropping columns not in the model1
X_test_data.drop(['DELAY_DEP_TIME_0', 'CARRIER_AA', 'CARRIER_EV', 'ORIGIN_DCA',
       'WEATHER_1', 'DAY_MONTH_9', 'DAY_MONTH_10'], axis=1, inplace=True)


# In[93]:


#Keeping only the 5 new observations
X_test_data = pd.DataFrame(X_test_data.tail(5))


# In[94]:


# Replacing nan values by 0
X_test_data = X_test_data.replace(np.nan, 0)
X_test_data = X_test_data.astype(int)


# In[95]:


X_test_data.info()


# In[96]:


X_test_data


# In[97]:


#Using the optimal model to predict X_test_data
model1.predict(X_test_data)


# Outcome 'delay', 'delay', 'ontime', 'delay', 'ontime'

# In[ ]:




