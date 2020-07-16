# Introduction
The objective in this project is to build machine learning models to classify or identify fraudulent card transactions from a given card transactions data, thus seeking to minimize the risk and loss of the business. The biggest challenge is to create a model that is very sensitive to fraud, since most transactions are legitimate, making detection difficult.<br>
## Data Description<br>
The features are scaled and the names of the features are not shown due to privacy reasons.Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Value'. The variable 'Time' contains the seconds between each transaction and the first transaction in the data set. The 'Amount' variable refers to the amount of the transaction. Feature 'Class' is the target variable with value 1 in case of fraud and 0 otherwise.<br>

**Importing Libraries:**<br>
```ruby
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```
**Load Data:**<br>
The dataset used in this project is available [here](https://www.kaggle.com/mlg-ulb/creditcardfraud/data).
```ruby
data = pd.read_csv('creditcard.csv')
data.head()
```
Inorder to check number of rows and columns in our dataset
```ruby
print(data.shape[0],data.shape[1])
```
To display the columns<br>
```ruby
print(data.shape[0],data.shape[1])
```  
```ruby
data.info()
``` 
```ruby
data.isnull().sum().max()
```
Deteremine the number of fradulent cases in dataset<br>
```ruby
print('Normal Transactions count:',data['Class'].value_counts().values[0])
print('Fraudulent Transactions count:',data['Class'].value_counts().values[1])
```
```ruby
print('Normal transactions are',(data['Class'].value_counts().values[0]/data.shape[0])*100,'% of the dataset')
print('Fraudulent transactions are',(data['Class'].value_counts().values[1]/data.shape[0])*100,'% of the dataset')
```
## Exploratory analysis<br>
### Visualization of Transaction class distribution<br>
```ruby
count_class=pd.value_counts(data['Class'],sort=True)
count_class.plot(kind='bar',rot=0)
plt.title('Transaction class distribution')
LABELS=['Normal','Fraud']
plt.xticks(range(2),LABELS)
plt.xlabel('Class')
plt.ylabel('Frequency')
```
### Visualization of Amount and Time Distribution<br>
```ruby
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
```
### Visualization of Amount and Time by class<br>
```ruby
sns.set_style("whitegrid")
sns.FacetGrid(data, hue="Class", size = 6).map(plt.scatter, "Time", "Amount").add_legend()
plt.show()
```
From the above graphs,we can conclude that the fraud transactions are evenly distributed throughout time<br>
### Get sense of the fraud and normal transaction amount<br>
```ruby
fraud=data[data['Class']==1]
normal=data[data['Class']==0]
fraud.Amount.describe()
```
```ruby
normal.Amount.describe()
```
# Normalization of data
Scaling is done to normalise the data within a particular range.<br>
Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time).<br>
RobustScaler reduces the influence of outliers.<br>
```ruby
from sklearn.preprocessing import StandardScaler,RobustScaler

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = std_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)
```
```ruby
scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)
```
Normalization is a process by which we scale values to be between specified limits, usually -1 to 1 or 0 to 1. This process is important because our machine learning models are heavily affected by differences in number size. The major difference will cause massive inaccuracies in our model. Normalization helps us to eliminate these sources of error rather than having it propagate throughout our analysis.<br>
```ruby
data = data.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = data.loc[data['Class'] == 1]
non_fraud_df = data.loc[data['Class'] == 0][:492]
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)
new_df.head()
```
### Visualization of Transaction Class Distribution after creating the subsample<br>
```ruby
count_class=pd.value_counts(new_df['Class'],sort=True)
count_class.plot(kind='bar',rot=0)
plt.title('Equal class distribution')
LABELS=['Normal','Fraud']
plt.xticks(range(2),LABELS)
plt.xlabel('Class')
```
Correlation matrix is used to check strong corellation between different variables in our dataset which helps us to determine strong linear relationships and also tells us which features are important for overall classification.<br>
```ruby
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)
plt.title('Heatmap of Correlation')
plt.show()
```
From the above graph,we have a lot of values very close to 0.There is no strong relationship between most of the v parameters(i.e from v1 to v28).there is variation in relationship between different parameters with the class.The lighter ones have the positive correlation whereas the darker ones have negative correlation.Thus, we can conclude that V10,V12,V14 and V17 are highly negatively correlated to class and V2,V4,V11 and V19 are highly positively correalted to class.<br>
# Data Cleansing<br>
We have identified the input features and the target variable so we will separate them into two objects ‘X’ and ‘y’ and draw the histogram of all the input features to see the data at a glance. The target variable which we would like to predict, is the 'Class' variable.<br>
```ruby
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)
plt.title('Heatmap of Correlation')
plt.show()
```
### Define X and y variables<br>
```ruby
x = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']
print(x.shape)
print(y.shape)
```
### Plot histograms of each parameter<br>
```ruby
x.hist(figsize = (20, 20))
plt.show()
```
## BoxPlots<br>
We will use boxplots to have a better understanding of the distribution of these features in fradulent and non fradulent transactions.<br>
