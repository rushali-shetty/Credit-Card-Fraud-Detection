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
##Exploratory analysis<br>
###Visualization of Transaction class distribution<br>
```ruby
count_class=pd.value_counts(data['Class'],sort=True)
count_class.plot(kind='bar',rot=0)
plt.title('Transaction class distribution')
LABELS=['Normal','Fraud']
plt.xticks(range(2),LABELS)
plt.xlabel('Class')
plt.ylabel('Frequency')
```
###Visualization of Amount and Time Distribution<br>
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
###Visualization of Amount and Time by class<br>
```ruby
sns.set_style("whitegrid")
sns.FacetGrid(data, hue="Class", size = 6).map(plt.scatter, "Time", "Amount").add_legend()
plt.show()
```
From the above graphs,we can conclude that the fraud transactions are evenly distributed throughout time<br>
###Get sense of the fraud and normal transaction amount<br>
```ruby
fraud=data[data['Class']==1]
normal=data[data['Class']==0]
fraud.Amount.describe()
```
