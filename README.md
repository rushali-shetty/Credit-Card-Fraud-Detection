# Introduction
The objective in this project is to build machine learning models to classify or identify fraudulent card transactions from a given card transactions data, thus seeking to minimize the risk and loss of the business. The biggest challenge is to create a model that is very sensitive to fraud, since most transactions are legitimate, making detection difficult.
##Data Description
The features are scaled and the names of the features are not shown due to privacy reasons.Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Value'. The variable 'Time' contains the seconds between each transaction and the first transaction in the data set. The 'Amount' variable refers to the amount of the transaction. Feature 'Class' is the target variable with value 1 in case of fraud and 0 otherwise.
**Importing Libraries:**<br>
```ruby
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
```
