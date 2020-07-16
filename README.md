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
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZIAAAESCAYAAADXMlMiAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAb2klEQVR4nO3de5RcZZnv8W83wQ5qElFBdARjEB8bAZUGwi1DxoARo4ODOsNCFFEREAYZdYQRkIs4IiojEQUH0IDAyAiiAiLR4QhJuGQsQMOx5hFQ1KNHTRhzQaY7JOnzx959KJu+VNxd3enu72etXqvqrXfv/eza3fXr991Vu9p6e3uRJOnP1T7WBUiSxjeDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZVMGesCNHFFxELgL8u7uwI/B/6nvL9fZv7PgAuOgYjYG3hPZh4fEXsBp2XmW1u8zV5gu8xc1crtNGzvw8BumfmuiLgc+Fpmfn+I/pcBl2ZmbYDHLge+BjwMPJiZz97MWhYAszPzYxHx18DBmXny5qxDWw6DRC3T+MIQEY8Cb8/MH45ZQUN7JfBigLLGlobIWMvM9zbR7RDgS0MtHxEz/8wS9gaeW67r28C3/8z1aAtgkGhMRMTZwH7Ai4AfAR+ieNF6AbAD8AvgbzPz92UILQLmATsBV2XmmRHxbOArwC7AJqAGHFdu4l+AfYFpQBvw3sxcVi7zeeAAYAPwTeAS4FxgRkR8BbgSuDgzd4uIGcAXgFcDvcCtwEczc0NEdAPnA68DXghckJmXDLCvs4GFwLOA9cCHM/P2hsefVdawC/A8YB1wZGZmRBwOnFHu30bgHzPzzsHa+21363K7hwC/B34HrCkf+wFwcbn/fc/Hk8DPgGOAfyqPzTUR8U7gU8B/A68oa31LufwPgfZyhNJVruPkzLynPMbPz8yTym2eDTwf+CpwPLBVRKwBHgLemplvjIgXl+ufWR63KzPz02Vg/QfwHWA2sC3wkcy8sf/zrdHnORKNpZcAr8nMo4AjgLszcz9gFvAE8I6Gvs/OzDnA/sCHI+KlwN8A0zLz1RT/4VIuO5viRXC/zNyVIhhOKx8/F5gKdFKEwwHAzsDHgCWZeUy/GhcCjwG7A3sBrwI+XD7WAazKzP0pRjD/EhFTGxcuX8y/CZybmbsBxwIXRUTj396hwOrM3C8zXw78J3BS+dingfdn5l7AmcDcYdobvR94OcW04iEUIdzffuWyr8rMLoog2SMzTwd+QzGKvLfs+4fM3DUzP99vHdsA38vM11CE29cj4hkDbAuAcn2XAteV22l0DfC/MnN3imNzVEQcUT42C7gtM/ehOJ6fG2wbGl0GicbSPZm5ASAzLwLuiogPAl8EdgMa592/Vfb7NcV/188FlgKvLP+7Pg34XGY+nJl3U7ygHRcRn6F4ke9b18HAFZm5MTPXZ+ZBmfmDIWo8lGJ00puZPRQvgIf2rwu4jyJYntVv+d2BjZl5S1l/LTN3z8xNfR0y83pgUUT8fURcRPHC3lfv14Aby//4twUuGKa90cHAteV+/pHiRbq/FRQjmnsj4uPADZl51yDPxZJB2ldn5nXlviwu214xSN9BlSOzAyhGgGTmGoqRaN/z/STFiASK5/u5m7sNtYZBorH0eN+NiPgUxWhhJfCvwGKKqY0+jSfme4G2zPw58DLgk8B04PsR8abyRO4tZd9vUbz4961rQ7l833Z3jIjnDVFje2P/8v7W/evKzL4+jTU/bXvlNneLiCkN908ArqAYhV0L/Fvfesr/2A+kmEJ6F3DnUO0DaKxnQ/8HM3M1T42yNgLXRcT7B1nX44O0b+x3v53iRb+33/YHHaU0LNf/+Wt8vtc3BHD/dWsMGSTaUsynGFF8lWLEcQiw1VALlC/AXwEWZ+apwG3AnuWyN5XnK34IvLlhXd8Hjo6I9ojoAK4HDqJ4kd2ap7sNOCki2sr+7wO+txn7lUBvRBxS1rwncDt/+rc3H1iUmVeU/d9Ecf5gSnl+6JmZeSnFVNUeEdExWHu/bd8KvDMippZTbn/Xv7iIeCPFuYe7MvNs4CqemiYc7Dnp73nleoiIN1GE60MU/xR0lc/dNOCNDcs8bd2ZuQ64BzixXNcM4J1s3vOtMWCQaEtxLvCZiPgxxTt4llKMNoZyFUVA/CQiasAMinMalwJzI2IFxRTII8BLy/MS51Cc8P4RcD/wncz8BsUL2KyI+Ea/bZwMbE8xBbSC4oX+E83uVDkddjhwVkQ8UNZ2eGaub+j2GYppuB9TTB/dB7ysnPY7Bbg2Iu4Dvg68u1znYO2NvkQRpA8Cd1C8/bq/W4H/DTwYET+kOAd1TvnYN4CrI+J1w+zm74G3lPv3T8BbytqvoQiTh4Cbyxr63A7Mj4j+51veDswrj93ysoZFw2xfY6zNy8hLkqpwRCJJqsQgkSRVYpBIkioxSCRJlUzKS6Q88MADvR0d/d8pqT9XT08PPp/aEvm7ObKeeOKJVV1dXdv1b5+UQdLR0UFnZ+dYlzFh1Ot1n09tkfzdHFm1Wu0XA7U7tSVJqsQgkSRVYpBIkioxSCRJlRgkkqRKDBJJUiUGiSSpEoNEklSJQSJJqsQg2YJ1P9n/G0y3TOPlk8Pj5fmUxptJeYmU8WLq1lsx87Rbhu+opjx6/oKxLkGakByRSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlU0Z6hRGxNfBlYCbQAZwH/B/gJuChstslmXldRBwLHAdsAM7LzJsjYhvgamB7YB1wdGaujIh9gYvKvosz85xye2cBC8r2UzJz+UjvkyRpcCMeJMBRwGOZ+Y6IeB5wP3AucGFmfravU0TsAJwM7AVMBZZGxPeAE4AVmXl2RBwBnAF8ALgUeAvwM+CWiNizXNVBwGxgR+AGYO8W7JMkaRCtCJKvA9c33N8AdAEREYdRjEpOAfYBlmVmD9ATEQ8DewAHAheUy94KnBkR04GOzHyEYkW3AfOAHorRSS/wy4iYEhHbZebKFuyXJGkAIx4kmfk4QERMowiUMyimuC7PzFpEnA6cBTwArGlYdB0wA5je0N7YtrZf31lAN/DYAOsYMkh6enqo1+t/zu6Nqs7OzrEuYcIZD8ddI6e7u9tjPgpaMSIhInYEbgS+mJnXRsRzMnN1+fCNwOeBO4FpDYtNA1ZTBMa0Idoa29cP0j6kjo4OX6QnKY/75FKv1z3mI6hWqw3YPuLv2oqIFwCLgVMz88tl820RsU95ex5QA5YDcyJiakTMADqBB4FlwBvKvocCSzJzLbA+InaOiDZgPrCk7Ds/ItojYiegPTNXjfQ+SZIG14oRyUeBbSnObZxZtn0Q+FxErAd+C7wvM9dGxEKKQGgHTs/M7oi4BLgyIpZSjDiOLNdxPHANsBXFeZF7ASJiCXB3uY4TW7A/kqQhtPX29o51DaOuXq/3jpfh7szTbhnrEiaMR89fMNYlaJQ5tTWyarVaraura6/+7X4gUZJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqmTLSK4yIrYEvAzOBDuA84CfAIqAXeBA4MTM3RcSxwHHABuC8zLw5IrYBrga2B9YBR2fmyojYF7io7Ls4M88pt3cWsKBsPyUzl4/0PkmSBteKEclRwGOZOQc4FLgYuBA4o2xrAw6LiB2Ak4EDgPnAJyOiAzgBWFH2vQo4o1zvpcCRwIHA7IjYMyL2BA4CZgNHAF9owf5IkobQiiD5OnBmw/0NQBdwR3n/VuBgYB9gWWb2ZOYa4GFgD4qg+G5j34iYDnRk5iOZ2QvcBswr+y7OzN7M/CUwJSK2a8E+SZIGMeJTW5n5OEBETAOupxhRfKYMACimq2YA04E1DYsO1N7YtrZf31lAN/DYAOtYOVSNPT091Ov1zd21UdfZ2TnWJUw44+G4a+R0d3d7zEfBiAcJQETsCNwIfDEzr42ICxoengaspgiGacO0D9d3/SDtQ+ro6PBFepLyuE8u9XrdYz6CarXagO0jPrUVES8AFgOnZuaXy+b7I2JueftQYAmwHJgTEVMjYgbQSXEifhnwhsa+mbkWWB8RO0dEG8U5lSVl3/kR0R4ROwHtmblqpPdJkjS4VoxIPgpsC5wZEX3nSj4ALIyIZwB14PrM3BgRCykCoR04PTO7I+IS4MqIWEox4jiyXMfxwDXAVhTnRe4FiIglwN3lOk5swf5IkobQ1tvbO3yvCaZer/eOl+HuzNNuGesSJoxHz18w1iVolDm1NbJqtVqtq6trr/7tfiBRklSJQSJJqsQgkSRVYpBIkioxSCRJlRgkkqRKDBJJUiUGiSSpEoNEklSJQSJJqsQgkSRV0lSQlFf0lSTpaZq9+u8NEbESuAL4TmZuamFNkqRxpKkRSWYeSHF5+IOAuyLiExExq6WVSZLGhc05R/Ib4GfAE8BuwEURcW5LqpIkjRvNniP5d4ovj9oWOCozD8vMN/HUNxlKkiapZkcklwH7ZOY/A43fhHXgyJckSRpPmg2S/YFzytsLI+I0gMzsbklVkqRxo9kg+evM/BBAZr4NeFPrSpIkjSfNBsmmiHgGQERsvRnLSZImuGY/R3Ip8GBErABeAVzQupIkSeNJU0GSmVdExLeBWcAjmbmqtWVJksaLpoIkIl4NvA+YWt4nM9/dysIkSeNDs1Nbi4CLgV+1rhRJ0njUbJD8NjMvb2klkqRxqdkgebT87Mj9lB9IzMzFLatKkjRuNBskHUCUP1CEiUEiSWr6XVvHRMTLgZ2BFRQXcJQkqel3bZ0E/A3wXIoT77sAJ7WuLEnSeNHs1NYRwBzg9sy8KCL+c7gFImI28KnMnBsRewI3AQ+VD1+SmddFxLHAccAG4LzMvDkitgGuBrYH1gFHZ+bKiNgXuKjsuzgzzym3cxawoGw/JTOXN7lPkqQR0GyQ9F0Spe/Kvz1DdY6IjwDvAP5YNu0JXJiZn23oswNwMrAXxedTlkbE94ATgBWZeXZEHAGcAXyA4tP1b6H4TpRbynCC4su2ZgM7AjcAeze5T5KkEdDsNbOuBe4EXhYR3wG+OUz/R4DDG+53AQsi4s6IuCIipgH7AMsysycz1wAPA3tQXJr+u+VytwIHR8R0oCMzH8nMXuA2YF7Zd3Fm9mbmL4EpEbFdk/skSRoBzZ5svzgi/oPimxEzM388TP8bImJmQ9Ny4PLMrEXE6cBZwAPAmoY+64AZwPSG9sa2tf36zgK6gccGWMfKoerr6emhXq8P1WWL0NnZOdYlTDjj4bhr5HR3d3vMR0GzJ9s/1nC3MyLenJmb8zW7N2bm6r7bwOcpRjjTGvpMA1ZTBMa0Idoa29cP0j6kjo4OX6QnKY/75FKv1z3mI6hWqw3Y3uzU1u/Kn98DLwZ22szt3xYR+5S35wE1ilHKnIiYGhEzgE7gQWAZT32F76HAksxcC6yPiJ0jog2YDywp+86PiPaI2Alo94KSkjS6mp3a+lLj/Yi4dTO3cwJwcUSsB34LvC8z10bEQopAaAdOz8zuiLgEuDIillKMOI4s13E8cA2wFcV5kXvLWpZQfJ98O3DiZtYlSaqo2amtlzfcfSFNjEgy81Fg3/L2fRRf19u/z2UU3wff2PYE8LYB+t7Tt75+7WcDZw9XjySpNZp9+2/jiKQb+HALapEkjUPNTm39VasLkSSNT81Obf2I4h1R3ZRfbgW0Ab2ZOatFtUmSxoFm37V1F/D2zNwVOAxYSvHd7b6vTpImuWbPkeyamXcDZOaKiNgpM4e8TIokaXJoNkhWR8THKT77cSDwi9aVJEkaT5qd2jqS4tPlr6e4aOJ7WlaRJGlcaTZIuoE/AKuABJ7TsookSeNKs0HyJYoPIb6O4t1bV7WsIknSuNJskOycmR8DujPzJoor7EqS1HSQTImI5wO95XeJbGphTZKkcaTZd22dTnGl3RcC91B8Y6EkSU2PSHbMzAB2BnbLzO+3sCZJ0jjS7IjkfcA1mTnkNw9KkiafZoOkIyLup3jr7yaAzDxy6EUkSZPBkEESEWdk5nnAqcBfAL8elaokSePGcCOS1wLnZeYdEXF7Zr52NIqSJI0fw51sbxvktiRJwPBB0jvIbUmSgOGntroi4i6K0ciuDbd7M/Np38EuSZp8hguSPUalCknSuDVkkGSm3zsiSRpSs59slyRpQAaJJKkSg0SSVIlBIkmqxCCRJFVikEiSKmn26r+bLSJmA5/KzLkR8TJgEcWn4x8ETszMTRFxLHAcsIHiml43R8Q2wNXA9sA64OjMXBkR+wIXlX0XZ+Y55XbOAhaU7adk5vJW7ZMk6elaMiKJiI8AlwNTy6YLgTMycw7FJ+MPi4gdgJOBA4D5wCcjogM4AVhR9r0KOKNcx6XAkcCBwOyI2DMi9gQOAmYDRwBfaMX+SJIG16qprUeAwxvudwF3lLdvBQ4G9gGWZWZPZq4BHqb4JP2BwHcb+0bEdKAjMx/JzF7gNmBe2XdxZvZm5i8pvlt+uxbtkyRpAC2Z2srMGyJiZkNTWxkAUExXzQCmA2sa+gzU3ti2tl/fWUA38NgA6xjymxx7enqo1+ubsUdjo7Ozc6xLmHDGw3HXyOnu7vaYj4KWnSPpZ1PD7WnAaopgmDZM+3B91w/SPqSOjg5fpCcpj/vkUq/XPeYjqFarDdg+Wu/auj8i5pa3DwWWAMuBORExNSJmAJ0UJ+KXAW9o7JuZa4H1EbFzRLRRnFNZUvadHxHtEbET0J6Zq0ZpnyRJjN6I5EPAZRHxDKAOXJ+ZGyNiIUUgtAOnZ2Z3RFwCXBkRSylGHH3fDX88cA2wFcV5kXsBImIJcHe5jhNHaX8kSaW23t7J931V9Xq9d7wMd2eedstYlzBhPHr+grEuQaPMqa2RVavVal1dXXv1b/cDiZKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSarEIJEkVWKQSJIqMUgkSZUYJJKkSgwSSVIlBokkqRKDRJJUiUEiSapkymhuLCLuB9aUd38OfAJYBPQCDwInZuamiDgWOA7YAJyXmTdHxDbA1cD2wDrg6MxcGRH7AheVfRdn5jmjuU+SNNmN2ogkIqYCZObc8ucY4ELgjMycA7QBh0XEDsDJwAHAfOCTEdEBnACsKPteBZxRrvpS4EjgQGB2ROw5WvskSRrdEcmrgGdGxOJyux8FuoA7ysdvBV4HbASWZWYP0BMRDwN7UATFBQ19z4yI6UBHZj4CEBG3AfOA+4YqpKenh3q9PpL71hKdnZ1jXcKEMx6Ou0ZOd3e3x3wUjGaQPAF8Brgc2IUiDNoys7d8fB0wA5jOU9Nfg7U3tq3t13fWcIV0dHT4Ij1Jedwnl3q97jEfQbVabcD20QySnwIPl8Hx04h4jGJE0mcasJoiGKYN0z5cX0nSKBnNd229G/gsQES8iGI0sTgi5paPHwosAZYDcyJiakTMADopTsQvA97Q2Dcz1wLrI2LniGijOKeyZJT2R5LE6I5IrgAWRcRSindpvRtYBVwWEc8A6sD1mbkxIhZSBEI7cHpmdkfEJcCV5fLrKU6wAxwPXANsRfGurXtHcZ8kadIbtSDJzMYX/0YHDdD3MuCyfm1PAG8boO89wL4jVKYkaTP5gURJUiUGiSSpEoNEklSJQSJJqsQgkSRVYpBIkioxSCRJlRgkkqRKDBJJUiUGiSSpEoNEklSJQSJJqsQgkSRVYpBIkioxSCRJlRgkkqRKDBJJUiUGiSSpEoNEklSJQSJJqsQgkSRVYpBIkioxSCRJlRgkkqRKDBJJUiUGiSSpEoNEklSJQSJJqmTKWBcwEiKiHfgi8CqgB3hvZj48tlVJ0uQwUUYkbwamZuZ+wGnAZ8e4HkmaNCZKkBwIfBcgM+8B9hrbcqSJrfvJjWNdQlM6OzvHuoSmjJfnczATYmoLmA6sabi/MSKmZOaGgTo/8cQTq2q12i9Gp7RqbnjbDmNdwoRRq9XGugRpvHvJQI0TJUjWAtMa7rcPFiIAXV1d27W+JEmaHCbK1NYy4A0AEbEvsGJsy5GkyWOijEhuBA6JiLuANuCYMa5HkiaNtt7e3rGuQZI0jk2UqS1J0hgxSCRJlRgkkqRKJsrJdlUQEXOBbwK7Z+avyrbzgf/KzEUt2uZM4GuZuW8r1q+Jqfy9+TFwX0Pz7Zl57gis+/XAEZn5rqrrmmwMEvVZD3wlIg7JTN+BoS3ZTzJz7lgXoacYJOpzO8VU54nAxX2NEfEh4AhgA3BnZp4aEWcD+wPPBt4DLAJ+BcwEvgbsBrwGuCUzPxoRBwFnlat8JvBOiuCSKitH1J+i+J36V+B/KH6P28oub6X4nTw+M48ol/ltZu4QEZ3Al4E/lj9/GN3qJwbPkajRCcA/RMQu5f1pwN9ShMb+wC4R8cbysXpm7k/xRzuLIlDeCHwc+CAwu2wDeCVwVGa+Fvg28LZR2BdNXLtGxA/6foC/oLho65zM/CrwcmBBOWpJYP4Q6/o48LHMPBi4q8V1T1iOSPT/ZeZjEXEKxQhjGTAVuCcznwSIiCUUoQDFH2ifn2XmmojoAX6Xmf9d9u+bIvs1sDAiHqf4o1/W8p3RRPYnU1vliKTx9/H3wJXl79srgLsHWEffaOWVwPLy9jJgfFzlcQvjiER/IjNvovijfBfQDcyOiCkR0Qb8JfDTsuumhsWGO6dyOXBMeRLzNzz1RyyNlE0AETEDOIdiOva9FCPmNorf5ReWfV4CPLdc7r+A/crbe49ivROKIxIN5BRgHrAO+HeK/9TagaUU7+561Wau76vAvRHxB+B3wItGrlTpT6yl+H29j6fOebyI4ndwdUTcC9SBn5f93w9cFxH/CKykCBxtJi+RIkmqxKktSVIlBokkqRKDRJJUiUEiSarEIJEkVeLbf6UWi4hXAhdQXB7m2cB3gB8Ax/VdskMazxyRSC0UEc+huP7YKZn5V8C+wO5AjGlh0ghyRCK11mEUlzl/CCAzN0bEOymuXTYXICJOAg4HtgbWlLdnUlyq5kmKC2b2XejyOop/ALemuAjhitHbFWlgjkik1noR8LPGhsx8nPLqxxHRDjwPODgz51AExN7AIUANOBj4BLAtsA9F0BwKnAxMH51dkIZmkEit9Qtgx8aGiHgpxXXLyMxNFKHybxFxBfBiijC5AlgFfBc4iWJUcitwB/At4Fz+9Hpn0pgxSKTWuhl4fUTsDBARWwMXUoQEEbEH8ObM/Dvg7yn+JtsopsSWZOY84OvAqRRTYf83M18HnAf88+juijQwr7UltVhEdAGfpgiJacBNFCOL44B3U4TNdKCn/LkCuAe4mmIksgn4B4rRzXXAs4CNwLmZuXg090UaiEEiSarEqS1JUiUGiSSpEoNEklSJQSJJqsQgkSRVYpBIkioxSCRJlfw/cbaQYEkWH5UAAAAASUVORK5CYII=%0A" width="500" height="300">
<br>
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
### Visualization of correlations using boxplot<br>
```ruby
f,ax=plt.subplots(2,2,figsize=(24,14))
f.suptitle('Features with high negative correlation',size=20)
sns.boxplot(x='Class',y='V10',data=new_df,ax=ax[0,0])
sns.boxplot(x='Class',y='V12',data=new_df,ax=ax[0,1])
sns.boxplot(x='Class',y='V14',data=new_df,ax=ax[1,0])
sns.boxplot(x='Class',y='V17',data=new_df,ax=ax[1,1])
```
```ruby
f,ax=plt.subplots(2,2,figsize=(24,14))
f.suptitle('Features with high positive correlation',size=20)
sns.boxplot(x='Class',y='V2',data=new_df,ax=ax[0,0])
sns.boxplot(x='Class',y='V4',data=new_df,ax=ax[0,1])
sns.boxplot(x='Class',y='V11',data=new_df,ax=ax[1,0])
sns.boxplot(x='Class',y='V19',data=new_df,ax=ax[1,1])
```
Remove the extreme outliers from features that have a high correlation with our classes.<br>
```ruby
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
v14_iqr = q75 - q25
v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)


v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)



v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
```
# Classifier<br>
An algorithm that maps the input data to a specific category.Classification is a type of supervised learning. The training data is used to make sure the machine recognizes patterns in the data and the test data is used only to access performance of model.<br>
```ruby
X=new_df.drop('Class',axis=1) 
y=new_df['Class']
```
```ruby
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
```
## Model Architecture

 ### Import the required classifiers
```ruby
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```
#### Logistic Regression<br>
```ruby
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```
#### Support Vector Classifier<br>
```ruby
svc=SVC()
svc.fit(X_train,y_train)
```
#### K-nearest neighbors<br>
```ruby
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
```
#### Random Forest Classifier<br>
```ruby
RDF_Classifier=RandomForestClassifier(random_state=0)
RDF_Classifier.fit(X_train,y_train)
```
#### DecisionTreeClassifier<br>
```ruby
DecisionTreeClassifier= DecisionTreeClassifier()
DecisionTreeClassifier.fit(X_train,y_train)
```
### Model Evaluation and Prediction<br>
 ```ruby
models_list=[('Logistic Regression',logmodel),('SVC',svc),('KNeighborsClassifier',knn),('RFC',RDF_Classifier),('DecisionTreeClassifier',DecisionTreeClassifier)]
models=[j for j in models_list]
print()
#print('===========================Model Evaluation Results================================')
for i,v in models:
      print('==========================={}=========================================='.format(i))
      a=cross_val_score(v, X_train, y_train, cv=5)
      print('Cross validation score=',a.mean())
```
### Test Models
```ruby
models_list=[('Logistic Regression',logmodel),('SVC',svc),('KNeighborsClassifier',knn),('RFC',RDF_Classifier),('DecisionTreeClassifier',DecisionTreeClassifier)]
models=[j for j in models_list]
print()
print('===========================Model Test Results================================')
for i,v in models:
      print('==========================={}=========================================='.format(i))
      pred_test = v.predict(X_test)
      print('Accuracy =',accuracy_score(y_test,pred_test))
      print('Confusion Matrix')
      print(confusion_matrix(y_test,pred_test))
      print('Classification Report')
      print(classification_report(y_test,pred_test))
```
