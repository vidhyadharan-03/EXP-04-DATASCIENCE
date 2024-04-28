# EXP-04-DATASCIENCE
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")

df.head()
df.dropna()
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/eab024ff-1177-4ecb-87f9-0966929cbf26)

```python
max_vals=np.max(np.abs(df[['Height']]))
max_vals
max_vals1=np.max(np.abs(df[['Weight']]))
max_vals1
print("Height =",max_vals)
print("Weight =",max_vals1)
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/7d305c6e-273d-41a6-89f9-f3934c99c0dd)

```python
df1=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/91d53a55-8cc8-4b25-9742-41ac89e4d146)

```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/6f3359c6-5707-4247-a5e6-1650eda8f70d)

```python
from sklearn.preprocessing import Normalizer
df2=pd.read_csv("/content/bmi.csv")
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/313249f0-168a-41b5-a9dd-fd70d8eaed85)

```python
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/c5e5175c-8382-4100-96e4-e3eb35be90db)

```python
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/cef54335-6192-4519-9713-edff506ecd24)


## FEATURE SELECTION SUING KNN CLASSIFICATION 

```python
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/e239b6de-b13b-48c2-b720-b4174aa0d1c3)

```python
data.isnull().sum()
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/6e3c1d9f-b688-433c-a3e3-86b234cb4e66)

```python
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/ec68be84-1627-486c-824a-fe20df920afe)

```python
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/81a0c733-373f-4f56-9316-3f04b966936b)

```python
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/2f9a7860-187b-4372-9adc-cfee909335a3)

```python
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/bb8f7a47-1eca-408c-8f6e-1e31b3490bde)

```python
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/529fecbf-a71c-4b72-8eea-16daa1d1185e)

```python
data2
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/76ce537b-d027-4c35-816a-efe83f87d87e)

```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/55a2e1fc-c0c6-42bd-96e6-74fe8eb9c9b3)

```python
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/669b5188-4764-46c5-a45f-2f297a437d39)

```python
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/d3937502-fc1a-4abd-8f45-3ed84f17579e)

![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/d26869c7-c71d-49e5-86dd-5706866bed9b)

![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/30bfaf8b-cba7-4f45-b282-75e0a27fe976)

```python
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/b85be5fd-4d37-413d-bc44-0aa058e7207f)

```python
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/3d147615-d159-416d-80ec-f8556303fc47)


![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/ca0bb1bf-0cbf-4fb1-934c-c924a3d91712)

![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/d570a09c-5d29-4fa5-a884-f296f4bc914d)

![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/5a92fcff-43a6-4cee-9883-d6a83b1260bb)


```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)

X=df[['Feature1','Feature3']]
y=df['Target']

selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)

selected_feature_indices = selector.get_support(indices=True)

```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/9238b0a2-81ae-4371-87db-5dc436d3d08e)

## Contigency Table
```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns

tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/402248a1-4b10-41ea-8fb1-6a2bc1bdcdf4)

![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/2fd79927-0f57-4985-b57f-3aaf75eae98a)

```python
chi2,p,_,_=chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/varshasharon/EXNO-4-DS/assets/98278161/91f1fb30-a3d8-4070-9597-606a45a0a543)


# RESULT:
       Thus, Feature selection and Feature scaling has been used on the given dataset.
