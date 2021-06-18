
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

!pip install -q kaggle

from google.colab import files
files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d corrieaar/apartment-rental-offers-in-germany

!kaggle datasets download -d iabhishekofficial/mobile-price-classification

!ls

!unzip '/content/apartment-rental-offers-in-germany.zip' -d '/content/'

!unzip '/content/mobile-price-classification.zip' -d '/content/'

train3 = pd.read_csv('/content/train.csv')
train3.head()

data = pd.read_csv('/content/immo_data.csv')
data.head()

data.info()

data.isna().sum()

mask = ((data.isna().sum()/len(data)) > 0.5)
data.columns[mask]

data = data.drop(columns=data.columns[mask])
data.columns

data.columns.size

data = data.drop(columns=['scoutId','heatingType','newlyConst','yearConstructed'
                          ,'firingTypes','yearConstructedRange','houseNumber','street'
                          ,'lift','description','facilities','date'])
data.shape

data._get_numeric_data().mean()

data.fillna(data._get_numeric_data().mean(),inplace=True)

data.isna().sum()

for col in data.columns :
  if data[col].dtype == 'int64' or data[col].dtype == 'float64':
    up = data[col].mean() + 3*data[col].std()
    low = data[col].mean() - 3*data[col].std()
    mask = (data[col] > up ) | (data[col] < low )
    data = data.drop(data[mask].index)

y = data['livingSpace']

y.head()

for col in data.columns :
  if (data[col].dtype == 'int64' or data[col].dtype == 'float64') and (col !='livingSpace'):
    data[col] = (data[col] - min(data[col])) /(max(data[col]) - min(data[col]))

data.head()

print(data.shape)
print(y.shape)

for col in data.columns :
  if data[col].dtype == 'object' or data[col].dtype == 'bool':
    data[col] = data[col].fillna(data[col].value_counts().head(1).index[0])

data.isna().sum()

for col in data.columns :
  if data[col].dtype == 'object' or data[col].dtype == 'bool':
    print(col)
    print(data[col].value_counts())
    print('_______________________________')

data['telekomTvOffer'].value_counts()

others = list(data['telekomTvOffer'].value_counts().tail(2).index)
def edit_telekomTvOffer(a):
  if a in others:
    return 'others'
  return a
data['telekomTvOffer'] = data['telekomTvOffer'].apply(edit_telekomTvOffer)
data['telekomTvOffer'].value_counts()

print(data['condition'].value_counts())

others = list(data['condition'].value_counts().tail(3).index)
def edit_condition(a):
  if a in others:
    return 'others'
  return a
data['condition'] = data['condition'].apply(edit_condition)
data['condition'].value_counts()

print(data['interiorQual'].value_counts())

others = list(data['interiorQual'].value_counts().tail(2).index)
def edit_interiorQual(a):
  if a in others:
    return 'others'
  return a
data['interiorQual'] = data['interiorQual'].apply(edit_interiorQual)
data['interiorQual'].value_counts()

data['geo_krs'].unique().size

others = list(data['geo_krs'].value_counts().tail(400).index)
def edit_geo_krs(a):
  if a in others:
    return 'others'
  return a
data['geo_krs'] = data['geo_krs'].apply(edit_geo_krs)
data['geo_krs'].value_counts()

print(data['regio3'].unique().size)
print(data['regio2'].unique().size)
print(data['streetPlain'].unique().size)

data = data.drop(columns=['regio2','regio3','streetPlain'])
data.shape

y.head()

cor_matrix = data.corr()

f , ax = plt.subplots(figsize=(25,30))
sns.heatmap(cor_matrix,square=True,annot=True)

cate_features = []

for col in data.columns :
  if data[col].dtype == 'object' or data[col].dtype == 'bool':
    cate_features.append(col)

cate_features

dum_features = pd.get_dummies(data[cate_features])
dum_features.head()

data = pd.concat([data,dum_features],axis=1)
data.head()

data = data.drop(columns=cate_features)
data.head()

data.shape

x = data.drop(columns=['livingSpace'])
print(x.shape)
print(y.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#  BAKHSHE 1 HALATE 1

df1 = data.iloc[:50000,:]
df2 = data.iloc[50000:100000,:]
df3 = data.iloc[100000:150000,:]
df4 = data.iloc[150000:200000,:]
df5 = data.iloc[200000:250000,:]

dfnot1 = pd.concat([df2,df3,df4,df5])
dfnot2 = pd.concat([df1,df3,df4,df5])
dfnot3 = pd.concat([df1,df2,df4,df5])
dfnot4 = pd.concat([df1,df2,df3,df5])
dfnot5 = pd.concat([df1,df2,df3,df4])

xnot1 = dfnot1['livingSpaceRange']
xnot2 = dfnot2['livingSpaceRange']
xnot3 = dfnot3['livingSpaceRange']
xnot4 = dfnot4['livingSpaceRange']
xnot5 = dfnot5['livingSpaceRange']

ynot1 = dfnot1['livingSpace']
ynot2 = dfnot2['livingSpace']
ynot3 = dfnot3['livingSpace']
ynot4 = dfnot4['livingSpace']
ynot5 = dfnot5['livingSpace']

xtest1 = df1['livingSpaceRange']
xtest2 = df2['livingSpaceRange']
xtest3 = df3['livingSpaceRange']
xtest4 = df4['livingSpaceRange']
xtest5 = df5['livingSpaceRange']

ytest1 = df1['livingSpace']
ytest2 = df2['livingSpace']
ytest3 = df3['livingSpace']
ytest4 = df4['livingSpace']
ytest5 = df5['livingSpace']

allxnot = [xnot1,xnot2,xnot3,xnot4,xnot5]
allynot = [ynot1,ynot2,ynot3,ynot4,ynot5]
allxtest = [xtest1,xtest2,xtest3,xtest4,xtest5]
allytest = [ytest1,ytest2,ytest3,ytest4,ytest5]

import sklearn.metrics as mr

scores = []
mse = []
for i in range(5):

  np.random.seed(42)
  b = np.random.randn(1)
  w = np.random.randn(1)
  lr = 0.01
  epochs = 5000

  for epoch in range(epochs):
      error = allynot[i] -((w*allxnot[i]) + b)

      loss = (error**2).mean()

      if loss > 10**50 :
        break

      if (epoch+1) % 1000 == 0:    
          print('epoch ' + str(epoch+1)+'  mse: ' + str(loss))
    
      wgrad = 0
      bgrad = 0
    
      wgrad = -1 * (allxnot[i] * error).mean()
      w = w - (lr * wgrad)
        
      bgrad = -1 * error.mean()
      b = b - (lr*bgrad)
  print('fold ' + str(i+1))
  print('--------------------------------------')

  ypred = (w * allxtest[i]) + b            
  scores.append(mr.r2_score(allytest[i],ypred))
  mse.append(mean_squared_error(allytest[i],ypred))

print('acuracy in 5_fold cross validation ' + str(np.mean(scores)))
print('mse in 5_fold cross validation ' + str(np.mean(mse)))

df1 = data.iloc[:25000,:]
df2 = data.iloc[25000:50000,:]
df3 = data.iloc[50000:75000,:]
df4 = data.iloc[75000:100000,:]
df5 = data.iloc[100000:125000,:]
df6 = data.iloc[125000:150000,:]
df7 = data.iloc[150000:175000,:]
df8 = data.iloc[175000:200000,:]
df9 = data.iloc[200000:225000,:]
df10 = data.iloc[225000:250000,:]

dfnot1 = pd.concat([df2,df3,df4,df5,df6,df7,df8,df9,df10])
dfnot2 = pd.concat([df1,df3,df4,df5,df6,df7,df8,df9,df10])
dfnot3 = pd.concat([df1,df2,df4,df5,df6,df7,df8,df9,df10])
dfnot4 = pd.concat([df1,df2,df3,df5,df6,df7,df8,df9,df10])
dfnot5 = pd.concat([df1,df2,df3,df4,df6,df7,df8,df9,df10])
dfnot6 = pd.concat([df1,df2,df3,df4,df5,df7,df8,df9,df10])
dfnot7 = pd.concat([df1,df2,df3,df4,df5,df6,df8,df9,df10])
dfnot8 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df9,df10])
dfnot9 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df10])
dfnot10 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9])

xnot1 = dfnot1['livingSpaceRange']
xnot2 = dfnot2['livingSpaceRange']
xnot3 = dfnot3['livingSpaceRange']
xnot4 = dfnot4['livingSpaceRange']
xnot5 = dfnot5['livingSpaceRange']
xnot6 = dfnot6['livingSpaceRange']
xnot7 = dfnot7['livingSpaceRange']
xnot8 = dfnot8['livingSpaceRange']
xnot9 = dfnot9['livingSpaceRange']
xnot10 = dfnot10['livingSpaceRange']

ynot1 = dfnot1['livingSpace']
ynot2 = dfnot2['livingSpace']
ynot3 = dfnot3['livingSpace']
ynot4 = dfnot4['livingSpace']
ynot5 = dfnot5['livingSpace']
ynot6 = dfnot6['livingSpace']
ynot7 = dfnot7['livingSpace']
ynot8 = dfnot8['livingSpace']
ynot9 = dfnot9['livingSpace']
ynot10 = dfnot10['livingSpace']

xtest1 = df1['livingSpaceRange']
xtest2 = df2['livingSpaceRange']
xtest3 = df3['livingSpaceRange']
xtest4 = df4['livingSpaceRange']
xtest5 = df5['livingSpaceRange']
xtest6 = df6['livingSpaceRange']
xtest7 = df7['livingSpaceRange']
xtest8 = df8['livingSpaceRange']
xtest9 = df9['livingSpaceRange']
xtest10 = df10['livingSpaceRange']

ytest1 = df1['livingSpace']
ytest2 = df2['livingSpace']
ytest3 = df3['livingSpace']
ytest4 = df4['livingSpace']
ytest5 = df5['livingSpace']
ytest6 = df6['livingSpace']
ytest7 = df7['livingSpace']
ytest8 = df8['livingSpace']
ytest9 = df9['livingSpace']
ytest10 = df10['livingSpace']

allxnot = [xnot1,xnot2,xnot3,xnot4,xnot5,xnot6,xnot7,xnot8,xnot9,xnot10]
allynot = [ynot1,ynot2,ynot3,ynot4,ynot5,ynot6,ynot7,ynot8,ynot9,ynot10]
allxtest = [xtest1,xtest2,xtest3,xtest4,xtest5,xtest6,xtest7,xtest8,xtest9,xtest10]
allytest = [ytest1,ytest2,ytest3,ytest4,ytest5,ytest6,ytest7,ytest8,ytest9,ytest10]

scores = []
mse = []
for i in range(10):

  np.random.seed(42)
  b = np.random.randn(1)
  w = np.random.randn(1)
  lr = 0.01
  epochs = 5000

  for epoch in range(epochs):
      error = allynot[i] -((w*allxnot[i]) + b)

      loss = (error**2).mean()

      if loss > 10**50 :
        break

      if (epoch+1) % 1000 == 0:    
          print('epoch ' + str(epoch+1)+'  mse: ' + str(loss))
    
      wgrad = 0
      bgrad = 0
    
      wgrad = -1 * (allxnot[i] * error).mean()
      w = w - (lr * wgrad)
        
      bgrad = -1 * error.mean()
      b = b - (lr*bgrad)
  print('fold ' + str(i+1))
  print('--------------------------------------')

  ypred = (w * allxtest[i]) + b            
  scores.append(mr.r2_score(allytest[i],ypred))
  mse.append(mean_squared_error(allytest[i],ypred))

print('acuracy in 10_fold cross validation ' + str(np.mean(scores)))
print('mse in 10_fold cross validation ' + str(np.mean(mse)))

#  BAKHSHE 1 HALATE 2

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

reg = LinearRegression()
acuracy5 = cross_val_score(reg,x['livingSpaceRange'].values.reshape(-1,1),y.values.reshape(-1,1),cv=5)
acuracy10 =cross_val_score(reg,x['livingSpaceRange'].values.reshape(-1,1),y.values.reshape(-1,1),cv=10)
mse5 = cross_val_score(reg,x['livingSpaceRange'].values.reshape(-1,1),y.values.reshape(-1,1),cv=5,scoring='neg_mean_squared_error')
mse10 = cross_val_score(reg,x['livingSpaceRange'].values.reshape(-1,1),y.values.reshape(-1,1),cv=10,scoring='neg_mean_squared_error')

print('acuracy in 5_fold cross validation ' + str(acuracy5.mean()))
print('acuracy in 10_fold cross validation ' + str(acuracy10.mean()))
print('MSE in 5 fold cross validation ' + str(-mse5.mean()))
print('MSE in 10 fold cross validation ' + str(-mse10.mean()))

#  BAKHSHE 1 HALATE 3

reg = LinearRegression()
newx = x[['livingSpaceRange' , 'noRooms' , 'numberOfFloors' , 'thermalChar']]
acuracy5 = cross_val_score(reg,newx,y.values.reshape(-1,1),cv=5)
acuracy10 =cross_val_score(reg,newx,y.values.reshape(-1,1),cv=10)
mse5 = cross_val_score(reg,newx,y.values.reshape(-1,1),cv=5,scoring='neg_mean_squared_error')
mse10 = cross_val_score(reg,newx,y.values.reshape(-1,1),cv=10,scoring='neg_mean_squared_error')

print('acuracy in 5_fold cross validation ' + str(acuracy5.mean()))
print('acuracy in 10_fold cross validation ' + str(acuracy10.mean()))
print('MSE in 5 fold cross validation ' + str(-mse5.mean()))
print('MSE in 10 fold cross validation ' + str(-mse10.mean()))

#  BAKHSHE 1 HALATE 4

reg = LinearRegression()
acuracy5 = cross_val_score(reg,x,y.values.reshape(-1,1),cv=5)
acuracy10 =cross_val_score(reg,x,y.values.reshape(-1,1),cv=10)
mse5 = cross_val_score(reg,x,y.values.reshape(-1,1),cv=5,scoring='neg_mean_squared_error')
mse10 = cross_val_score(reg,x,y.values.reshape(-1,1),cv=10,scoring='neg_mean_squared_error')

print('acuracy in 5_fold cross validation ' + str(acuracy5.mean()))
print('acuracy in 10_fold cross validation ' + str(acuracy10.mean()))
print('MSE in 5 fold cross validation ' + str(-mse5.mean()))
print('MSE in 10 fold cross validation ' + str(-mse10.mean()))

#  BAKHSHE 1 HALATE 5

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1)
ridge = ridge.fit(x,y)
acuracy5 = cross_val_score(ridge,x,y.values.reshape(-1,1),cv=5)
acuracy10 =cross_val_score(ridge,x,y.values.reshape(-1,1),cv=10)
mse5 = cross_val_score(ridge,x,y.values.reshape(-1,1),cv=5,scoring='neg_mean_squared_error')
mse10 = cross_val_score(ridge,x,y.values.reshape(-1,1),cv=10,scoring='neg_mean_squared_error')

ridge.coef_

print('acuracy in 5_fold cross validation ' + str(acuracy5.mean()))
print('acuracy in 10_fold cross validation ' + str(acuracy10.mean()))
print('MSE in 5 fold cross validation ' + str(-mse5.mean()))
print('MSE in 10 fold cross validation ' + str(-mse10.mean()))

#  BAKHSHE 1 HALATE 6

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso = lasso.fit(x,y)
acuracy5 = cross_val_score(lasso,x,y.values.reshape(-1,1),cv=5)
acuracy10 =cross_val_score(lasso,x,y.values.reshape(-1,1),cv=10)
mse5 = cross_val_score(lasso,x,y.values.reshape(-1,1),cv=5,scoring='neg_mean_squared_error')
mse10 = cross_val_score(lasso,x,y.values.reshape(-1,1),cv=10,scoring='neg_mean_squared_error')

lasso.coef_

print('acuracy in 5_fold cross validation ' + str(acuracy5.mean()))
print('acuracy in 10_fold cross validation ' + str(acuracy10.mean()))
print('MSE in 5 fold cross validation ' + str(-mse5.mean()))
print('MSE in 10 fold cross validation ' + str(-mse10.mean()))

#  BAKHSHE 3 KAR BA DATA

train3.info()

train3.head()

for col in train3.columns :
  if train3[col].dtype == 'int64' or train3[col].dtype == 'float64':
    up = train3[col].mean() + 3*train3[col].std()
    low = train3[col].mean() - 3*train3[col].std()
    mask = (train3[col] > up ) | (train3[col] < low )
    data = train3.drop(train3[mask].index)

y = train3['price_range']

for col in train3.columns :
  if (train3[col].dtype == 'int64' or train3[col].dtype == 'float64') and (col !='price_range'):
    train3[col] = (train3[col] - min(train3[col])) /(max(train3[col]) - min(train3[col]))

train3.head()

print(train3.shape)
print(y.shape)

cor_matrix = train3.corr()

f , ax = plt.subplots(figsize=(25,30))
sns.heatmap(cor_matrix,square=True,annot=True)

x = train3.drop(columns=['price_range'])
print(x.shape)
print(y.shape)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

#  BAKHSHE 3 HALATE 1

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(xtrain,ytrain)

ypred = log.predict(xtest)
df = pd.DataFrame({'test':ytest,'pred':ypred})
df.head()

from sklearn.metrics import confusion_matrix , classification_report

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))

#  BAKHSHE 3 HALATE 2

print('Class 0 have '+ str(len(y.values[(y==0)]))+ ' itme')
print('Class 1 have '+ str(len(y.values[(y==1)]))+ ' itme')
print('Class 2 have '+ str(len(y.values[(y==2)]))+ ' itme')
print('Class 3 have '+ str(len(y.values[(y==3)]))+ ' itme')

#  BAKHSHE 3 HALATE 3

y.values[(y==2)|(y==3)] = 1

y.unique()

#  BAKHSHE 3 HALATE 4

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

newlog = LogisticRegression()
newlog.fit(xtrain,ytrain)

ypred = newlog.predict(xtest)
df = pd.DataFrame({'test':ytest,'pred':ypred})
df.head()

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))

#  BAKHSHE 3 HALATE 5

print('Class 0 :' + str(len(y[y==0])))
print('Class 1 :' + str(len(y[y==1])))

import random

rand = []
j = 0
while True:
  if j!= 1000:
    i = random.randint(0,1999)
    if (i not in rand) and (y[i]==1) :
      rand.append(i)
      j = j+1
  else:
    break

rand.sort()
rand

y.drop(index=rand,inplace=True)

x.drop(index=rand,inplace=True)
x.head()

print('Class 0 :' + str(len(y[y==0])))
print('Class 1 :' + str(len(y[y==1])))

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

newlog = LogisticRegression()
newlog.fit(xtrain,ytrain)

ypred = newlog.predict(xtest)
df = pd.DataFrame({'test':ytest,'pred':ypred})
df.head()

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))

#  BAKHSHE 3 HALATE 6

y = train3['price_range']

x = train3.drop(columns=['price_range'])
print(x.shape)
print(y.shape)

from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
feature_selector = SequentialFeatureSelector(newlog,k_features=10,forward=True,scoring='roc_auc')

from sklearn import preprocessing
yy = preprocessing.label_binarize(y, classes=[0, 1, 2, 3])

features = feature_selector.fit(np.array(x.fillna(0)),yy)

filtered_features= x.columns[list(features.k_feature_idx_)]
filtered_features

#  BAKHSHE 3 HALATE 7

xtrain, xtest, ytrain, ytest = train_test_split(x[filtered_features], y, test_size=0.3, random_state=42)

newlog = LogisticRegression()
newlog.fit(xtrain,ytrain)

ypred = newlog.predict(xtest)
df = pd.DataFrame({'test':ytest,'pred':ypred})
df.head()

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))

#  BAKHSHE 3 HALATE 8

from sklearn.decomposition import PCA

pca = PCA(0.71)
x_pca = pca.fit_transform(x)
x_pca.shape

#  BAKHSHE 3 HALATE 9

xtrain, xtest, ytrain, ytest = train_test_split(x_pca, y, test_size=0.3, random_state=42)

newlog = LogisticRegression()
newlog.fit(xtrain,ytrain)

ypred = newlog.predict(xtest)
df = pd.DataFrame({'test':ytest,'pred':ypred})
df.head()

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))

#  BAKHSHE 3 HALATE 10

feature_selector = SequentialFeatureSelector(newlog,k_features=10,forward=False,scoring='roc_auc')

features = feature_selector.fit(np.array(x.fillna(0)),yy)

filtered_features= x.columns[list(features.k_feature_idx_)]
filtered_features

xtrain, xtest, ytrain, ytest = train_test_split(x[filtered_features], y, test_size=0.3, random_state=42)

newlog = LogisticRegression()
newlog.fit(xtrain,ytrain)

ypred = newlog.predict(xtest)
df = pd.DataFrame({'test':ytest,'pred':ypred})
df.head()

print(confusion_matrix(ytest,ypred))

print(classification_report(ytest,ypred))

#  BAKHSHE 3 HALATE 11

newlog = LogisticRegression()
newlog.fit(x,y)

acuracy5 = cross_val_score(newlog,x,y.values.reshape(-1,1),cv=5)
acuracy10 =cross_val_score(newlog,x,y.values.reshape(-1,1),cv=10)
mse5 = cross_val_score(newlog,x,y.values.reshape(-1,1),cv=5,scoring='neg_mean_squared_error')
mse10 = cross_val_score(newlog,x,y.values.reshape(-1,1),cv=10,scoring='neg_mean_squared_error')

print('acuracy in 5_fold cross validation ' + str(acuracy5.mean()))
print('acuracy in 10_fold cross validation ' + str(acuracy10.mean()))
print('MSE in 5 fold cross validation ' + str(-mse5.mean()))
print('MSE in 10 fold cross validation ' + str(-mse10.mean()))