# -*- coding: utf-8 -*-
"""
	@script-author: Jyothi Tom, Vikram Shakthi, Christina John
	@script-description: Python code to build classification models to predict the risk of extreme birth weight values.   
    	@script-details: Written in Google Colaboratory.

"""

pip install catboost

import numpy as np
import pandas as pd
import seaborn as sns 
from math import sqrt
import matplotlib.pyplot as plt  
from scipy.stats import *  
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as multi
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/drive/My Drive/Pregnancy Paper/Data/US_births_work_csv.csv")

data.head()

data.shape

data.columns

data.dtypes

#Dropping irrelevant features
data.drop(['WTGAIN','DWgt_R','MM_AICU','LD_INDL'], axis=1, inplace=True)

#Removing records of women who started prenatal care in their 10th month
data.drop(data[data.PRECARE==10].index,axis=0,inplace=True)

"""## Missing Values"""

#To calculate the percent of missing values in variables of type 'object'
def missing_perc(dat, obj_cols):
  for col in obj_cols:
    perc = len(dat[col][dat[col]=='U'])/dat.shape[0]
    print(col,":", perc)

obj_cols = data.columns[data.dtypes=='object']
obj_cols

print(len(data.DBWT[data.DBWT==9999])/data.shape[0])

print(len(data.BMI[data.BMI==99.9])/data.shape[0])

print(len(data.CIG_0[data.CIG_0==99])/data.shape[0])

len(data.PWgt_R[data.PWgt_R==999])/data.shape[0]

len(data.M_Ht_In[data.M_Ht_In==99])/data.shape[0]

len(data.PREVIS[data.PREVIS==99])/data.shape[0]

len(data.PRIORTERM[data.PRIORTERM==99])/data.shape[0]

len(data.ILP_R[data.ILP_R==999])/data.shape[0]

len(data.MBSTATE_REC[data.MBSTATE_REC==3])/data.shape[0]

len(data.MEDUC[data.MEDUC==9])/data.shape[0]

len(data.MRACE6[data.MRACE6==9])/data.shape[0]

len(data.NO_INFEC[data.NO_INFEC==9])/data.shape[0]

len(data.NO_MMORB[data.NO_MMORB==9])/data.shape[0]

len(data.NO_RISKS[data.NO_RISKS==9])/data.shape[0]

len(data.PRECARE[data.PRECARE==99])/data.shape[0]

print(len(data.DMAR[data.DMAR==9])/data.shape[0])

print(len(data.FRACE6[data.FRACE6==9])/data.shape[0])

#Father's educ
print(data.isnull().sum()[data.isnull().sum()>0]/data.shape[0])

missing_perc(data, obj_cols)

"""Since the features 'Father's Education' and 'Father's Race' had a huge proportion of missing values, these features were dropped. The rest of the records with missing values were dropped as it made up of only a small proportion of the entire dataset."""

data.drop(['FRACE6', 'FEDUC'], axis=1, inplace=True)

#Dropping missing/unknown values from each column
#Numerical
data.drop(data.index[data.BMI==99.9], axis=0, inplace=True)
data.drop(data.index[data.CIG_0==99], axis=0, inplace=True)
data.drop(data.index[data.DBWT==9999], axis=0, inplace=True)
data.drop(data.index[data.PWgt_R==999], axis=0, inplace=True)
data.drop(data.index[data.M_Ht_In==99], axis=0, inplace=True)
data.drop(data.index[data.PREVIS==99], axis=0, inplace=True)
data.drop(data.index[data.PRIORTERM==99], axis=0, inplace=True)
data.drop(data.index[data.ILP_R==999], axis=0, inplace=True)

#Categorical (of int type)
data.drop(data.index[data.MBSTATE_REC==3], axis=0, inplace=True)
data.drop(data.index[data.MEDUC==9], axis=0, inplace=True)
data.drop(data.index[data.NO_INFEC==9], axis=0, inplace=True)
data.drop(data.index[data.NO_MMORB==9], axis=0, inplace=True)
data.drop(data.index[data.NO_RISKS==9], axis=0, inplace=True)
data.drop(data.index[data.PRECARE==99], axis=0, inplace=True)

#To drop records with missing values in variables of type 'object'
def drop_missing(dat, obj_cols):
  for col in obj_cols:
    dat.drop(data.index[dat[col]=='U'], axis=0, inplace=True)
  return dat

data = drop_missing(data,obj_cols)

data.ILP_R.replace(888,0, inplace=True)   #Here, 888 signifies 'no previous pregnancies'. Hence, 0 will be a better value for the same.

data.columns

data.head()

data.shape

"""# Functions"""

#Heatmap
def heat_map(df):
  corr = df.corr()
  sns.heatmap(corr, annot = True, annot_kws={'size':12}, cmap="Blues")
  plt.gcf().set_size_inches(10,5)
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.show()

#KDE Plots
def kde(df,col):
  df.groupby('DBWT')[col].plot.kde()
  plt.ylabel('Probability')
  plt.xlabel(col)
  plt.title('KDE Plot for column '+ col)
  plt.legend(labels=df['DBWT'].unique().tolist())
  plt.show()

#Kruskal-Wallis H Test
def kruskal_wallis(df, col):
  krusk_stats,p = kruskal(*[group[col].values for name, group in df.groupby('DBWT')])
  print('H Statistic:',krusk_stats)
  print('p-value:',p)

#Chi-Sq Test of Independence and Cramer's V
def chisq(df, col1, col2='DBWT'):
  conting_tbl = pd.crosstab(df[col1], df[col2])
  # stats, p, = chi2_contingency(conting_tbl)[:2]
  chi2,p = chi2_contingency(conting_tbl, correction=True)[:2]
  print("ChiSq Statistics:", chi2)
  print("p-value:", p)

  n = sum(conting_tbl.sum())
  phi2 = chi2/n
  nrows,ncols = conting_tbl.shape
  phi2corr = max(0, phi2 - ((ncols-1)*(nrows-1))/(n-1))    
  rowscorr = nrows - ((nrows-1)**2)/(n-1)
  colscorr = ncols - ((ncols-1)**2)/(n-1)
  result= sqrt(phi2corr / min((colscorr-1), (rowscorr-1)))
  print("Cramer's V:", round(result,5))

"""# Classification Problem"""

num_cols = ['CIG_0', 'MAGER', 'M_Ht_In', 'PREVIS', 'PRIORTERM', 'PWgt_R','ILP_R', 'DBWT', "BMI"]

newdf = data.copy(deep=True)     #For Classification

"""The regression models built performed poorly, given the huge variance within the dataset. It was decided to convert this into a classification problem with 2 classes, namely 'Class 0' indicating normal birth weights and 'Class 1' indicating birth weights outside the normal range."""

newdf.DBWT.loc[(newdf.DBWT>=2500)&(newdf.DBWT<=4500)] = 0     #Normal

newdf.DBWT.loc[newdf.DBWT!=0] = 1     #Not Normal

newdf.DBWT.value_counts()/newdf.shape[0]

num_cols.remove('DBWT')

"""## Numerical Features"""

newdf['CIG_0'][newdf.CIG_0 != 0] = 1  #Smokers
newdf.CIG_0.unique()

num_cols.remove('CIG_0')

age_bins = pd.IntervalIndex.from_tuples([(10,15), (15,20),(20,25), (25,30),(30,35), (35,40),(40,45), (45,50)],closed='right')
AGE = pd.cut(newdf.MAGER, bins=age_bins)
AGE.value_counts()

newdf = newdf[(newdf.MAGER>=15) & (newdf.MAGER<46)]      #Considering mothers aged 15 to 45

newdf['Age'] = pd.cut(newdf.MAGER, bins=5, precision=0)    # MAGER - [15-40]
newdf['Age'].value_counts()

newdf.drop('MAGER', axis=1, inplace=True)
num_cols.remove('MAGER')

chisq(newdf, 'Age', 'DBWT')

newdf = newdf[newdf.PREVIS<26]

newdf["Visits"] = pd.cut(newdf.PREVIS, bins=5, precision=0)
newdf["Visits"].value_counts()

newdf.drop('PREVIS', axis=1,inplace=True)
num_cols.remove('PREVIS')

newdf.PRIORTERM[newdf.PRIORTERM!=0] = 1  #Prior terminations

newdf.PRIORTERM.value_counts()

num_cols.remove('PRIORTERM')

newdf["PWgt_R"] = StandardScaler().fit_transform(np.array(newdf.PWgt_R).reshape(-1, 1))
newdf["M_Ht_In"] = StandardScaler().fit_transform(np.array(newdf.M_Ht_In).reshape(-1, 1))    #Use if 'Height' variable is not binned
newdf["ILP_R"] = StandardScaler().fit_transform(np.array(newdf.ILP_R).reshape(-1, 1))
newdf["BMI"] = StandardScaler().fit_transform(np.array(newdf.BMI).reshape(-1, 1))

heat_map(newdf[num_cols])

for col in num_cols:
  sns.distplot(newdf[col])
  plt.show()

"""Since none of the numerical features followed Normal distribution, ANOVA test could not be used to analyse the effect of these features on Birth Weight. Hence, Kriskal-Wallis H Test was used."""

for col in num_cols: 
  print("\n",col.upper())
  kde(newdf, col)
  kruskal_wallis(newdf,col)

newdf[num_cols].corr()

"""Since 'Mother's Weight (PWgt_R)' and 'Mother's Height (M_Ht_In)' were highly correlated, they were dropped and 'Mother's BMI (BMI)' was used instead."""

num_cols.remove('PWgt_R')
num_cols.remove('M_Ht_In')

newdf.drop(['M_Ht_In', 'PWgt_R'], axis=1, inplace=True)

newdf.shape

num_cols = ['BMI', 'ILP_R']

"""## Categorical Features"""

cat_cols = [col for col in newdf.columns if col not in num_cols+['DBWT']]

for col in cat_cols:
  print("\n",col.upper())
  chisq(newdf, col)

"""Using Chi-Square Test of Independence and Cramer's V, the association between the nominal features and Birth Weight were studied. The irrelevant features were removed."""

cat_cols.remove("IP_GON")
cat_cols.remove("NO_MMORB")
cat_cols.remove("MEDUC")
cat_cols.remove("NO_INFEC")

newdf.drop(['IP_GON', 'NO_MMORB', 'MEDUC', 'NO_INFEC'],axis=1,inplace=True)

newdf.shape

"""## Modelling using Imbalanced Dataset"""

X = newdf.drop('DBWT', axis=1)
y = newdf['DBWT'].copy(deep=True)

X.columns

Xdumm = pd.get_dummies(X, columns=cat_cols, drop_first=True)

Xdumm.rename(columns = {'Age_(21.0, 27.0]':'Age_21_27','Age_(27.0, 33.0]':'Age_28_33','Age_(33.0, 39.0]':'Age_34_39','Age_(39.0, 45.0]':'Age_40_45', 'Visits_(5.0, 10.0]':'Visits_6_10','Visits_(10.0, 15.0]':'Visits_11_15','Visits_(15.0, 20.0]':'Visits_16_20','Visits_(20.0, 25.0]':'Visits_21_25'}, inplace=True)

Xdumm.columns

Xdumm.shape

y.value_counts()/y.shape[0]

cats = [col for col in Xdumm.columns if col not in num_cols+['DBWT']]

X_train, X_test, y_train, y_test = train_test_split(Xdumm, y, test_size=0.3, random_state=24)

y_train.unique()

"""### Decision Tree Classifier"""

dtc =  DecisionTreeClassifier(random_state=0, max_depth=10, class_weight='balanced')   #With weights

dtc.fit(X_train, y_train)  #With weights

dtc_pred = dtc.predict(X_test)

dtc_pred

dtc_prob = dtc.predict_proba(X_test)[:,1]

dtc_prob

confusion_matrix(y_test, dtc_pred)

#Weights
print("Precision Score:", round(precision_score(y_test, dtc_pred),5))
print("Recall Score:", round(recall_score(y_test, dtc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, dtc_pred),5))
print("F1 Score:", round(f1_score(y_test, dtc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, dtc_prob),5))

"""### Random Forest Classifier"""

rfc = RandomForestClassifier(class_weight='balanced')

rfc.fit(X_train, y_train)

#Obtaining the importance of all the features in the Random Forest Classifier model
feature_imp = pd.Series(rfc.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print("Important Features are :\n",feature_imp)

rfc_pred = rfc.predict(X_test)

rfc_pred

rfc_prob = rfc.predict_proba(X_test)[:,1]

rfc_prob

confusion_matrix(y_test, rfc_pred)

#With Weights
print("Precision Score:", round(precision_score(y_test, rfc_pred),5))
print("Recall Score:", round(recall_score(y_test, rfc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, rfc_pred),5))
print("F1 Score:", round(f1_score(y_test, rfc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, rfc_prob),5))

"""### Gradient Boosting Classifier"""

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

#Loss Function used
gbc.loss_
#Deviance

#Obtaining the importance of all the features in the Gradient Boosting Classifier model
feature_imp = pd.Series(gbc.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print("Important Features are :\n",feature_imp)

gbc_pred = gbc.predict(X_test)

gbc_pred

gbc_prob = gbc.predict_proba(X_test)[:,1]

gbc_prob

confusion_matrix(y_test, gbc_pred)

print("Precision Score:", round(precision_score(y_test, gbc_pred),5))
print("Recall Score:", round(recall_score(y_test, gbc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, gbc_pred),5))
print("F1 Score:", round(f1_score(y_test, gbc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, gbc_prob),5))

"""### XGBoost"""

xbgc = xgb.XGBClassifier()

xbgc.fit(X_train,y_train)

xbgc_pred = xbgc.predict(X_test)

xbgc_prob = xbgc.predict_proba(X_test)[:,1]

confusion_matrix(y_test, xbgc_pred)

print("Precision Score:", round(precision_score(y_test, xbgc_pred),5))
print("Recall Score:", round(recall_score(y_test, xbgc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, xbgc_pred),5))
print("F1 Score:", round(f1_score(y_test, xbgc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, xbgc_prob),5))

"""### CatBoost"""

CBC = CatBoostClassifier()

CBC.fit(X_train,y_train, cat_features=cats)

cbc_pred = CBC.predict(X_test)

cbc_pred

np.unique(cbc_pred)

cbc_prob = CBC.predict_proba(X_test)[:,1]     #Taking probabilities for positive outcome

cbc_prob

confusion_matrix(y_test, cbc_pred)

print("Precision Score:", round(precision_score(y_test, cbc_pred),5))
print("Recall Score:", round(recall_score(y_test, cbc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, cbc_pred),5))
print("F1 Score:", round(f1_score(y_test, cbc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, cbc_prob),5))

"""## Modelling using Balanced dataset (obtained using SMOTE)"""

from imblearn.over_sampling import SMOTENC
# from imblearn.under_sampling import RandomUnderSampler

X.columns

smote = SMOTENC(categorical_features=[1,2,4,5,6,7,8,9,10,11,12], sampling_strategy='minority', k_neighbors=5)

X_smote, y_smote = smote.fit_resample(X,y)

from collections import Counter
Counter(y_smote)

X1 = pd.DataFrame(X_smote.tolist())

X1.columns = X.columns.copy()

X1.head()

smote_df = pd.concat([X1,pd.Series(y_smote.tolist(), name='DBWT')], axis=1)

smote_df.to_csv('/content/drive/My Drive/Pregnancy Paper/Data/smote_df.csv',index=False)

smote_df = pd.read_csv("/content/drive/My Drive/Pregnancy Paper/Data/smote_df.csv")

smote_df.head()

smote_df.DBWT.value_counts()

y_smote = smote_df.DBWT.copy()
X1 = smote_df.drop('DBWT',axis=1)

X1.head()

Xdumm = pd.get_dummies(X1, columns=cat_cols, drop_first=True)

Xdumm.rename(columns = {'Age_(21.0, 27.0]':'Age_22_27','Age_(27.0, 33.0]':'Age_28_33','Age_(33.0, 39.0]':'Age_34_39','Age_(39.0, 45.0]':'Age_40_45', 'Visits_(5.0, 10.0]':'Visits_6_10','Visits_(10.0, 15.0]':'Visits_11_15','Visits_(15.0, 20.0]':'Visits_16_20','Visits_(20.0, 25.0]':'Visits_21_25'}, inplace=True)

Xdumm.columns

Xdumm.shape

cats = [col for col in Xdumm.columns if col not in num_cols+['DBWT']]

X_train, X_test, y_train, y_test = train_test_split(Xdumm, y_smote, test_size=0.3, random_state=24)

"""### Decision Tree Classifier"""

dtc =  DecisionTreeClassifier(random_state=0, max_depth=10)

dtc.fit(X_train, y_train)

dtc_pred = dtc.predict(X_test)

dtc_pred

dtc_prob = dtc.predict_proba(X_test)[:,1]

dtc_prob

#After SMOTE
print("Classification Report:\n", classification_report(y_test,dtc_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dtc_pred))
print("Precision Score:", round(precision_score(y_test, dtc_pred),5))
print("Recall Score:", round(recall_score(y_test, dtc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, dtc_pred),5))
print("F1 Score:", round(f1_score(y_test, dtc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, dtc_prob),5))

"""### Random Forest Classifier"""

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

#Obtaining the importance of all the features in the Random Forest Classifier model
feature_imp = pd.Series(rfc.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print("Important Features are :\n",feature_imp)

rfc_pred = rfc.predict(X_test)

rfc_pred

rfc_prob = rfc.predict_proba(X_test)[:,1]

rfc_prob

#With SMOTE
print("Classification Report:\n", classification_report(y_test, rfc_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rfc_pred))
print("Precision Score:", round(precision_score(y_test, rfc_pred),5))
print("Recall Score:", round(recall_score(y_test, rfc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, rfc_pred),5))
print("F1 Score:", round(f1_score(y_test, rfc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, rfc_prob),5))

"""### Gradient Boosting Classifier"""

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

#Loss Function used
gbc.loss_
#Deviance

#Obtaining the importance of all the features in the Gradient Boosting Classifier model
feature_imp = pd.Series(gbc.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print("Important Features are :\n",feature_imp)

gbc_pred = gbc.predict(X_test)

gbc_pred

gbc_prob = gbc.predict_proba(X_test)[:,1]

gbc_prob

print("Classification Report:\n", classification_report(y_test,gbc_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,gbc_pred))
print("Precision Score:", round(precision_score(y_test, gbc_pred),5))
print("Recall Score:", round(recall_score(y_test, gbc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, gbc_pred),5))
print("F1 Score:", round(f1_score(y_test, gbc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, gbc_prob),5))

"""### XGBoost"""

xbgc = xgb.XGBClassifier()

xbgc.fit(X_train,y_train)

xbgc_pred = xbgc.predict(X_test)

xbgc_pred

xbgc_prob = xbgc.predict_proba(X_test)[:,1]

xbgc_prob

#With SMOTE
print("Classification Report:\n", classification_report(y_test,xbgc_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xbgc_pred))
print("Precision Score:", round(precision_score(y_test, xbgc_pred),5))
print("Recall Score:", round(recall_score(y_test, xbgc_pred),5))
print("Accuracy Score:", round(accuracy_score(y_test, xbgc_pred),5))
print("F1 Score:", round(f1_score(y_test, xbgc_pred, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y_test, xbgc_prob),5))

"""### CatBoost"""

CBC = CatBoostClassifier()

X1.head()

X0_train, X0_test, y0_train, y0_test = train_test_split(X1, y_smote, test_size=0.3, random_state=24)

CBC.fit(X0_train, y0_train, cat_features=cat_cols)

cbc_pred0 = CBC.predict(X0_test)

cbc_pred0

cbc_prob0 = CBC.predict_proba(X0_test)[:,1]     #Taking probabilities for positive outcome

cbc_prob0

print("Classification Report:", classification_report(y0_test, cbc_pred0))
print("Confusion Matrix:\n", confusion_matrix(y0_test, cbc_pred0))
print("Precision Score:", round(precision_score(y0_test, cbc_pred0),5))
print("Recall Score:", round(recall_score(y0_test, cbc_pred0),5))
print("Accuracy Score:", round(accuracy_score(y0_test, cbc_pred0),5))
print("F1 Score:", round(f1_score(y0_test, cbc_pred0, average='weighted'),5))
print("ROC: ", round(roc_auc_score(y0_test, cbc_prob0),5))
