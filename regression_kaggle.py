import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#loading the data
df_train = pd.read_csv('train.csv')

# understand the different columns
df_train.columns

#descriptive statistics summary for sales
df_train['SalePrice'].describe()

#histogram for sales price
sns.distplot(df_train['GrLivArea'])
plt.show() # needed to see the plots in pycharm

#sknewness and kurtois of the salesprice
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#scatter plot grlivarea/saleprice (to see the relationship)
df_train.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
plt.show()

#scatter plot TotalBsmtSF/saleprice (to see the relationship)
df_train.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000))
plt.show()

#boxplot to show the relationship between overallqual/saleprice
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=df_train)
fig.axis(ymin=0, ymax=800000)
plt.show()

#boxplot to show the relationship between overallqual/saleprice
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=df_train)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.show()

#correlation matrix at a glance, including the categorical variables
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9)) # need to include this to fit all the variables
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

#saleprice correlation matrix - importnatn ones
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#select variables with high correlation,
#select variables that are independent of one another.

#pairwise scatterplot of important matrix
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()

#how to handle missing data
#1. drop entire column if its the same
#2. remove some values if only a small percentage
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) #axis is to concat by column
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...

#standardizing data(0,1) and find the threshold of being an outlier
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
#if its close to 0/1, its ok. But if too far, we need to be careful.

#looking at individual points- GrLive Area, delete points which dont make sense
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# Before using multivariate analysis, we need to check for 4 assumptions
# 1. Normality - follow normal distribution
# 2. Homoscedasticity - variables have the same variance
# 3. Linearity - linear patterns. if not linear, will need some transformation
# 4. Absence of correlated errors - if detected, try to add in a variable that will explain.

# To spot for normality (go through var by var)
# Histogram - Kurtosis and skewness.
# Normal probability plot - Data distribution should closely follow the
# diagonal that represents the normal distribution.

# if not normal:
# A simple data transformation can solve the problem.
# In case of positive skewness, log transformations usually works well.

#Check sale price
sns.distplot(df_train['SalePrice'], fit=norm)
plt.show()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
# In this case, is positive sknewness -> log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm);
plt.show()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

#check grlivarea
sns.distplot(df_train['GrLivArea'], fit=norm);
plt.show()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.show()
#sknewness -> log transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'], fit=norm);
plt.show()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.show()

#check totalbsmtsf
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
plt.show()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
plt.show()
# if there are sknewness and many points at 0, we need to selectively
# transform the var - because log do not work on 0.
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
plt.show()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.show()

#scatter plot to check for homoscedatcit - single diagonal line
#normally if the variables are normal, its fine
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])

#convert categorical variable into dummy - hot encoding
df_train = pd.get_dummies(df_train)