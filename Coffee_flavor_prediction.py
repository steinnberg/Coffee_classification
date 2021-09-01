# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:25:53 2021

@author: Kered
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('max_columns', 100)


#importing datas
df= pd.read_csv('merged_data_cleaned.csv')
df.head(5)
#Exploring the data
len(df.Variety)
df.Variety.unique()
df.columns

#Exploring nulber of species of coffee
df.Species.value_counts()
#types of columns
df.dtypes

#Copying datas
df2 = df.copy()

#Selecting datas columns
df2.drop(['Lot.Number','Mill','ICO.Number','Company','Producer','Owner','Farm.Name', 'In.Country.Partner','Certification.Body','Certification.Address','Certification.Contact','unit_of_measurement', ], axis=1, inplace=True)
df2.drop(['Number.of.Bags','Bag.Weight','Harvest.Year','Grading.Date','Owner.1','Expiration','altitude_low_meters','altitude_high_meters' ], axis=1, inplace=True)

#Exploring main feature of datas
sns.heatmap(df2.corr()) 
#Stat insight
df2.describe().T.round()


#Cleaning datas
df2['missing_values'] = df2.isnull().any(axis=1)
df2.head(5)

number_of_rows_of_df2 = df2.count().sum()
missing_raws = df2.missing_values.count().sum()
print(missing_raws)
print(number_of_rows_of_df2)

#We can find the number of missing rows in each column
df2.isna().sum()

#fixing datas
df2_fixed = df2.interpolate(inplace=False)
#checking datas
df2_fixed.isna().sum()


df2.dropna(inplace =True)

#labeling the two types
dct= {'Arabica':1, 'Robusta':0}
df2['Species'] = df2['Species'].map(dct)
df2.head(4)
df2.dtypes

#Descriptive analysis
sns.pairplot(data=df2)
#Heatmap correlation figure
sns.heatmap(df2.corr())

### select columns 
unique_df = df2[["Species", "Country.of.Origin", "Variety", "Processing.Method", "Color"]].copy()
#merging data values
for i,j in unique_df.iteritems():
    print('column',i)
    display(j.value_counts().to_frame())
    print()

#selecting columns
df_1=[df2["Species"].value_counts()]# for j in df2["Species"]]
df_2=[df2["Country.of.Origin"].value_counts()]# for j in df2["Species"]]
df_3=[df2["Variety"].value_counts()]
df_4=[df2["Processing.Method"].value_counts()]
df_5=[df2["Color"].value_counts()]

data = [[df_1,df_2,df_3, df_4, df_5]]
DF = pd.DataFrame(data)
DF_2 = pd.DataFrame(df_2).T
DF_2

#Map visualization
Name=[]
N=[]
for idx,name in enumerate(df2["Country.of.Origin"].value_counts().index.tolist()):
    Name.append(name)
    N.append(df2["Country.of.Origin"].value_counts()[idx])
DATA=pd.DataFrame([Name,N]).T
DATA.columns = ["Town", "N"]
DATA.head(10)

import folium
#Creating a base map
m = folium.Map()
m

#Setting up the world countries data URL
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
country_shapes = f'{url}/world-countries.json'
folium.Choropleth(
    #The GeoJSON data to represent the world country
    geo_data=country_shapes,
    name='choropleth COVID-19',
    data=DATA,
    #The column aceppting list with 2 value; The country name and  the numerical value
    columns=['Town', 'N'],
    key_on='feature.properties.name',
    fill_color='PuRd',
    #nan_fill_color='white'
).add_to(m)
#show
m
#saving
m.save('C:/Users/Kered/Documents/Ironhack/Project5_predict_coffee_flavor/Coffe_map_world.html')

#Word chart visualization
#Word Cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


#WordCloud
# Create a list of word
my_list=df2["Variety"]

word_could_dict=Counter(my_list)
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.savefig('yourfile.png', bbox_inches='tight')
plt.close()


#Make hypothesis testing
df2_clean = df2[['Species','Aroma','Flavor','Aftertaste','Acidity','Body','Balance','Uniformity','Clean.Cup','Sweetness','Cupper.Points','Total.Cup.Points','Moisture','Category.One.Defects','Quakers','Category.Two.Defects','altitude_mean_meters']]
df2_clean.head(10)
#Stat insight
df2_clean.describe().T
df2_clean.reset_index()


# import function to split data
from sklearn.model_selection import train_test_split

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#import the scoring metrics
#import the scoring metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

#Predict acidity of the coffee
df2_clean.Acidity.value_counts(normalize=True).round(2)
len(df2_clean.Acidity)
sns.scatterplot(data = df2_clean, x = 'Cupper.Points', y = 'Acidity')
from statsmodels.formula.api import ols
model = ols('Acidity ~ Flavor', data=df2_clean) # dependent ~ independent
model_fit = model.fit() #calculates everything
model_fit.summary()

model_fit.resid.hist(bins=100)


#More accurate model
df2_clean.corr().Acidity.sort_values()
sns.heatmap(df2_clean.corr(), vmin=-1, cmap='YlGn')
ols('Acidity ~ Aroma + Body  + Balance + Aftertaste  + Flavor', data=df).fit().summar
#Print results
y_pred = model_fit.predict()
resids = model_fit.resid
sns.distplot(resids)

df0 = df2_clean.copy()
df0['y_pred']=y_pred
df0 = df0.sort_values(by='Flavor')
sns.scatterplot(data = df2_clean, x = 'Flavor', y = 'Acidity')
sns.lineplot(data = df0, x = 'Flavor', y = 'y_pred')

X = df2_clean.drop(['Acidity','Aroma','Body','Balance','Aftertaste','Flavor'], axis=1)
#Scikit learn linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, df2_clean.Acidity)

#Exploring data result of the model
lr.coef_
lr.score(X, df2_clean.Acidity)
                   

                   

                   

