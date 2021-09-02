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
                   

#######New subset dataset
df3 = df[["Country.of.Origin", "Harvest.Year", "Variety", "Processing.Method", "Category.One.Defects", "Category.Two.Defects", "Quakers", "altitude_mean_meters", "Total.Cup.Points"]]
df3 = df3.dropna()
df3 = df3.reset_index()

df3 = df3.drop("index", axis = 1)
df3.head()                   

#Building new dataset
cleaned_df = df3.copy()
cleaned_df.loc[cleaned_df["Harvest.Year"] == "2017 / 2018", "Harvest.Year"] = "2018"
cleaned_df.loc[cleaned_df["Harvest.Year"] == "2016 / 2017", "Harvest.Year"] = "2017"
cleaned_df.loc[cleaned_df["Harvest.Year"] == "2015/2016", "Harvest.Year"] = "2016"
cleaned_df.loc[cleaned_df["Harvest.Year"] == "2014/2015", "Harvest.Year"] = "2015"
cleaned_df.loc[cleaned_df["Harvest.Year"] == "2013/2014", "Harvest.Year"] = "2014"
cleaned_df.loc[cleaned_df["Harvest.Year"] == "2011/2012", "Harvest.Year"] = "2012"


a = cleaned_df['Country.of.Origin'].value_counts() <= 5
b = cleaned_df['Country.of.Origin'].value_counts()
for i in range(len(a.index)):
    if(a[i]):
        cleaned_df.loc[cleaned_df["Country.of.Origin"] == a.index[i], "Country.of.Origin"] = "Others"
        
        
a = cleaned_df['Variety'].value_counts() <= 1
b = cleaned_df['Variety'].value_counts()
for i in range(len(a.index)):
    if(a[i]):
        cleaned_df.loc[cleaned_df["Variety"] == a.index[i], "Variety"] = "Others"
                   
cleaned_df.drop(cleaned_df.loc[cleaned_df['altitude_mean_meters'] > 2000].index, inplace = True) 
cleaned_df.drop(cleaned_df.loc[cleaned_df['altitude_mean_meters'] < 182].index, inplace = True) 


cut_labels = ["Specialty", "Premium", "Exchange", "Below Standard"] # 1 = Specialty Grade, 2 = Premium Coffee Grade, 3 = Exchange Coffee Grade
cut_bins = [-1, 3, 15, 23, 100]
cleaned_df['Green.Beans.Grade'] = cleaned_df["Category.One.Defects"].values + cleaned_df["Category.Two.Defects"].values + cleaned_df["Quakers"].values
cleaned_df['Green.Beans.Grade'] = pd.cut(cleaned_df['Green.Beans.Grade'], bins=cut_bins, labels=cut_labels)

cut_labels = ["UGQ", "Premium", "Specialty"] # 1 = Specialty Quality, 2 = Premium Quality, 3 = Usually Good Quality
cut_bins = [50, 80, 84, 90]
cleaned_df['Cupping.Grade'] = pd.cut(cleaned_df['Total.Cup.Points'], bins=cut_bins, labels=cut_labels)
#Plot
model_df = cleaned_df[["Country.of.Origin", "Harvest.Year", "Variety", "Processing.Method", "Green.Beans.Grade", 'Cupping.Grade', "Category.One.Defects",	"Category.Two.Defects",	"Quakers"]]
ax = sns.countplot(x="Cupping.Grade", data=model_df)
ax.tick_params(labelsize=15)


#Preporecessing
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
encode_df = model_df.copy()
column_name = ["Country.of.Origin", "Harvest.Year", "Variety", "Processing.Method", "Green.Beans.Grade", 'Cupping.Grade']

label = list()
for i in range(0,6):
    encoder.fit(encode_df[column_name[i]])
    encode_df.loc[:,column_name[i]] = (encoder.transform(encode_df[column_name[i]]))
    label.append(encoder.inverse_transform(encode_df[column_name[i]]))

    unique, counts = np.unique(label[i], return_counts=True)
    print(np.asarray((unique, counts)).T)
    unique, counts = np.unique(encode_df.loc[:,column_name[i]], return_counts=True)
    print(np.asarray((unique, counts)).T)


#Spliting data test-train
from sklearn.model_selection import train_test_split
X = encode_df.drop("Cupping.Grade", axis = 1)
Y = encode_df["Cupping.Grade"]
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# Instantiate model with 1000 decision trees


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 9)
rf = RandomForestClassifier(n_estimators = 30, random_state = 42)
# Train the model on training data
rf.fit(x_train, y_train)
y_pred=rf.predict(x_test)

print("Accuracy: %0.5f" % (metrics.accuracy_score(y_test, y_pred)))
accuracy_score(y_test,y_pred)
confusion_matrix(y_test, y_pred)


#Visualization
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cm = (confusion_matrix(y_test, y_pred))
a = ["Premium", "Specialty", "UGQ"]

sns.heatmap(cm, xticklabels = a, yticklabels = a,annot=True, fmt='g')
print(classification_report(y_test, y_pred))

#Ploting the matrix confusion result
import scikitplot as skplt

y_probas = rf.predict_proba(x_test)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()


#KNeightbors algo
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
confusion_matrix(y_test, y_pred2)

#printing accuracy of the model
accuracy_score(y_test, y_pred2)



#Logistic regression
model3 = LogisticRegression(max_iter=1e8)
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
confusion_matrix(y_test, y_pred3)

accuracy_score(y_test, y_pred3)


#Decision tree algo
model4 = DecisionTreeClassifier()
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
#display(confusion_matrix(y_test, y_pred4))
print('Accuracy is', accuracy_score(y_test, y_pred4))



#Evaluation of the metric
X_train, X_test, y_train, y_test = train_test_split(
    df2_clean.drop('Acidity',axis=1), #X
    df2_clean.Acidity,                #y
    test_size=0.3,           
    random_state=42,
    )


#columns I want
sub_cols = {"Aroma","Flavor","Aftertaste","Acidity","Body","Balance","Uniformity","Total.Cup.Points"}

y = df2_clean["Total.Cup.Points"]
x = df2_clean.drop("Total.Cup.Points", axis=1)


from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score


## families are a broad type of model
from sklearn.ensemble import RandomForestRegressor

# cross validation tools
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, # saves 20 percent to test
                                                    random_state=123) #arbitrary 


## defining a pipeline, takes care of scalar creation and model fit

pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

# We can also list the tunable hyper parameters like this
pipeline.get_params()


#Declare Hyper parameters 
## This should be saved in a dictionary
## Note how to save keys with list values, i frequently forget this

hyperparameters = { 
    'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
    'randomforestregressor__max_depth': [None, 5, 3, 1]
}

#searshing best model
clf = GridSearchCV(pipeline, hyperparameters,cv=10) 
clf.fit(x_train,y_train)

clf.best_params_

clf.refit

y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))


x = range(0,len(y_pred))

fig = plt.figure()
plt.title('Predicted Coffee Quality Points')
plt.plot(x, y_pred, color = 'm', label = 'Predicted Values')
plt.plot(x, list(y_test), color = 'b', label = 'Test values')
plt.axis([-10,len(y_pred),70,90])
plt.legend(loc = 1)
plt.savefig('comparaison')
plt.show()


