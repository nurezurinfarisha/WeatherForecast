#Import some libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
#%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Loading data set
df = pd.read_csv(r"C:\Users\User\Desktop\AI Machine Learning\weatherAUS.csv")

#dimensions of dataset
#There are 145460 rows and 23 columns
print(df.shape)

#Preview dataset
print(df)

#remove date predictor in dataset
df.drop("Date", axis = 1, inplace = True)

#preview dataset while dropping date
print(df.shape)

#Checking the columns in dataset
print(df.columns)

# Info of dataset
print(df.info())

#Checking null value count in data
print(df.isnull().sum())

#Dropping the null values in dependent variable
#Use df1 veriable for train data
df1 = df[df['RainTomorrow'].notna()]   #This is our training data
print(df1)

#Dimension of dataset while dropping null values
print(df1.shape)

#info of dataset
print(df1.info())

# find categorical variables

categorical = [var for var in df1.columns if df1[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)

#Display categorical variables
print(df1[categorical].head())

# find numerical variables

numerical = [var for var in df1.columns if df1[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

#Display nemerical variables
print(df1[numerical].head())     #Missing values in numerical variables

#checking null values in dataframe
print(df1.isna().sum())

#replacing null values in categorical variables with mode value
df1['WindGustDir'].fillna(df1['WindGustDir'].mode()[0], inplace=True)
df1['WindDir9am'].fillna(df1['WindDir9am'].mode()[0], inplace=True)
df1['WindDir3pm'].fillna(df1['WindDir3pm'].mode()[0], inplace=True)
df1['RainToday'].fillna(df1['RainToday'].mode()[0], inplace=True)
df1['RainTomorrow'].fillna(df1['RainTomorrow'].mode()[0], inplace=True)

#Count of null values in dataset
print(df1.isna().sum())

# view the categorical variables
print(df1[categorical].head())

# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'Location'.'RainTomorrow','Date','WindDir9am',	'WindDir3pm',	'RainToday',	'RainTomorrow'
df1['Location']= label_encoder.fit_transform(df1['Location'])
df1['WindGustDir']= label_encoder.fit_transform(df1['WindGustDir'])
df1['WindDir9am']= label_encoder.fit_transform(df1['WindDir9am'])
df1['WindDir3pm']= label_encoder.fit_transform(df1['WindDir3pm'])
df1['RainToday']= label_encoder.fit_transform(df1['RainToday'])
df1['RainTomorrow']= label_encoder.fit_transform(df1['RainTomorrow'])

#Display data variables
print(df1)
print(df1.info())

#There is not null record in our data
print(df1.isnull().sum())

#filling the null values in numerical variable with median
df1 = df1.fillna(df1.median())

X = df1.drop(['RainTomorrow'], axis=1)

y = df1['RainTomorrow']

# Clean like a baby's butt

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# check the shape of X_train and X_test
print(X_train.shape, X_test.shape)

# train a logistic regression model on the training set 
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)

print(y_pred_test)

#**Check accuracy score**
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()#--------------------------------------------------------------> Heat Map printing

#import classification regression
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))

# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

print(y_pred_prob)

# store the probabilities in dataframe

y_pred_prob_data1 = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

print(y_pred_prob_data1)

#Create new dataset and find null valuesdf2,isna().sum()
df2 = df[~df['RainTomorrow'].notna()]
print(df2)

print(df2.isna().sum())

# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'Location'.'RainTomorrow','Date','WindDir9am',	'WindDir3pm',	'RainToday',	'RainTomorrow'
df2['Location']= label_encoder.fit_transform(df2['Location'])
df2['WindGustDir']= label_encoder.fit_transform(df2['WindGustDir'])
df2['WindDir9am']= label_encoder.fit_transform(df2['WindDir9am'])
df2['WindDir3pm']= label_encoder.fit_transform(df2['WindDir3pm'])
df2['RainToday']= label_encoder.fit_transform(df2['RainToday'])
df2['RainTomorrow']= label_encoder.fit_transform(df2['RainTomorrow'])

#filling the null values in numerical variable
df2['MinTemp'] = df2['MinTemp'].fillna(df2['MinTemp'].median())
df2['MaxTemp'] = df2['MaxTemp'].fillna(df2['MaxTemp'].median())
df2['Rainfall'] = df2['Rainfall'].fillna(df2['Rainfall'].median())
df2['Evaporation'] = df2['Evaporation'].fillna(df2['Evaporation'].median())
df2['Sunshine'] = df2['Sunshine'].fillna(df2['Sunshine'].median())
df2['WindGustSpeed'] = df2['WindGustSpeed'].fillna(df2['WindGustSpeed'].median())
df2['WindSpeed9am'] = df2['WindSpeed9am'].fillna(df2['WindSpeed9am'].median())
df2['WindSpeed3pm'] = df2['WindSpeed3pm'].fillna(df2['WindSpeed3pm'].median())
df2['Humidity9am'] = df2['Humidity9am'].fillna(df2['Humidity9am'].median())
df2['Humidity3pm'] = df2['Humidity3pm'].fillna(df2['Humidity3pm'].median())
df2['Pressure9am'] = df2['Pressure9am'].fillna(df2['Pressure9am'].median())
df2['Pressure3pm'] = df2['Pressure3pm'].fillna(df2['Pressure3pm'].median())
df2['Cloud9am'] = df2['Cloud9am'].fillna(df2['Cloud9am'].median())
df2['Cloud3pm'] = df2['Cloud3pm'].fillna(df2['Cloud3pm'].median())
df2['Temp9am'] = df2['Temp9am'].fillna(df2['Temp9am'].median())
df2['Temp3pm'] = df2['Temp3pm'].fillna(df2['Temp3pm'].median())

print(df2.info())

from sklearn.utils.validation import indexable
X = df2.drop(['RainTomorrow'], axis=1)
y = df2['RainTomorrow']

print(X.shape)
print(y.shape)

print(X)

print(X.isna().sum())

print(y.isna().sum())
print(X.isna().sum())

print(X.head())

y_pred_df2 = logreg.predict(X)

print(y_pred_df2)

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y, y_pred_df2)))

from sklearn.metrics import classification_report

print(classification_report(y, y_pred_df2))

df2['y_pred_df2'] = y_pred_df2

print(df2)

def predict(input):
    #---------------------Input Side----------------------------------
    dst = pd.read_csv(r"{}".format(input))
    print(dst)
    print(dst.shape)
    dst.drop("Date", axis = 1, inplace = True)
    print(dst)
    print(dst.shape)
    print(dst.isnull().sum())

    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'Location'.'RainTomorrow','Date','WindDir9am',	'WindDir3pm',	'RainToday',	'RainTomorrow'
    dst['Location']= label_encoder.fit_transform(dst['Location'])
    dst['WindGustDir']= label_encoder.fit_transform(dst['WindGustDir'])
    dst['WindDir9am']= label_encoder.fit_transform(dst['WindDir9am'])
    dst['WindDir3pm']= label_encoder.fit_transform(dst['WindDir3pm'])
    dst['RainToday']= label_encoder.fit_transform(dst['RainToday'])

    #print(dst)

    input_pred = logreg.predict(dst)
    dst['Will it rain tomorrow?'] = input_pred
    dst['Will it rain tomorrow?'].mask(dst['Will it rain tomorrow?'] == 0, 'No', inplace = True)
    dst['Will it rain tomorrow?'].mask(dst['Will it rain tomorrow?'] == 1, 'Yes', inplace = True)

    #print(dst)

    #Combine columns to show location, date, and if it will rain the next day
    inidf = pd.read_csv(r"{}".format(input))
    print("debugssssssss")
    print(inidf)
    print("debugssssssss")
    dateCol = inidf["Date"]
    locCol = inidf["Location"]
    rainPredCol = dst['Will it rain tomorrow?']

    print(dateCol)
    print(locCol)

    ldf = inidf[["Date", "Location"]]
    ldf['Will it rain?'] = rainPredCol
    
    #Increment each day by 1 to make will it rain tomorrow? become today?
    ldf['Date'] = pd.to_datetime(ldf['Date']).apply(pd.DateOffset(1))
    

    print(ldf)

    return ldf