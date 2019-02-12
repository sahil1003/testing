# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:31:25 2019

@author: Admin
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import the dataset
data = pd.read_excel('C:/Users/Admin/Desktop/sahilpy/indian_patients.xlsx')
data.head()

data.rename(columns={"Albumin.1":"Albumin_and_Globulin_Ratio"},inplace=True)
data.rename(columns = {"Globulin_Ratio,Class": "Class"},inplace = True)                                 
data.head()
#check for the na values 
data.isnull().sum()
data= pd.DataFrame(data)
data.isna().sum()
#sns.pairplot(data)##visualizing the data using pairplot
print(data.columns)


##creating the boxplots , it is a univariate data analysis , below we are checking the outliers in the data
sns.boxplot(data['Age']) #Not contain Outliers
#sns.boxplot(data['Unnamed: 0'])#not contain outliers
sns.boxplot(data['Total_Bilirubin'])
sns.boxplot(data['Direct_Bilirubin'])
sns.boxplot(data['Alkaline_Phosphotase'])
sns.boxplot(data['Alamine_Aminotransferase'])
sns.boxplot(data['Aspartate_Aminotransferase'])
sns.boxplot(data['Albumin'])#not contain Outliers
sns.boxplot(data[ 'Total_Protiens'])
sns.boxplot(data[ 'Albumin_and_Globulin_Ratio'])

##here we are checking the distribution of the data 
plt.hist(data['Total_Protiens'])
plt.hist(data['Albumin_and_Globulin_Ratio'])
plt.hist(data['Total_Bilirubin'])
plt.hist(data['Direct_Bilirubin'])
plt.hist(data['Alkaline_Phosphotase'])
plt.hist(data['Alamine_Aminotransferase'])
plt.hist(data['Aspartate_Aminotransferase'])
plt.hist(data['Albumin'])

##here we are imputing the na values with mean and mode

mean_value=data['Total_Protiens'].mean()
data['Total_Protiens']=data['Total_Protiens'].fillna(mean_value)
mean_value1=data['Albumin_and_Globulin_Ratio'].mean()
data['Albumin_and_Globulin_Ratio']=data['Albumin_and_Globulin_Ratio'].fillna(mean_value1)
data.Gender.mode()
data['Gender']=data['Gender'].fillna('Male')

#we are applying log on the data , it reduces the variability in data it reduces the outliers, the data become compact or dense
data['Albumin_and_Globulin_Ratio']=np.log(data['Albumin_and_Globulin_Ratio'])
data['Total_Bilirubin']=np.log(data['Total_Bilirubin'])
data['Direct_Bilirubin']=np.log(data['Direct_Bilirubin'])
data['Alkaline_Phosphotase']=np.log(data['Alkaline_Phosphotase'])
data['Alamine_Aminotransferase']=np.log(data['Alamine_Aminotransferase'])
data['Aspartate_Aminotransferase']=np.log(data['Aspartate_Aminotransferase'])


##in below code , we are finding the outliers and remove the outliers
Q1 = data['Total_Bilirubin'].quantile(0.25)
Q3 = data['Total_Bilirubin'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print (lower_limit)
print (upper_limit)
print (data.Total_Bilirubin[data.Total_Bilirubin <= lower_limit].shape)
print (data.Total_Bilirubin[data.Total_Bilirubin >= upper_limit].shape)
data['Total_Bilirubin'] = data.query('(@Q1 - 1.5 * @IQR) <= Total_Bilirubin <= (@Q3 + 1.5 * @IQR)')
#mean_value3=data['Total_Bilirubin'].median()
#data['Total_Bilirubin']=data['Total_Bilirubin'].fillna(mean_value3)
#data=data.dropna(how = 'any')
#data.dropna(['Total_Bilirubin'])

Q4 = data['Direct_Bilirubin'].quantile(0.25)
Q5 = data['Direct_Bilirubin'].quantile(0.75)
IQR1 = Q5 - Q4
print(IQR1)
lower_limit1 = Q4 - 1.5 * IQR1
upper_limit1 = Q5 + 1.5 * IQR1
print (lower_limit1)
print (upper_limit1)
print (data.Direct_Bilirubin[data.Direct_Bilirubin <= lower_limit1].shape)
print (data.Direct_Bilirubin[data.Direct_Bilirubin >= upper_limit1].shape)
data['Direct_Bilirubin'] = data.query('(@Q4 - 1.5 * @IQR1) <= Direct_Bilirubin <= (@Q5 + 1.5 * @IQR1)')
#mean_value4=data['Direct_Bilirubin'].median()
#data['Direct_Bilirubin']=data['Direct_Bilirubin'].fillna(mean_value4)
#data=data.dropna(how = 'any')

Q5 = data['Alkaline_Phosphotase'].quantile(0.25)
Q6 = data['Alkaline_Phosphotase'].quantile(0.75)
IQR2 = Q6 - Q5
print(IQR2)
lower_limit2 = Q5 - 1.5 * IQR2
upper_limit2 = Q6 + 1.5 * IQR2
print (lower_limit2)
print (upper_limit2)
print (data.Alkaline_Phosphotase[data.Alkaline_Phosphotase <= lower_limit2].shape)
print (data.Alkaline_Phosphotase[data.Alkaline_Phosphotase >= upper_limit2].shape)
data['Alkaline_Phosphotase'] = data.query('(@Q5 - 1.5 * @IQR2) <= Alkaline_Phosphotase <= (@Q6 + 1.5 * @IQR2)')
#mean_value5=data['Alkaline_Phosphotase'].median()
#data['Alkaline_Phosphotase']=data['Alkaline_Phosphotase'].fillna(mean_value5)
#data=data.dropna(how = 'any')

Q7 = data['Alamine_Aminotransferase'].quantile(0.25)
Q8 = data['Alamine_Aminotransferase'].quantile(0.75)
IQR3 = Q8 - Q7
print(IQR3)
lower_limit3 = Q7 - 1.5 * IQR3
upper_limit3 = Q8 + 1.5 * IQR3
print (lower_limit3)
print (upper_limit3)
print (data.Alamine_Aminotransferase[data.Alamine_Aminotransferase <= lower_limit3].shape)
print (data.Alamine_Aminotransferase[data.Alamine_Aminotransferase >= upper_limit3].shape)
data['Alamine_Aminotransferase'] = data.query('(@Q7 - 1.5 * @IQR3) <= Alamine_Aminotransferase <= (@Q8 + 1.5 * @IQR3)')
#mean_value6=data['Alamine_Aminotransferase'].median()
#data['Alamine_Aminotransferase']=data['Alamine_Aminotransferase'].fillna(mean_value6)
#data=data.dropna(how = 'any')
#data.shape

Q9 = data['Aspartate_Aminotransferase'].quantile(0.25)
Q10 = data['Aspartate_Aminotransferase'].quantile(0.75)
IQR4 = Q10 - Q9
print(IQR4)
lower_limit4 = Q9 - 1.5 * IQR4
upper_limit4 = Q10 + 1.5 * IQR4
print (lower_limit4)
print (upper_limit4)
print (data.Aspartate_Aminotransferase[data.Aspartate_Aminotransferase <= lower_limit4].shape)
print (data.Aspartate_Aminotransferase[data.Aspartate_Aminotransferase >= upper_limit4].shape)
data['Aspartate_Aminotransferase'] = data.query('(@Q9 - 1.5 * @IQR4) <= Aspartate_Aminotransferase<= (@Q10 + 1.5 * @IQR4)')
#mean_value7=data['Aspartate_Aminotransferase'].median()
#data['Aspartate_Aminotransferase']=data['Aspartate_Aminotransferase'].fillna(mean_value7)


Q11 = data['Total_Protiens'].quantile(0.25)
Q12 = data['Total_Protiens'].quantile(0.75)
IQR5 = Q12 - Q11
print(IQR5)
lower_limit5 = Q11 - 1.5 * IQR5
upper_limit5 = Q12 + 1.5 * IQR5
print (lower_limit5)
print (upper_limit5)
print (data.Total_Protiens[data.Total_Protiens <= lower_limit5].shape)
print (data.Total_Protiens[data.Total_Protiens >= upper_limit5].shape)
data['Total_Protiens'] = data.query('(@Q11 - 1.5 * @IQR5) <= Total_Protiens <= (@Q12 + 1.5 * @IQR5)')
#mean_value=data['Total_Protiens'].mean()
#data['Total_Protiens']=data['Total_Protiens'].fillna(mean_value)

data=data.dropna(how = 'any')
data.shape


X = data.iloc[:,0:-1].values
X=pd.DataFrame(X)
X.head()
Y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

Y=pd.DataFrame(Y)
Y.head()
##creating the dummies of the data
dum=pd.get_dummies(X[1])
dum.head()

X = pd.concat([X,dum],axis=1)
X.head()

X = X.drop([1],axis=1)
X.head()


#splitting the data in train and test
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#we are scaling the data, scaling is important if we have very smaller values and higher values in data , so model gives more preference or weightage to the higher values so it is a good practice to scale the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
Xtrain = sc.fit_transform(Xtrain)  
Xtest = sc.transform(Xtest)  

'''
#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
explained_variance = pca.explained_variance_ratio_
'''
# Applying the logistic regression model
#I built logistic , random forest , xgboost , svm models, but the logistic regression gives more better results among all other models


from sklearn.linear_model import LogisticRegression
cla = LogisticRegression(random_state=50)
cla.fit(Xtrain,Ytrain)


# Predicting the Test set results
y_pred = cla.predict(Xtest)

#creating the confusion matrix and accuracy score to evaluate the model
from sklearn.metrics import  confusion_matrix,accuracy_score


cm=confusion_matrix(Ytest,y_pred)  
print(cm)
print(accuracy_score(Ytest, y_pred))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Patient Has Liver Disease or Not Has a Disease Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


'''
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=1000,random_state=1)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(Xtrain,Ytrain)
y_pred1=clf.predict(Xtest)
print(confusion_matrix(Ytest,y_pred1))  
print(accuracy_score(Ytest, y_pred1))

importances_rf = pd.Series(clf.feature_importances_,index=Xtrain.columns)
sorted_importances_rf = importances_rf.sort_values(ascending=False)
sorted_importances_rf .plot(kind = 'barh',color = 'lightgreen')
plt.show()

Xtrain=pd.DataFrame(Xtrain)
Xtest=pd.DataFrame(Xtest)

Xtrain=Xtrain.iloc[:,3:6]
Xtest=Xtest.iloc[:,3:6]



from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(Xtrain,Ytrain)

y_pred2=classifier.predict(Xtest)

print(confusion_matrix(Ytest,y_pred2))    
print(accuracy_score(Ytest, y_pred2))


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(Xtrain, Ytrain)
y_pred3=classifier.predict(Xtest)
print(confusion_matrix(Ytest,y_pred3))  
print(accuracy_score(Ytest, y_pred3))
'''