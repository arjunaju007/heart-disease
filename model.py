##import ml libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#import dataset
dataset = pd.read_csv(r'C:\Users\joeka\Desktop\Heart_disease_machine learning\heart.csv')

##split dataset++
from sklearn.model_selection import train_test_split

##define target class
predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

##importing ml algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

##training model
rf.fit(X_train,Y_train)
 
y_pred=rf.predict(X_test)

# Constructing the confusion matrix.
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(Y_test, y_pred)
print(conf_mat)