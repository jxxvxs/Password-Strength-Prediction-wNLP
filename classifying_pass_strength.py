




"""## Main Import"""

# import warnings
# warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import seaborn as sns





"""## Creating dataset var"""

floc = 'datasets.csv'


data=pd.read_csv(floc,error_bad_lines=False)
data.head()





"""## Checking dataset values"""

data['strength'].unique()

data[data['password'].isnull()]

data.isnull().sum()

data.isna().sum()

data.dropna(inplace=True)

sns.countplot(data['strength'])





"""## Creating array of the data"""

array_of_password = np.array(data)


print(array_of_password)





"""## Randomizing array of the data"""

import random


random.shuffle(array_of_password)

password_ = 0
strength_ = 1


x=[labels[password_] for labels in array_of_password]
y=[labels[strength_] for labels in array_of_password]

x

y





"""## Function to convert string to a list"""

def string_divide_fn(string):
    list_ =[]
    for char in string:
        list_.append(char)
    return list_


string_divide_fn('Test_123(=*~/')





"""## TF*IDF vectorizer to convert and applying it to data"""

from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer=TfidfVectorizer(tokenizer=string_divide_fn)

X=vectorizer.fit_transform(x)

X.shape

vectorizer.get_feature_names()

first_document_vector=X[0]


first_document_vector

first_document_vector.T.todense()

df=pd.DataFrame(first_document_vector.T.todense(),  index=vectorizer.get_feature_names(),  columns=['TF*IDF'])


df.sort_values(by=['TF*IDF'],  ascending=False)





"""## Splitting with train_test_split"""

from sklearn.model_selection import train_test_split


X_train,  X_test,  Y_train,  Y_test    = train_test_split(X,y,test_size=0.2)


X_train.shape





"""## Applying Logistic Regression on data"""

from sklearn.linear_model import LogisticRegression


clf   = LogisticRegression(random_state=0,  multi_class='multinomial')


clf.fit(X_train,  Y_train)





"""## Checking prediction"""

temp_   = np.array( [   'i+Y*()VH12#L'   ] )


_temp   = vectorizer.transform(temp_)
clf.predict(_temp)





"""## Running prediction on X_test"""

Y_pred  = clf.predict(X_test)


print(Y_pred)





"""## Model Accuracy"""

from sklearn.metrics import confusion_matrix,accuracy_score


conm  = confusion_matrix(Y_test,Y_pred)


print(conm)

accuracy_score(Y_test,Y_pred)





"""## Model Report"""

from sklearn.metrics import classification_report


print(classification_report(Y_test,Y_pred))