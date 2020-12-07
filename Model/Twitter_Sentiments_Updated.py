""" Importing Libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score,accuracy_score


"""Loading the Datasets and Sample Submission"""
df_train=pd.read_csv(r"C:\Users\DEBIPRASAD\Desktop\Projetc Work\Twitter Sentiments\train_2kmZucJ.csv")
df_test=pd.read_csv(r"C:\Users\DEBIPRASAD\Desktop\Projetc Work\Twitter Sentiments\test_oJQbWVk.csv")
df_sample=pd.read_csv(r"C:\Users\DEBIPRASAD\Desktop\Projetc Work\Twitter Sentiments\sample_submission_LnhVWA4.csv")



""" Positive Comments """
print("Positive Comments")
print(df_train[df_train['label']==0].head(5))



""" Negative Comments """
print("Negative Comments")
print(df_train[df_train['label']==1].head(4))

"""Count of the Positive(0) Tweets and Negative(1) Tweets"""
df_train.label.value_counts().plot.bar(color='cyan',figsize=(8,6))
plt.show()


""" Histogram Visualization as per length"""
df_train.tweet.str.len().plot.hist(color='pink',figsize=(6,4))
df_test.tweet.str.len().plot.hist(color='green',figsize=(6,4))
plt.show()


"""Object for CountVectorizer"""
cv=CountVectorizer(stop_words='english') 
cv.fit(df_train.tweet)
words=cv.transform(df_train.tweet)
"""Cumulative Sum of the Words"""
sum_words=words.sum(axis=0)

"""Tuple of Counters Produced for Each Words"""
words_counter=[(word,sum_words[0,i]) for word,i in cv.vocabulary_.items()]
words_counter=sorted(words_counter,key=lambda x:x[1],reverse=True)

""" DataFrame Mode"""
df_words_counter=pd.DataFrame(words_counter,columns=['words','count'])
"""View the DataFrame"""
print(df_words_counter.head(4))

""" Top 15 Most Frequently Used Words"""
df_words_counter.head(15).plot(x='words',y='count',kind='bar',color='pink')
plt.show()

""" Least 15 Frequently Used Words"""
df_words_counter.tail(15).plot(x='words',y='count',kind='bar',color='cyan')
plt.show()


""" Finding Good and Bad Hashtags """
""" Finding Hashtags for Non Racist and Non Sexist Comments"""
Hash_pos=[]
for i in df_train[df_train['label']==0].tweet:
    ht=re.findall(r'#(\w+)',i)
    Hash_pos.append(ht)
#Count of Words where Label is Positive
Hash_pos=sum(Hash_pos,[])

"""Finding Hashtags for Racist and Sexist Comments"""
Hash_neg=[]
for i in df_train[df_train['label']==1].tweet:
    ht=re.findall(r'#(\w+)',i)
    Hash_neg.append(ht)
Hash_neg=sum(Hash_neg,[])

pos=nltk.FreqDist(Hash_pos)
df_pos=pd.DataFrame({'Positive Hashtags':pos.keys(),'Count':pos.values()})
neg=nltk.FreqDist(Hash_neg)
df_neg=pd.DataFrame({'Negative Hashtags':neg.keys(),'Count':neg.values()})

print(df_pos.head(6))
print(df_neg.head(6))

sr=stopwords.words('english')

train_corpus = []
for i in range(0, df_train.shape[0]):
    df_train_i=word_tokenize(df_train.tweet[i])
    df_train_i=df_train_i[:]
    for j in df_train_i:
        if j in sr:
            df_train_i.remove(j)
    review=re.sub('[^a-zA-Z]',' ',' '.join(df_train_i))
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    # stemming
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # joining them back with space
    review=' '.join(review)
    train_corpus.append(review)
  
test_corpus=[]
for i in range(0, df_test.shape[0]):
    df_test_i=word_tokenize(df_test.tweet[i])
    df_test_i=df_test_i[:]
    for j in df_test_i:
        if j in sr:
            df_test_i.remove(j)
    review=re.sub('[^a-zA-Z]',' ',' '.join(df_test_i))
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    # stemming
    # stemming is time taking
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # joining them back with space
    review =' '.join(review)
    test_corpus.append(review)
  
cv2=CountVectorizer(max_features=2800)
x_train=cv2.fit_transform(train_corpus).toarray()
y_train=df_train.label.values

x_test=cv2.transform(test_corpus).toarray()
test_id=df_test.id.values

""" Splitting of Training Set and Feature Scaling """
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=0.25,random_state=42)
ssc=StandardScaler()
ssc.fit(x_train)
x_train=ssc.transform(x_train)
x_validation=ssc.transform(x_validation)
x_test=ssc.transform(x_test)

""" Decision Tree Model """
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_train_pred=dt.predict(x_train)
y_validation_pred=dt.predict(x_validation)
print("DT Train Efficiency ",end=" ")
print(accuracy_score(y_train,y_train_pred))
print("DT Test Efficiency ",end=" ")
print(accuracy_score(y_validation,y_validation_pred))
print("DT F1_Score ",end=" ")
print(f1_score(y_validation,y_validation_pred))

y_test=dt.predict(x_test)
df_dt_submi=pd.DataFrame()
df_dt_submi['id']=list(test_id)
df_dt_submi['label']=list(y_test)
df_dt_submi.to_csv('submission_dt.csv')

""" End """

""" Logistic Regression """
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_train_pred=lr.predict(x_train)
y_validation_pred=lr.predict(x_validation)
print("LR Train Efficiency ",end=" ")
print(accuracy_score(y_train,y_train_pred))
print("LR Test Efficiency ",end=" ")
print(accuracy_score(y_validation,y_validation_pred))
print("LR F1_Score ",end=" ")
print(f1_score(y_validation,y_validation_pred))

y_test=lr.predict(x_test)
df_lr_submi=pd.DataFrame()
df_lr_submi['id']=list(test_id)
df_lr_submi['label']=list(y_test)
df_lr_submi.to_csv('submission_lr.csv')

""" End """

""" SVM """
sv=svm.SVC()
sv.fit(x_train,y_train)
y_train_pred=sv.predict(x_train)
y_validation_pred=sv.predict(x_validation)
print("SVM Train Efficiency ",end=" ")
print(accuracy_score(y_train,y_train_pred))
print("SVM Test Efficiency ",end=" ")
print(accuracy_score(y_validation,y_validation_pred))
print("SVM F1_Score ",end=" ")
print(f1_score(y_validation,y_validation_pred))

y_test=sv.predict(x_test)
df_svm_submi=pd.DataFrame()
df_svm_submi['id']=list(test_id)
df_svm_submi['label']=list(y_test)
df_svm_submi.to_csv('submission_svm.csv')

""" End """

""" Random Forest Classifier """
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_train_pred=rf.predict(x_train)
y_validation_pred=rf.predict(x_validation)
print("RF Train Efficiency ",end=" ")
print(accuracy_score(y_train,y_train_pred))
print("RF Test Efficiency ",end=" ")
print(accuracy_score(y_validation,y_validation_pred))
print("RF F1_Score ",end=" ")
print(f1_score(y_validation,y_validation_pred))

y_test=rf.predict(x_test)
df_rf_submi=pd.DataFrame()
df_rf_submi['id']=list(test_id)
df_rf_submi['label']=list(y_test)
df_rf_submi.to_csv('submission_rf.csv')

""" End """