#!/usr/bin/env python
# coding: utf-8




import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import pandas as pd
import numpy as np




df = pd.read_csv('Reviews.csv')
df.head()




fig = px.histogram(df, x="score")
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Sentiment Score')
fig.show()




df['sentiment'] = df['score'].apply(lambda rating : +1 if rating == 1 else -1)
positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]




df['sentimentt'] = df['sentiment'].replace({-1 : 'negative'})
df['sentimentt'] = df['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(df, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()





dfNew = df[['review','sentiment']]
dfNew.head()




index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.6]
test = df[df['random_number'] > 0.6]





from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['review'])
test_matrix = vectorizer.transform(test['review'])





from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=5000)





X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']




lr.fit(X_train,y_train)




predictions = lr.predict(X_test)



from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)




print(classification_report(predictions,y_test))



#Regularization terms added below



from sklearn.preprocessing import StandardScaler



# Create a scaler object
sc = StandardScaler(with_mean=False)

# Fit the scaler to the training data and transform
X_train_std = sc.fit_transform(X_train)

# Apply the scaler to the test data
X_test_std = sc.transform(X_test)



C = [10, 1, .1, .001]

for c in C:
    clf = LogisticRegression(penalty='l1', C=c, solver='liblinear')
    clf.fit(X_train, y_train)
    print('C:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf.score(X_train_std, y_train))
    print('Test accuracy:', clf.score(X_test_std, y_test))
    print('')



new_lr= LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
new_lr.fit(X_train, y_train)




new_pred=new_lr.predict(X_test_std)




new_2 = np.asarray(y_test)
confusion_matrix(new_pred,y_test)




print('Training accuracy:',new_lr.score(X_train_std, y_train))
print('Test accuracy:', new_lr.score(X_test_std, y_test))




print(classification_report(new_pred,y_test))




test_review = vectorizer.transform(["Test Any Review Text"])
prediction=new_lr.predict_proba(test_review)
for a in prediction:
    if a[0]>a[1]:
        print('Negative')
    else:
        print('Positive')
print(prediction)




from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, new_lr)




from lime.lime_text import LimeTextExplainer
class_names = ['Negative', 'Positive']
explainer = LimeTextExplainer(class_names = class_names)




from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer 
from nltk.corpus import stopwords

def text_to_words(selftext):

    text = BeautifulSoup(selftext).get_text()

    lower_case = text.lower()

    retokenizer = RegexpTokenizer(r'[a-z]+')
    word_tokens = retokenizer.tokenize(lower_case)

    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer() 
    tokens_lem = [lemmatizer.lemmatize(i) for i in word_tokens]

    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in tokens_lem if not w in stops]

    return(" ".join(meaningful_words))

def pre_process (X):

    # prints out number of posts
    num_text = X.size
    clean_text = []
    print(f'Number of posts: {num_text}\n')

    print("Cleaning and parsing posts...")

    for i in range(0, num_text):
        # if the index is evenly divisible by 500, print a message
        if((i+1) % 500 == 0):
            print('Review %d of %d' % ( i+1, num_text))                                                                    
        clean_text.append(text_to_words(X.iloc[i]))

    return clean_text




predstring='Predict any individual text entry'
    
   

    

    

import nltk
nltk.download('stopwords')


predstring=predstring.lower()





print(predstring)



idx = 10
exp = explainer.explain_instance(predstring, c.predict_proba, num_features=1)
print('Document id: %d' % idx)





exp.as_list()




get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()

