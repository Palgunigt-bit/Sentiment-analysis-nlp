# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""
 
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('Reviews.csv')

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

def removeApostrophe(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"can\'t", "can not", review)
    phrase = re.sub(r"n\'t", " not", review)
    phrase = re.sub(r"\'re", " are", review)
    phrase = re.sub(r"\'s", " is", review)
    phrase = re.sub(r"\'d", " would", review)
    phrase = re.sub(r"\'ll", " will", review)
    phrase = re.sub(r"\'t", " not", review)
    phrase = re.sub(r"\'ve", " have", review)
    phrase = re.sub(r"\'m", " am", review)
    return phrase

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

def removeHTMLTags(review):
    soup = BeautifulSoup(review, 'lxml')
    return soup.get_text()

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

def removeSpecialChars(review):
    return re.sub('[^a-zA-Z]', ' ', review)

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

def removeAlphaNumericWords(review):
    return re.sub("\S*\d\S*", "", review).strip()

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

def doTextCleaning(review):
    review = removeHTMLTags(review)
    review = removeApostrophe(review)
    review = removeAlphaNumericWords(review)
    review = removeSpecialChars(review) 

    review = review.lower()  # Lower casing
    review = review.split()  # Tokenization
    
    #Removing Stopwords and Lemmatization
    lmtzr = WordNetLemmatizer()
    review = [lmtzr.lemmatize(word, 'v') for word in review if not word in set(stopwords.words('english'))]
    
    review = " ".join(review)    
    return review


import nltk 
nltk.download('wordnet')

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""
from tqdm import tqdm

corpus = []   
for index, row in tqdm(dataset.iterrows()):
    review = doTextCleaning(row['Text'])
    corpus.append(review)

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# Creating the transform with Tri-gram
cv = CountVectorizer(ngram_range=(1,3), max_features = 2)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,6].values

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

# Creating Naive Bayes classifier
classifier = GaussianNB()

# Fitting the training set into the Naive Bayes classifier
classifier.fit(X_train, y_train)

# -*- coding: utf-8 -*-
"""
NLP sentiment analysis in python
"""

#Predict sentiment for new Review
def predictNewReview():
    newReview = input("Type the Review: ")
    
    if newReview =='':
        print('Invalid Review')  
    else:
        newReview = doTextCleaning(newReview)
        reviewVector = cv.transform([newReview]).toarray()  
        prediction =  classifier.predict(reviewVector)
        if prediction[0] == 1:
            print( "Positive Review" )
        else:        
            print( "Negative Review")