Sentiment Analysis using NLP

This project performs sentiment analysis on customer reviews using Natural Language Processing and Machine Learning.
The model predicts whether a review is Positive or Negative.

Problem Statement
Customer reviews contain useful insights about products and services. By analyzing them, companies can improve product quality, increase sales, and reduce customer churn.
This project uses the Amazon Food Reviews Dataset to build a sentiment classifier.

Steps in the Project

Data Preprocessing

Remove punctuations, HTML tags, special characters

Tokenization and Lemmatization

Remove stopwords

Feature Extraction

Bag of Words model with N grams (1 to 3)

Model Building

Train Test split

Naive Bayes Classifier

Prediction

Input a new review

Model predicts Positive or Negative

Installation

Clone the repository
git clone :

cd sentiment-analysis-nlp

Install dependencies
pip install -r requirements.txt

Download dataset from Kaggle Amazon Fine Food Reviews and save it as Reviews.csv in the project folder

Run the Project
python nlp.py

Example Prediction
Type the Review: The food was excellent and delivery was fast
Output: Positive Review

Type the Review: The product quality is very poor
Output: Negative Review

Conclusion
We applied text preprocessing, feature extraction using bag of words with tri grams, and trained a Naive Bayes Classifier to predict customer sentiments as Positive or Negative.