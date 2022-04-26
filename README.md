# StackOverflowTagPrediction

StackOverflow Tag Prediction is a project that consists of Natural Language Processing (NLP), Machine Learning (ML), and the use of various data manipulation techniques. The premise is to predict tags on a technical question that the user inputs on a web interface.

This prediction will be achieved through Machine Learning, The ML model will be trained on a dataset that contains 10% of all StackOverflow questions, answers and tags at the time the dataset was published (2018). The questions, answers and tags are broken up in 3 .CSV files. Even at 10%, this is a sizable dataset and balancing it can present challenges. The two CSV files that we will be working with for our project are questions and tags. We excluded the answers.csv file from our project because of the repetitiveness between the questions and answers files. As well as the fact that the tags in the tags.csv files are the ones associated with the questions.

The interface is web based, built with Flask. The user will be able to type their question in a box and, upon submission, predicted tags will appear along with the confidence level.

The project is based on the contents taught at the Cuny Tech Prep (CTP) Data Science track. This track follows a curriculum that introduces several data science related categories such as NLP, Image Classifiers, data visualisation, Regression models, Neural Networks etc,. This project is going to be presented with industry professionals present at the end of the term.
Data Breakdown
●Questions.csv:

the questions file has 7 columns labeled as followed: ID, OwnerUserID, CreationDate, ClosedDate, Score, Title, and Body. It includes 1,264,216 rows of questions.
●Tags.csv:

the tags file has 2 columns, ID and Tag, and has 3,750,994 rows of data to work with
●Questions.csv

contains 10% of all the questions from Stack Overflow and the tags csv contains all the tags that are associated with those questions, many questions include multiple tags which is the reason for the difference in row numbers between the two files.
Technologies

● Python-- Main language

● Jupyter-- Data analysis platform

● Git/Github-- Version control

● Anaconda-- Distribution platform

● Pandas-- Data analysis

● scikit-learn-- Machine learning library

● NLTKPython-- Natural Language library

● Plotly-- Data visualisation

● Flask-- Web framework

● Heroku-- Deployment
