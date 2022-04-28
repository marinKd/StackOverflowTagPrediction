features makes it quite difficult. To add, the tags themselves are mostly mutually
exclusive, i.e a C++ question cannot also be a Java question.
The interface is web based, built with Flask. The Flask interface takes input and submits
it to the model, and imports the model results. The user is able to type their question in
a box and, upon submission, predicted tags appear along with the confidence level.
The project is based on the contents taught at the Cuny Tech Prep (CTP) Data Science
track. This track follows a curriculum that introduces several data science related
categories such as NLP, Image Classifiers, Data Visualisation, Regression models,
Neural Networks etc,. This project was presented with industry professionals present at
the end of the term.

To format the data and make it suitable for the models input, the unique tags were split
into columns. The presence of the tag for each question was marked with a 1, whereas
the tags that were not present were marked with 0.
For the questions, the body text and title were merged together. The title is likely to
contain important features that should make the model more accurate, so it was not
dropped. The text went through a pipeline of lemmatization, stopword removal,
uncapitalization, and tokenization. Afterwards, TfidfVectorizer was initiated on the text.
Lastly, a train/test split of 30% of the data was performed on the text and tags.
Several ML models were built, in order to compare and pick the best performing one.
Binary Relevance, Classifier Chains, and Label Powerset were the models that were
trained and tested. Binary Relevance and Classifier Chains both returned an accuracy
score of 24%, while MultinomialNB with Label Powerset returned 84%, with 7% hamming loss. The reason
for the huge difference is difficult to understand. Label Powerset is a problem
transformation approach to multi-label classification that transforms a multi-label
problem to a multi-class problem with 1 multi-class classifier trained on all unique label
combinations found in the training data.

The original idea was to create a multi-label prediction model, as in multiple
non-mutually exclusive predictions per question. However, to achieve the right data
balance for multi-label prediction proved to be extremely difficult. The sheer amount of
Project Description
StackOverflow Tag Prediction is a project that consists of Natural Language Processing
(NLP), Machine Learning (ML), and the use of various data manipulation techniques.
The premise is to predict tags on a technical question that the user inputs on a web
interface. This prediction is achieved through Machine Learning, specifically, 
a Naive Bayes classification for multinomial modes, with Label Powerset transformation.

The ML model is trained on a dataset that contains 10% of all StackOverflow questions,
answers and tags at the time the dataset was published on kaggle.com (2018). The
questions, answers and tags are broken up in 3 .CSV files. Even at 10%, this is a
sizable dataset and balancing it presents challenges. The two CSV files that we work
with are questions.csv and tags.csv. We excluded the answers.csv file from our project
because of the repetitiveness between the questions and answers files. As well as the
fact that the tags in the tags.csv files are the ones associated with the questions. Also
excluded were all the rows that had a StackOverflow score of less than 15.
The 4 most popular tags (Java, C#, Javascript, Android) were selected for prediction.
This was a decision made when testing out the models and compromising between the
number of classes and a satisfactory accuracy.

