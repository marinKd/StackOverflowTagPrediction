import neattext as nt
import neattext.functions as nfx
import pandas as pd
import pickle
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import LabelPowerset

questions = pd.read_csv('/Users/rose/Desktop/project2/StackOverflowTagPrediction/Questions.csv', encoding='latin-1')
tags = pd.read_csv('/Users/rose/Desktop/project2/StackOverflowTagPrediction/Tags.csv', encoding='latin-1')

questions.pop('OwnerUserId')
questions.pop('CreationDate')
questions.pop('ClosedDate')

tags['Tag'] = tags['Tag'].astype(str)
grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
grouped_tags.reset_index()

grouped_tags_final = pd.DataFrame({'Id': grouped_tags.index, 'Tag': grouped_tags.values})

df = questions.merge(grouped_tags_final, on='Id')
df = df[df['Score'] > 15]
questions.pop('Score')

df['Body'] = df['Body'].apply(lambda x: BeautifulSoup(x).get_text())
df["Text"] = df["Title"] + ", " + df["Body"]
df.pop('Title')
df.pop('Body')

df['Tag'] = df['Tag'].apply(lambda x: x.split())

# Taking all tags
all_tags = [item for sublist in df['Tag'].values for item in sublist]
len(all_tags)

# Taking unique tags
my_set = set(all_tags)
unique_tags = list(my_set)
len(unique_tags)


def most_common(tags):
    tags_filtered = []
    for i in range(0, len(tags)):
        if tags[i] in tags_features:
            tags_filtered.append(tags[i])
    return tags_filtered


flat_list = [item for sublist in df['Tag'].values for item in sublist]

keywords = nltk.FreqDist(flat_list)

keywords = nltk.FreqDist(keywords)

frequencies_words = keywords.most_common(4)
tags_features = [word[0] for word in frequencies_words]

df['Tag'] = df['Tag'].apply(lambda x: most_common(x))
df['Tag'] = df['Tag'].apply(lambda x: x if len(x) > 0 else None)

df.dropna(subset=['Tag'], inplace=True)

df1 = pd.DataFrame([])
df1 = df1.assign(**{k: 0 for k in tags_features})
df = pd.concat((df1, df), axis=1)
df.fillna(0, inplace=True)
df.pop('Score')
df = df.reset_index()
df.pop('index')
df.loc[12].iat[5]

for i in range(df.shape[0]):
    for j in range(len(tags_features)):
        for x in range(len(df.loc[i].iat[5])):
            if (df.loc[i].iat[5][x] == tags_features[j]):
                df.iat[i, j] = 1

df['Text'].apply(lambda x: nt.TextFrame(x).noise_scan())
df['Text'].apply(lambda x: nt.TextExtractor(x).extract_stopwords())
df['Text'] = df['Text'].apply(nfx.remove_stopwords)
df['Text'] = df['Text'].apply(lambda x: x.lower())
token = ToktokTokenizer()
lemma = WordNetLemmatizer()


def lemmatizeWords(text):
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))


df['Text'] = df['Text'].apply(lambda x: lemmatizeWords(x))

corpus = df['Text'].values
df['Tag'] = df['Tag'].apply(lambda x: ''.join(x))
y = df.loc[:, ['java', 'c#', 'javascript', 'android']]

# Initalize our vectorizer
tfidf = TfidfVectorizer(analyzer='word',
                        min_df=0.0,
                        max_df=1.0,
                        strip_accents=None,
                        encoding='utf-8',
                        preprocessor=None,
                        token_pattern=r"(?u)\S\S+",
                        max_features=100000)

X = tfidf.fit_transform(corpus)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# model = BinaryRelevance(MultinomialNB())

def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    clf_predictions = clf.predict(xtest)
    acc = accuracy_score(ytest, clf_predictions)
    ham = hamming_loss(ytest, clf_predictions)
    result = {"accuracy:": acc, "hamming_score": ham}
    return result, clf_predictions, clf

labelP_result, labelP_predictions, labelP_model = build_model(MultinomialNB(), LabelPowerset, X_train, y_train, X_test,
                                                              y_test)
labelP_result

#ex1 = df['Text'].iloc[0]
#vec_example = tfidf.transform([ex1])
# Save our vectorizer and model.
pickle.dump(tfidf, open('models/vectorizer.pkl', 'wb'))
print('Writing in vectorizer')
pickle.dump(labelP_model, open('models/text-classifier.pkl', 'wb'))
