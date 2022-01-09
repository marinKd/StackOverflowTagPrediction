import flask
import os
import pickle

import numpy as np
import pandas as pd
from skimage import io
from skimage import transform

app = flask.Flask(__name__, template_folder='templates')

path_to_vectorizer = 'models/vectorizer.pkl'
path_to_text_classifier = 'models/text-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f:
    tfdif = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']

        # Turn the text into numbers using our vectorizer
        X = tfdif.transform([user_input_text])

        # Make a prediction 
        predictions = model.predict(X).toarray()

        # Get the value of the predicted probabilities
        xpercent_java = model.predict_proba(X)[:, 0]
        xpercent_c = model.predict_proba(X)[:, 1]
        xpercent_javascript = model.predict_proba(X)[:, 2]
        xpercent_android = model.predict_proba(X)[:, 3]

        for i in range(4):
            if predictions[0][i] == 1:
                if i == 0:
                    prediction = "Java"
                if i == 1:
                    prediction = "C#"
                if i == 2:
                    prediction = "Javascript"
                if i == 3:
                    prediction = "Android"

        return flask.render_template('index.html',
                                     input_text=user_input_text,
                                     result=prediction,
                                     percent_java=xpercent_java,
                                     percent_c=xpercent_c,
                                     percent_javascript=xpercent_javascript,
                                     percent_android=xpercent_android
                                     )

@app.route('/input_values/', methods=['GET', 'POST'])
def input_values():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return (flask.render_template('input_values.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        var_one = flask.request.form['input_variable_one']
        var_two = flask.request.form['another-input-variable']
        var_three = flask.request.form['third-input-variable']

        list_of_inputs = [var_one, var_two, var_three]

        return (flask.render_template('input_values.html',
                                      returned_var_one=var_one,
                                      returned_var_two=var_two,
                                      returned_var_three=var_three,
                                      returned_list=list_of_inputs))

    return (flask.render_template('input_values.html'))


# @app.route('/images/')
# def images():
#    return flask.render_template('images.html')


@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')


if __name__ == '__main__':
    app.run(debug=True)
