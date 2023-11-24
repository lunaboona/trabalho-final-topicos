import os

from flask import flash, get_flashed_messages, render_template, redirect, url_for

from ml import app
from ml.forms import FormKNN, FormSVC, FormMLP, FormRandomForest, FormDecisionTree


@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    classifiers = ['knn', 'svc', 'mlp', 'random_forest', 'decision_tree']
    return render_template("home.html", classifiers = classifiers)


@app.route('/classifier/<clf>', methods=['POST', 'GET'])
def classifier(clf):
    match clf:
        case 'knn':
            form = FormKNN();
        case 'svc':
            form = FormSVC();
        case 'mlp':
            form = FormMLP();
        case 'random_forest':
            form = FormRandomForest();
        case 'decision_tree':
            form = FormDecisionTree();
    return render_template("classifier.html", classifier = clf, form = form)


    # if form.validate_on_submit():
    #     params = form.data;
    #     # pegar o texto
    #     _post_text = _formNewPost.text.data


    # return render_template("profile.html", user=current_user, form=_formNewPost, users=users)
