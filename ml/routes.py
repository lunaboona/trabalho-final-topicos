from flask import render_template
from ml import app
from ml.forms import FormKNN, FormSVC, FormMLP, FormRandomForest, FormDecisionTree
from ml.iris import get_result

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

    if form.validate_on_submit():
        params = {};
        for key in form.data:
            if key in form.fields:
                params[key] = form.data[key]
        acc, macro_avg, cm_img = get_result(clf, **params)
        return render_template("classifier.html", classifier = clf, form = form, acc = acc, macro_avg = macro_avg, cm_img = cm_img)

    return render_template("classifier.html", classifier = clf, form = form)
