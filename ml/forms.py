from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired

class FormKNN(FlaskForm):
    fields = ['n_neighbors']
    n_neighbors = IntegerField('n_neighbors', default=3, validators=[DataRequired()])
    btn = SubmitField('Executar')

class FormSVC(FlaskForm):
    fields = []
    btn = SubmitField('Executar')

class FormMLP(FlaskForm):
    fields = ['random_state', 'max_iter']
    random_state = IntegerField('random_state', default=1, validators=[DataRequired()])
    max_iter = IntegerField('max_iter', default=300, validators=[DataRequired()])
    btn = SubmitField('Executar')

class FormRandomForest(FlaskForm):
    fields = ['max_depth', 'random_state']
    max_depth = IntegerField('max_depth', default=2, validators=[DataRequired()])
    random_state = IntegerField('random_state', default=1, validators=[DataRequired()])
    btn = SubmitField('Executar')

class FormDecisionTree(FlaskForm):
    fields = ['max_leaf_nodes', 'random_state']
    max_leaf_nodes = IntegerField('max_leaf_nodes', default=4, validators=[DataRequired()])
    random_state = IntegerField('random_state', default=1, validators=[DataRequired()])
    btn = SubmitField('Executar')
