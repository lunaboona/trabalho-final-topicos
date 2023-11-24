import base64
from io import BytesIO
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

def get_classifier(name, **kwargs):
    match name:
      case 'knn':
          return KNeighborsClassifier(**kwargs)
      case 'svc':
          return SVC(**kwargs)
      case 'mlp':
          return MLPClassifier(**kwargs)
      case 'random_forest':
          return RandomForestClassifier(**kwargs)
      case 'decision_tree':
          return DecisionTreeClassifier(**kwargs)

def get_result(clf_name, **kwargs):
  iris = load_iris()
  X = iris.data
  y = iris.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  clf = get_classifier(clf_name, **kwargs)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  classes = iris.target_names.tolist()
  acc = round(accuracy_score(y_test, y_pred), 3)
  macro_avg = round(f1_score(y_test, y_pred, average="macro"), 3)
  cm_img = plot_confusion_matrix(y_test, y_pred,classes)
  return acc, macro_avg, cm_img
