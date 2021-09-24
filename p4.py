import matplotlib.pyplot as plt
import itertools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


def graphs(y_pred, y_test):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_pred[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_pred.ravel()
    )
    average_precision["micro"] = average_precision_score(y_test, y_pred, average="micro")

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"]
    )
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")

    plt.show()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = itertools.cycle(['red', 'blue', 'green'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='Class {0} (AUC = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

gnb = GaussianNB()

Y_pred = gnb.fit(X_train, Y_train).predict(X_test)

y_pred = label_binarize(Y_pred, classes=[0, 1, 2])
y_test = label_binarize(Y_test, classes=[0, 1, 2])

print(confusion_matrix(Y_test, Y_pred))

# ROC and precision-recall code adapted from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
graphs(y_pred, y_test)

clf = LinearDiscriminantAnalysis(solver='svd')

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))

y_pred = label_binarize(Y_pred, classes=[0, 1, 2])
y_test = label_binarize(Y_test, classes=[0, 1, 2])

graphs(y_pred, y_test)

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

print(confusion_matrix(Y_test, Y_pred))

y_pred = label_binarize(Y_pred, classes=[0, 1, 2])
y_test = label_binarize(Y_test, classes=[0, 1, 2])

graphs(y_pred, y_test)
