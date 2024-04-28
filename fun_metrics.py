import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



labels_str : list[str] = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']


def display_confusion_matrix(y_test : np.ndarray, y_pred : np.ndarray) -> None:
    '''
    This function is used to display the confusion matrix of the model
    
    Parameters
    ----------
    y_test : np.ndarray
        The real values
    y_pred : np.ndarray
        The predicted values
    
    Returns
    -------
    None
    '''

    matrix = confusion_matrix(y_test, y_pred, normalize='true')

    sns.heatmap(matrix, annot=True, cmap='coolwarm', xticklabels=labels_str, yticklabels=labels_str)

    plt.title('Confusion matrix')
    plt.xlabel('Predict values')
    plt.ylabel('Real values')
    plt.show()

def display_metrics_and_confusion_matrix(clf, X_test : np.ndarray, y_test : np.ndarray) -> None:
    '''
    This function is used to display the metrics and the confusion matrix of the model
    
    Parameters
    ----------
    clf :
        The model to use
    X_test : np.ndarray
        The test data
    y_test : np.ndarray
        The real values
        
    Returns
    -------
    None
    '''
    
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    print(f"Mean accuracy : {score}")
    print(classification_report(y_test, y_pred, zero_division=1)) # zero_division=1 -> To catch warning

    display_confusion_matrix(y_test, y_pred)



def metrics_compare(clf1, X1_test, y1_test, clf2, X2_test, y2_test, name1, name2) :

    pred1 = clf1.predict(X1_test)
    p1 = precision_score(y1_test, pred1, average=None)
    r1 = recall_score(y1_test, pred1, average=None)
    f1 = f1_score(y1_test, pred1, average=None)
    mean_p1 = precision_score(y1_test, pred1, average='macro')
    mean_r1 = recall_score(y1_test, pred1, average='macro')
    mean_f1 = f1_score(y1_test, pred1, average='macro')
    acc1 = accuracy_score(y1_test, pred1)
    means1 = [mean_p1, mean_r1, mean_f1, acc1]

    pred2 = clf2.predict(X2_test)
    p2 = precision_score(y2_test, pred2, average=None)
    r2 = recall_score(y2_test, pred2, average=None)
    f2 = f1_score(y2_test, pred2, average=None)
    mean_p2 = precision_score(y2_test, pred2, average='macro')
    mean_r2 = recall_score(y2_test, pred2, average='macro')
    mean_f2 = f1_score(y2_test, pred2, average='macro')
    acc2 = accuracy_score(y2_test, pred2)
    means2 = [mean_p2, mean_r2, mean_f2, acc2]

    indices = np.arange(len(p1)) # type: ignore
    bar_width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].bar(indices - bar_width/2, p1, bar_width, label=name1, color='skyblue')
    axes[0, 0].bar(indices + bar_width/2, p2, bar_width, label=name2, color='lightgreen')
    axes[0, 0].set_title('Precision')
    axes[0, 0].set_xticks(indices)
    axes[0, 0].set_xticklabels([f'Class {i+1}' for i in range(len(p1))]) # type: ignore
    axes[0, 0].legend()

    axes[1, 0].bar(indices - bar_width/2, f1, bar_width, label=name1, color='skyblue')
    axes[1, 0].bar(indices + bar_width/2, f2, bar_width, label=name2, color='lightgreen')
    axes[1, 0].set_title('f1-score')
    axes[1, 0].set_xticks(indices)
    axes[1, 0].set_xticklabels([f'Class {i+1}' for i in range(len(f1))]) # type: ignore
    axes[1, 0].legend()

    axes[0, 1].bar(indices - bar_width/2, r1, bar_width, label=name1, color='skyblue')
    axes[0, 1].bar(indices + bar_width/2, r2, bar_width, label=name2, color='lightgreen')
    axes[0, 1].set_title('Recall')
    axes[0, 1].set_xticks(indices)
    axes[0, 1].set_xticklabels([f'Class {i+1}' for i in range(len(r1))]) # type: ignore
    axes[0, 1].legend()

    axes[1, 1].bar(np.arange(len(means1)) - bar_width/2, means1, bar_width, label=name1, color='skyblue')
    axes[1, 1].bar(np.arange(len(means1)) + bar_width/2, means2, bar_width, label=name2, color='lightgreen')
    axes[1, 1].set_title('Model means')
    axes[1, 1].set_xticks(np.arange(len(means1)))
    axes[1, 1].set_xticklabels(['Precision', 'Recall', 'F1-score', 'Accuracy'])
    axes[1, 1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def keras_metrics_compare( clf1, X1_test, y1_test, clf2, X2_test, y2_test, name1, name2) :
    '''
    This function takes 2 keras models and their respective dataset, and displays barplots
    of different metrics to comapre the 2 models. name1 and name2 are for the lengend of the plot.

    Args:
        clf1, clf2 : SKlearn model
        X1_test, y1_test, X1_test, y1_test : np.ndarray
        name1, name2 : str

    Returns:
        Display the barplots.
    '''

    pred1_prob = clf1.predict(X1_test)
    pred1 = np.argmax(pred1_prob, axis=1)
    p1 = precision_score(y1_test, pred1, average=None)
    r1 = recall_score(y1_test, pred1, average=None)
    f1 = f1_score(y1_test, pred1, average=None)
    mean_p1 = precision_score(y1_test, pred1, average='macro')
    mean_r1 = recall_score(y1_test, pred1, average='macro')
    mean_f1 = f1_score(y1_test, pred1, average='macro')
    acc1 = accuracy_score(y1_test, pred1)
    means1 = [mean_p1, mean_r1, mean_f1, acc1]

    pred2_prob = clf2.predict(X2_test)
    pred2 = np.argmax(pred2_prob, axis=1)
    p2 = precision_score(y2_test, pred2, average=None)
    r2 = recall_score(y2_test, pred2, average=None)
    f2 = f1_score(y2_test, pred2, average=None)
    mean_p2 = precision_score(y2_test, pred2, average='macro')
    mean_r2 = recall_score(y2_test, pred2, average='macro')
    mean_f2 = f1_score(y2_test, pred2, average='macro')
    acc2 = accuracy_score(y2_test, pred2)
    means2 = [mean_p2, mean_r2, mean_f2, acc2]

    indices = np.arange(len(p1)) # type: ignore
    bar_width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].bar(indices - bar_width/2, p1, bar_width, label=name1, color='skyblue')
    axes[0, 0].bar(indices + bar_width/2, p2, bar_width, label=name2, color='lightgreen')
    axes[0, 0].set_title('Precision')
    axes[0, 0].set_xticks(indices)
    axes[0, 0].set_xticklabels([f'Class {i+1}' for i in range(len(p1))]) # type: ignore
    axes[0, 0].legend()

    axes[1, 0].bar(indices - bar_width/2, f1, bar_width, label=name1, color='skyblue')
    axes[1, 0].bar(indices + bar_width/2, f2, bar_width, label=name2, color='lightgreen')
    axes[1, 0].set_title('f1-score')
    axes[1, 0].set_xticks(indices)
    axes[1, 0].set_xticklabels([f'Class {i+1}' for i in range(len(f1))]) # type: ignore
    axes[1, 0].legend()

    axes[0, 1].bar(indices - bar_width/2, r1, bar_width, label=name1, color='skyblue')
    axes[0, 1].bar(indices + bar_width/2, r2, bar_width, label=name2, color='lightgreen')
    axes[0, 1].set_title('Recall')
    axes[0, 1].set_xticks(indices)
    axes[0, 1].set_xticklabels([f'Class {i+1}' for i in range(len(r1))]) # type: ignore
    axes[0, 1].legend()

    axes[1, 1].bar(np.arange(len(means1)) - bar_width/2, means1, bar_width, label=name1, color='skyblue')
    axes[1, 1].bar(np.arange(len(means1)) + bar_width/2, means2, bar_width, label=name2, color='lightgreen')
    axes[1, 1].set_title('Model means')
    axes[1, 1].set_xticks(np.arange(len(means1)))
    axes[1, 1].set_xticklabels(['Precision', 'Recall', 'F1-score', 'Accuracy'])
    axes[1, 1].legend(loc='lower right')

    plt.tight_layout()
    plt.show()