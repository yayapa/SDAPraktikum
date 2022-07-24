import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score


class Statistics:
    def __init__(self):
        self.clean()

    def clean(self):
        self.all_y_true = []
        self.all_y_pred = []

    def add(self, y_pred, y_true):
        self.all_y_true.append(y_true)
        self.all_y_pred.append(y_pred)

    def get_metrics(self, prediction, y_test, show=False, avg='weighted'):
        acc = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average=avg)
        recall = recall_score(y_test, prediction, average=avg)
        f1 = f1_score(y_test, prediction, average=avg)
        matthew = matthews_corrcoef(y_test, prediction)
        gmean = geometric_mean_score(y_test, prediction)
        if show:
            print("Accuracy Score -> ", acc)
            print("Precision Score -> ", precision)
            print("Recall Score -> ", recall)
            print("F1 Score -> ", f1)
            print("Matthews Correlation Coefficient -> ", matthew)
            print("G Mean Score -> ", gmean)
        return acc, precision, recall, f1, matthew, gmean

    def get_mean_std(self, a):
        return np.mean(a), np.std(a)

    def show(self, avg='weighted'):
        all_acc, all_precision, all_recall, all_f1, all_matthew, all_gmean = [], [], [], [], [], []
        for y_pred, y_true in zip(self.all_y_pred, self.all_y_true):
            acc, precision, recall, f1, matthew, gmean = self.get_metrics(y_pred, y_true, False, avg)
            all_acc.append(acc)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_matthew.append(matthew)
            all_gmean.append(gmean)
        print("Accuracy Score -> ", self.get_mean_std(all_acc))
        print("Precision Score -> ", self.get_mean_std(all_precision))
        print("Recall Score -> ", self.get_mean_std(all_recall))
        print("F1 Score -> ", self.get_mean_std(all_f1))
        print("Matthews Correlation Coefficient -> ", self.get_mean_std(all_matthew))
        print("G Mean Score -> ", self.get_mean_std(all_gmean))

