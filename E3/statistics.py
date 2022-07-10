import numpy as np
from sklearn import metrics


class Statistics:
    def __init__(self):
        self.clean()

    def clean(self):
        self.all_y_true = []
        self.all_y_pred = []
        self.all_test_accuracy = []
        self.all_val_accuracy = []

    def add(self, test_accuracy, val_accuracy, y_pred, y_true):
        self.all_y_true.append(y_true)
        self.all_y_pred.append(y_pred)
        self.all_test_accuracy.append(test_accuracy)
        self.all_val_accuracy.append(val_accuracy)

    def _get_metrics(self, y_pred, y_true):
        f1_score_w = metrics.f1_score(y_true, y_pred, average="weighted")
        f1_score_m = metrics.f1_score(y_true, y_pred, average="macro")
        f1_score_per_class = metrics.f1_score(y_true, y_pred, average=None)
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        return f1_score_w, f1_score_m, f1_score_per_class, conf_matrix

    def _get_mean_std(self, a):
        return np.mean(a), np.std(a)

    def show(self):
        all_f1_score_w, all_f1_score_m, all_f1_score_per_class, all_conf_matrix = [], [], [], []
        for y_pred, y_true in zip(self.all_y_pred, self.all_y_true):
            f1_score_w, f1_score_m, f1_score_per_class, conf_matrix = self._get_metrics(y_pred, y_true)
            all_f1_score_w.append(f1_score_w)
            all_f1_score_m.append(f1_score_m)
            all_f1_score_per_class.append(f1_score_per_class)
            all_conf_matrix.append(conf_matrix)
        print("Test accuracy. Mean (std) = %.3f (%.3f)" % self._get_mean_std(self.all_test_accuracy))
        print("Val accuracy. Mean (std) = %.3f (%.3f)" % self._get_mean_std(self.all_val_accuracy))
        print("f1 score weighted. Mean (std) = %.3f (%.3f)" % self._get_mean_std(all_f1_score_w))
        print("f1 score mean. Mean (std) = %.3f (%.3f)" % self._get_mean_std(all_f1_score_m))