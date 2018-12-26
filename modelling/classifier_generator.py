import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

DEFAULT_SCORING_FUNC = lambda tp, fp: tp / (6 * fp + tp)


class ClassifierGenerator:

    def __init__(self, name, clf, train, test, label_col="label", params=None):
        self.name = name
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()
        self.train = train
        self.test = test
        self.label_col = label_col
        self.X_train, self.y_train = self.train.drop(columns=[self.label_col]), self.train[self.label_col]
        self.X_test, self.y_test = self.test.drop(columns=[self.label_col]), self.test[self.label_col]
        self.__name__ = clf.__name__

    def fit(self):
        assert len(self.X_train) == len(self.y_train), "Sample size differs."
        self.clf.fit(self.X_train, self.y_train)
        return self.clf

    def calculate_metrics_for_threshold(self,
                                        X,
                                        y,
                                        scoring_func=DEFAULT_SCORING_FUNC,
                                        threshold=0.5):
        y_pred = self.predict_for_threshold(X, threshold)
        assert len(y_pred) == len(y), "Prediction length does not match sample length."

        conf_mat = confusion_matrix(y, y_pred, labels=self.clf.classes_)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        tp = conf_mat[0, 0]
        fp = conf_mat[0, 1]
        fn = conf_mat[1, 0]
        tn = conf_mat[1, 1]
        scoring_func_score = scoring_func(tp, fp)

        metrics_dict = {'sample_size': len(y), 'f1_score': f1, 'precision': precision, 'recall': recall, 'tp': tp,
                        'fp': fp, 'fn': fn, 'tn': tn, 'scoring_func_score': scoring_func_score, "threshold": threshold}

        return metrics_dict

    def predict_for_threshold(self, X, threshold=0.5):
        probas = self.clf.predict_proba(X)
        return [0 if proba < threshold else 1 for proba in probas[:, 1]]

    def best_threshold_for_scoring_func(self, X, y, scoring_func=DEFAULT_SCORING_FUNC, thresholds=np.arange(0, 1, 0.1)):
        metrics = {thresh: self.calculate_metrics_for_threshold(X, y, scoring_func, thresh)
                   for thresh in thresholds}
        best_threshold = max(metrics, key=lambda x: metrics[x]["scoring_func_score"])
        return best_threshold
