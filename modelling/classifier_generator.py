import logging
import time
import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold


def DEFAULT_SCORING_FUNC(tp, fp):
    return tp / (6 * fp + tp)


logger = logging.getLogger("ModelLogger")


class ClassifierGenerator:

    def __init__(self, name, clf, train, test, label_col="label", params=None):
        self.name = name
        self.params = params
        if params:
            self.clf = clf(**self.params)
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

    def predict_proba(self, X):
        probas = self.clf.predict_proba(X)[:, 1]
        res = pd.DataFrame(index=self.test.index, data={"probas": probas})
        return res

    def best_threshold_for_scoring_func(self, X, y, scoring_func=DEFAULT_SCORING_FUNC, thresholds=np.arange(0, 1, 0.1)):
        metrics = {thresh: self.calculate_metrics_for_threshold(X, y, scoring_func, thresh)
                   for thresh in thresholds}
        best_threshold = max(metrics, key=lambda x: metrics[x]["scoring_func_score"])
        return best_threshold

    def generate_curve_metrics(self):
        probas = self.clf.predict_proba(self.X_test)[:, 1]
        precision_recall_crv = precision_recall_curve(self.y_test, probas)
        roc_crv = roc_curve(self.y_test, probas)
        return {"precision_recall_curve": precision_recall_crv, "roc_curve": roc_crv}

    def generate_feature_importances(self):
        feature_names = self.X_train.columns
        feature_importances = self.clf.feature_importances_
        return pd.DataFrame({"name": feature_names, "importance": feature_importances})

    def generate_score_distributions(self, dataset, label_col, seed=1):
        n_splits = len(dataset) // 100
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=False)
        logger.info("Using {} splits".format(n_splits))
        scores = []
        dataset = dataset.drop_duplicates()
        for i, indices in enumerate(kf.split(dataset)):
            train_index, test_index = indices[0], indices[1]
            X = dataset.drop(columns=[label_col])
            y = dataset[label_col]
            logger.info("Iteration {}".format(i))
            start = time.monotonic()
            clf = self.clf.__class__(**self.params)
            clf.fit(X.iloc[train_index], y.iloc[train_index])
            end = time.monotonic()
            logger.debug("Iteration {} finished, time: {:0.3f} seconds".format(i, end - start))
            scores.extend(clf.predict_proba(X.iloc[test_index])[:, 1])
        logger.info("Length of the distribution: {}".format(len(scores)))
        return scores
