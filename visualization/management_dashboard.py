import pandas as pd

from dashboard import Dashboard


class ManagementDashboard(Dashboard):

    def __init__(self, xgb_clf, rf_clf, train, test):
        super().__init__(xgb_clf, rf_clf, train, test)

    @staticmethod
    def calculate_precision_and_recall(clf, features, labels, thresh):
        metrics = clf.calculate_metrics_for_threshold(features, labels, threshold=thresh)
        return pd.Series(metrics)

    def daily_metrics(self, label_col):
        classifiers = [self.xgb_clf, self.rf_clf]

        best_thresholds = [
            clf.best_threshold_for_scoring_func(self.test.drop(columns=[label_col]),
                                                self.test[label_col])
            for clf in classifiers]
        metrics = {}
        for clf, thresh in zip(classifiers, best_thresholds):
            res = self.test.groupby(self.test.index).apply(
                lambda x: self.calculate_precision_and_recall(clf,
                                                              x.drop(columns=[label_col]),
                                                              x[label_col],
                                                              thresh)
            )
            metrics[clf.__name__] = pd.DataFrame(res)
        return metrics
