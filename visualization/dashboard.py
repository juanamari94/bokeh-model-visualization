class Dashboard:
    def __init__(self, xgb_clf, rf_clf, train, test):
        self.xgb_clf = xgb_clf
        self.rf_clf = rf_clf
        self.train = train
        self.test = test