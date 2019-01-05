import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from xgboost.sklearn import XGBClassifier

from classifier_generator import ClassifierGenerator

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=format)
logger = logging.getLogger("ModelLogger")

DATA_DIR = os.path.join(os.getcwd(), "data_itc.csv")


def apply_tsne(data, n_components):
    tsne_data = data.reset_index().drop(columns=["label", "date"])
    labels = data["label"]
    tsne = TSNE(n_components=n_components, verbose=True)
    tsne_res = tsne.fit_transform(tsne_data.values)
    return pd.DataFrame({"comp1": tsne_res[:, 0], "comp2": tsne_res[:, 1], "labels": labels})


def generate_data(data_dir, delimiter):
    logger.info("Reading from: {}".format(data_dir))
    df = pd.read_csv(data_dir, delimiter=delimiter)
    df = df.set_index(pd.DatetimeIndex(df["date"])).drop(columns=["date"])
    max_month = df.index.month.max()
    train = df.loc[df.index.month < max_month]
    test = df.loc[df.index.month >= max_month]
    assert len(train) + len(test) == len(df), "Sum of splits does not have as many samples as the total samples."
    return train, test


def generate_classifiers(train, test):
    classifiers = [XGBClassifier, RandomForestClassifier]
    params = [dict(max_depth=2, learning_rate=1, objetive="binary:logistic"), dict(n_estimators=10)]
    models = []
    for classifier, params_ in zip(classifiers, params):
        logger.info("Generating classifier: {} with params: {}".format(classifier.__name__, params_))
        clf = ClassifierGenerator(classifier.__name__, classifier, train, test, params=params_)
        clf.fit()
        models.append(clf)
    return models


def calculate_precision_and_recall(clf, features, labels, thresh):
    metrics = clf.calculate_metrics_for_threshold(features, labels, threshold=thresh)
    return pd.Series(metrics)


def generate_daily_metrics(classifiers, test, label_col="label"):
    best_thresholds = [
        clf.best_threshold_for_scoring_func(test.drop(columns=[label_col]),
                                            test[label_col])
        for clf in classifiers]
    metrics = {}
    for clf, thresh in zip(classifiers, best_thresholds):
        res = test.groupby(test.index).apply(
            lambda x: calculate_precision_and_recall(clf,
                                                     x.drop(columns=[label_col]),
                                                     x[label_col],
                                                     thresh)
        )
        metrics[clf.__name__] = pd.DataFrame(res)
    return metrics


def save(data, names, dest_dir, file_name):
    data_dict = {key: obj for key, obj in zip(names, data)}
    if not os.path.exists(dest_dir) or not os.path.isdir(dest_dir):
        logger.info("Creating directory: {}".format(dest_dir))
        os.mkdir(dest_dir)
    file_path = os.path.join(dest_dir, file_name)
    logger.info("Saving to: {}".format(file_path))
    with open(file_path, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Main Data assembly
    logger.info("Generating data")
    train, test = generate_data(DATA_DIR, ";")
    logger.info("Assembling classifiers")
    models = generate_classifiers(train, test)
    logger.info("Assembling daily metrics")
    daily_metrics = generate_daily_metrics(models, test, "label")
    management_data = models + [train, test, daily_metrics]

    # Management Dashboard
    management_names = ["xgb_clf", "rf_clf", "train_data", "test_data", "daily_metrics"]
    dest_dir = os.path.join(os.getcwd(), "serialized_data")
    logger.info("Saving management dashboard data")
    save(management_data, management_names, dest_dir, "management_data.pickle")

    # Data Analysis Dashboard
    logger.info("Assembling feature importances")
    importances = {model.__name__: model.generate_feature_importances() for model in models}
    logger.info("Applying TSNE to the training set")
    train_tsne = apply_tsne(train, 2)
    logger.info("Applying TSNE to the test set")
    test_tsne = apply_tsne(test, 2)
    logger.info("Calculating probabilities for the test set")
    model_probas = {model.__name__: model.predict_proba(test.drop(columns=["label"])) for model in models}
    logger.info("Generating score distributions for the training set")
    score_distributions_train = {model.__name__: model.generate_score_distributions(train, "label") for model in models}
    logger.info("Generating score distributions for the test set")
    score_distributions_test = {model.__name__: model.generate_score_distributions(test, "label") for model in models}
    logger.info("Generating Precision and Recall Curve and ROC Curve")
    curves = {model.__name__: model.generate_curve_metrics() for model in models}

    data_analysis_names = ["xgb_clf_name", "rf_clf_name", "feature_importances", "daily_metrics", "train_tsne",
                           "test_tsne", "test_probabilities",
                           "train_distributions", "test_distributions", "curves"]
    data_analysis_data = [model.__name__ for model in models] + [importances, daily_metrics, train_tsne, test_tsne,
                                                                 model_probas, score_distributions_train,
                                                                 score_distributions_test, curves]
    logger.info("Saving data analysis data")
    save(data_analysis_data, data_analysis_names, dest_dir, "analysis_data.pickle")
