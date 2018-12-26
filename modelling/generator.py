import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from classifier_generator import ClassifierGenerator

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=format)
logger = logging.getLogger("ModelLogger")

DATA_DIR = os.path.join(os.getcwd(), "data_itc.csv")


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


def save(data, names, dest_dir):
    data_dict = {key: obj for key, obj in zip(names, data)}
    file_name = "modelling_data.pickle"
    if not os.path.exists(dest_dir) or not os.path.isdir(dest_dir):
        logger.info("Creating directory: {}".format(dest_dir))
        os.mkdir(dest_dir)
    file_path = os.path.join(dest_dir, file_name)
    logger.info("Saving to: {}".format(file_path))
    with open(file_path, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logger.info("Generating data")
    train, test = generate_data(DATA_DIR, ";")
    logger.info("Assembling classifiers")
    models = generate_classifiers(train, test)
    data = models + [train, test]
    names = ["xgb_clf", "rf_clf", "train_data", "test_data"]
    dest_dir = os.path.join(os.getcwd(), "serialized_data")
    logger.info("Saving trained models and data")
    save(data, names, dest_dir)
