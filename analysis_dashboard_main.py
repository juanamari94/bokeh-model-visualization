import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "modelling"))
sys.path.append(os.path.join(os.path.dirname(__file__), "visualization"))
print(sys.path)

from model_loader import ModelLoader
from visualization.analysis_dashboard import AnalysisDashboard

SERIALIZED_DATA_FILENAME = "analysis_data.pickle"

SERIALIZED_DATA_PATH = os.path.join(os.path.dirname(__file__), "serialized_data", SERIALIZED_DATA_FILENAME)


def main():
    loader = ModelLoader(SERIALIZED_DATA_PATH)
    (xgb_clf_name, rf_clf_name, importances, daily_metrics, train_tsne, test_tsne, model_probas,
     score_distributions_train,
     score_distributions_test, curves) = loader.assemble_serialized_data().values()
    analysis_dashboard = AnalysisDashboard(xgb_clf_name, rf_clf_name, importances, daily_metrics, train_tsne, test_tsne,
                                           model_probas, score_distributions_train,
                                           score_distributions_test, curves)
    analysis_dashboard.run()


# dashboard = AnalysisDashboard(xgb_clf, rf_clf, train, test, daily_metrics)

main()
