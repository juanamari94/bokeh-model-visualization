import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "modelling"))
sys.path.append(os.path.join(os.path.dirname(__file__), "visualization"))
print(sys.path)

from visualization.management_dashboard import ManagementDashboard
from model_loader import ModelLoader

SERIALIZED_DATA_FILENAME = "management_data.pickle"

SERIALIZED_DATA_PATH = os.path.join(os.path.dirname(__file__), "serialized_data", SERIALIZED_DATA_FILENAME)


def main():
    loader = ModelLoader(SERIALIZED_DATA_PATH)
    xgb_clf, rf_clf, train, test, daily_metrics = loader.assemble_serialized_data().values()
    dashboard = ManagementDashboard(xgb_clf, rf_clf, train, test, daily_metrics)
    dashboard.run()


main()
