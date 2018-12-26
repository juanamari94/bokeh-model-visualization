import os

from model_loader import ModelLoader
from management_dashboard import ManagementDashboard

SERIALIZED_DATA_FILENAME = "modelling_data.pickle"

SERIALIZED_DATA_PATH = os.path.join(os.pardir, "serialized_data", SERIALIZED_DATA_FILENAME)

if __name__ == "__main__":
    loader = ModelLoader(SERIALIZED_DATA_PATH)
    xgb_clf, rf_clf, train, test = loader.assemble_serialized_data().values()
    dashboard = ManagementDashboard(xgb_clf, rf_clf, train, test)
    res = dashboard.daily_metrics("label")
    print(res)