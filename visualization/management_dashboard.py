import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Dropdown, DateRangeSlider
from bokeh.plotting import figure

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

    def assemble_precision_recall_tab(self, source, metrics):
        precision_recall_f = figure(plot_width=700, plot_height=400, x_axis_type="datetime",
                                    title=self.xgb_clf.__name__)
        precision_recall_f.xaxis.axis_label = "Date"
        precision_recall_f.yaxis.axis_label = "Precision"

        precision_recall_f.line("date", "precision", source=source)

        def update(attr, old, new):
            model = model_dropdown.value
            date_start, date_end = time_period_slider.value_as_datetime
            precision_recall_f.title.text = model
            model_metrics = metrics[model][date_start:date_end]
            source.data = ColumnDataSource.from_df(model_metrics)

        model_menu = [(self.rf_clf.__name__, self.rf_clf.__name__),
                      None, (self.xgb_clf.__name__, self.xgb_clf.__name__)]

        model_dropdown = Dropdown(label="Model", button_type="warning", menu=model_menu, value=self.xgb_clf.__name__)

        vis_x_axis = self.test
        min_date, max_date = vis_x_axis.index.min(), vis_x_axis.index.max()

        time_period_slider = DateRangeSlider(title="Time Period", value=(min_date, max_date), start=min_date,
                                             end=max_date, step=1)

        model_dropdown.on_change('value', update)
        time_period_slider.on_change('value', update)

        layout = row(precision_recall_f, column(model_dropdown, time_period_slider))
        return layout

    def assemble_management_dashboard(self):
        daily_metrics = self.daily_metrics("label")
        daily_metrics_source = ColumnDataSource(data=daily_metrics[self.xgb_clf.__name__])
        precision_recall_layout = self.assemble_precision_recall_tab(daily_metrics_source, daily_metrics)
        curdoc().add_root(precision_recall_layout)

    def run(self):
        self.assemble_management_dashboard()
