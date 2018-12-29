import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Dropdown, DateRangeSlider, Panel, Tabs, DataTable, TableColumn, Div
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

    def assemble_precision_recall_tab(self, source, metrics, model_dropdown, time_period_slider):
        precision_recall_f = figure(plot_width=700, plot_height=400, x_axis_type="datetime",
                                    title=self.xgb_clf.__name__)
        precision_recall_f.xaxis.axis_label = "Date"
        precision_recall_f.yaxis.axis_label = "Precision"

        precision_recall_f.line("date", "precision", source=source, color="orange", legend="Precision")
        precision_recall_f.line("date", "recall", source=source, color="green", legend="Recall")

        precision_recall_f.legend.location = "bottom_right"

        def update(attr, old, new):
            model = model_dropdown.value
            date_start, date_end = time_period_slider.value_as_datetime
            precision_recall_f.title.text = model
            model_metrics = metrics[model][date_start:date_end]
            source.data = ColumnDataSource.from_df(model_metrics)

        model_dropdown.on_change('value', update)
        time_period_slider.on_change('value', update)

        layout = row(precision_recall_f)
        return layout

    @staticmethod
    def _calculate_metrics(df):
        tp = df["tp"].sum()
        fp = df["fp"].sum()
        fn = df["fn"].sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        sample_size = df["sample_size"].sum()
        return pd.DataFrame({"precision": precision, "recall": recall, "f1_score": f1, "sample_size": sample_size},
                            index=[0])

    def assemble_metric_comparison_table(self, daily_metrics, model_dropdown, time_period_slider):
        model = model_dropdown.value
        date_start, date_end = time_period_slider.value_as_datetime
        target_df = daily_metrics[model][date_start:date_end]
        metric_data = self._calculate_metrics(target_df)

        source = ColumnDataSource(data=metric_data)

        columns = [
            TableColumn(field="precision", title="Precision"),
            TableColumn(field="recall", title="Recall"),
            TableColumn(field="f1_score", title="F1 Score"),
            TableColumn(field="sample_size", title="Sample Size")
        ]

        data_table = DataTable(source=source, columns=columns, width=600, height=280)
        title_div = Div()
        title_div.text = "Precision, Recall, F1 Score and Sample Size for <strong>{}</strong>".format(model)

        def update(attr, old, new):
            model = model_dropdown.value
            date_start, date_end = time_period_slider.value_as_datetime
            title_div.text = "Precision, Recall, F1 Score and Sample Size for <strong>{}</strong>".format(model)
            model_metrics = daily_metrics[model][date_start:date_end]
            source.data = ColumnDataSource.from_df(self._calculate_metrics(model_metrics))

        model_dropdown.on_change('value', update)
        time_period_slider.on_change('value', update)

        return row(column(title_div, data_table))

    def assemble_sample_size_over_time(self, daily_metrics, date_range_slider):
        sample_size_f = figure(plot_width=700, plot_height=400, x_axis_type="datetime",
                               title="Sample size over time")
        sample_size_f.xaxis.axis_label = "Date"
        sample_size_f.yaxis.axis_label = "Sample size"
        date_start, date_end = date_range_slider.value_as_datetime
        # We don't really care about the classifier here
        source_df = daily_metrics[self.xgb_clf.__name__]
        # Create a new column data source comprised of a Series object
        sample_size_cumsum = pd.DataFrame(source_df[date_start:date_end]["sample_size"].cumsum())
        cumsum_source = ColumnDataSource(data=sample_size_cumsum)
        sample_size_f.line("date", "sample_size", source=cumsum_source, color="cadetblue", legend="Sample size")

        def update(attr, old, new):
            date_start, date_end = date_range_slider.value_as_datetime
            sample_size_cumsum = pd.DataFrame(source_df[date_start:date_end].cumsum())
            cumsum_source.data = ColumnDataSource.from_df(sample_size_cumsum)

        date_range_slider.on_change('value', update)

        return row(sample_size_f)

    def assemble_global_widgets(self):
        model_menu = [(self.rf_clf.__name__, self.rf_clf.__name__),
                      None, (self.xgb_clf.__name__, self.xgb_clf.__name__)]

        model_dropdown = Dropdown(label="Model", button_type="warning", menu=model_menu, value=self.xgb_clf.__name__)

        vis_x_axis = self.test
        min_date, max_date = vis_x_axis.index.min(), vis_x_axis.index.max()

        time_period_slider = DateRangeSlider(title="Time Period", value=(min_date, max_date), start=min_date,
                                             end=max_date, step=1)

        return model_dropdown, time_period_slider

    def assemble_management_dashboard(self):
        model_dropdown, time_period_slider = self.assemble_global_widgets()
        daily_metrics = self.daily_metrics("label")
        daily_metrics_source = ColumnDataSource(data=daily_metrics[self.xgb_clf.__name__])
        precision_recall_layout = self.assemble_precision_recall_tab(daily_metrics_source,
                                                                     daily_metrics,
                                                                     model_dropdown,
                                                                     time_period_slider)
        metrics_table_layout = self.assemble_metric_comparison_table(daily_metrics, model_dropdown, time_period_slider)
        sample_size_layout = self.assemble_sample_size_over_time(daily_metrics, time_period_slider)
        pr_tab = Panel(child=precision_recall_layout, title="Precision and Recall over time")
        table_tab = Panel(child=metrics_table_layout, title="Metrics over time")
        sample_size_tab = Panel(child=sample_size_layout, title="Sample size over time")
        dashboard_layout = column(row(model_dropdown, time_period_slider),
                                  Tabs(tabs=[pr_tab, table_tab, sample_size_tab]),
                                  sizing_mode="scale_width")
        curdoc().add_root(dashboard_layout)

    def run(self):
        self.assemble_management_dashboard()
