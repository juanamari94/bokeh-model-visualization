import math

import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Dropdown, Slider, Panel, Tabs, DateRangeSlider
from bokeh.plotting import figure
from sklearn.metrics import precision_score, recall_score


class AnalysisDashboard:

    def __init__(self,
                 xgb_clf_name,
                 rf_clf_name,
                 importances,
                 daily_metrics,
                 train_tsne,
                 test_tsne,
                 model_probas,
                 score_distributions_train,
                 score_distributions_test, curves):
        self.xgb_clf_name = xgb_clf_name
        self.rf_clf_name = rf_clf_name
        self.importances = importances
        self.daily_metrics = daily_metrics
        self.train_tsne = train_tsne
        self.test_tsne = test_tsne
        self.model_probas = model_probas
        self.test_labels = self.test_tsne["labels"]
        self.train_labels = self.train_tsne["labels"]
        self.score_distributions_train = score_distributions_train
        self.score_distributions_test = score_distributions_test
        self.curves = curves

        self.model_dropdown, self.threshold_slider = self.assemble_global_widgets()

    def assemble_global_widgets(self):
        model_menu = [(self.rf_clf_name, self.rf_clf_name),
                      None, (self.xgb_clf_name, self.xgb_clf_name)]

        model_dropdown = Dropdown(label="Model", button_type="warning", menu=model_menu, value=self.xgb_clf_name)

        threshold_slider = Slider(start=0, end=1, value=0.5, step=.01, title="Threshold value")

        return model_dropdown, threshold_slider

    @staticmethod
    def daily_precision_and_recall_for_threshold(threshold, probas, labels):
        probas["labels"] = labels
        probas["y_pred"] = probas.apply(lambda x: 1 if x["probas"] >= threshold else 0, axis=1)
        precision_s = probas.groupby(probas.index).apply(lambda x: precision_score(x["labels"], x["y_pred"]))
        precision = pd.DataFrame(index=precision_s.index, data=precision_s, columns=["precision"])
        recall_s = probas.groupby(probas.index).apply(lambda x: recall_score(x["labels"], x["y_pred"]))
        recall = pd.DataFrame(index=recall_s.index, data=recall_s, columns=["recall"])
        return {"precision": precision, "recall": recall}

    def assemble_precision_recall_tab(self):
        dropdown_value = self.model_dropdown.value
        threshold_value = self.threshold_slider.value
        precision_and_recall = self.daily_precision_and_recall_for_threshold(threshold_value,
                                                                             self.model_probas[dropdown_value],
                                                                             self.test_labels)
        daily_metrics = self.daily_metrics[dropdown_value]
        vis_x_axis = daily_metrics.index
        min_date, max_date = vis_x_axis.min(), vis_x_axis.max()

        time_period_slider = DateRangeSlider(title="Time Period", value=(min_date, max_date), start=min_date,
                                             end=max_date, step=1)

        precision_recall_f = figure(plot_width=700, plot_height=400, x_axis_type="datetime",
                                    title=dropdown_value)

        precision_recall_f.xaxis.axis_label = "Date"
        precision_recall_f.yaxis.axis_label = "Precision"

        prec_source = ColumnDataSource(precision_and_recall["precision"])
        rec_source = ColumnDataSource(precision_and_recall["recall"])

        precision_recall_f.line("date", "precision", source=prec_source, color="orange", legend="Precision")
        precision_recall_f.line("date", "recall", source=rec_source, color="green", legend="Recall")

        precision_recall_f.legend.location = "bottom_right"

        def update(attr, old, new):
            model = self.model_dropdown.value
            threshold = self.threshold_slider.value
            date_start, date_end = time_period_slider.value_as_datetime
            precision_recall_f.title.text = model
            precision_and_recall = self.daily_precision_and_recall_for_threshold(threshold, self.model_probas[model],
                                                                                 self.test_labels)
            prec_metrics = precision_and_recall["precision"][date_start:date_end]
            recall_metrics = precision_and_recall["recall"][date_start:date_end]
            prec_source.data = ColumnDataSource.from_df(prec_metrics)
            rec_source.data = ColumnDataSource.from_df(recall_metrics)

        self.model_dropdown.on_change('value', update)
        time_period_slider.on_change('value', update)
        self.threshold_slider.on_change('value', update)

        layout = column(time_period_slider, precision_recall_f)
        return layout

    def assemble_feature_importances_tab(self):
        dropdown_value = self.model_dropdown.value

        model_importances = self.importances[dropdown_value]

        importances_source = ColumnDataSource(data=model_importances)

        importances_f = figure(plot_width=700, plot_height=400,
                               title=dropdown_value, x_range=model_importances["name"])

        importances_f.xaxis.major_label_orientation = math.pi / 2

        importances_f.xaxis.axis_label = "Feature name"
        importances_f.yaxis.axis_label = "Importance"

        importances_f.vbar(x="name", top="importance", source=importances_source, width=0.5)

        def update(attr, old, new):
            dropdown_value = self.model_dropdown.value
            importances_f.title.text = self.model_dropdown.value
            importances_source.data = ColumnDataSource.from_df(self.importances[dropdown_value])

        self.model_dropdown.on_change('value', update)

        return row(importances_f)

    def assemble_curves_tab(self):
        dropdown_value = self.model_dropdown.value

        model_curves = self.curves[dropdown_value]

        prc_precision = model_curves["precision_recall_curve"][0]
        prc_recall = model_curves["precision_recall_curve"][1]
        prc_dict = {'precision': prc_precision, 'recall': prc_recall}

        roc_fpr = model_curves["roc_curve"][0]
        roc_tpr = model_curves["roc_curve"][1]
        roc_dict = {'fpr': roc_fpr, 'tpr': roc_tpr}

        prc_f = figure(plot_width=700, plot_height=400,
                       title=dropdown_value + " - Precision Recall Curve")
        prc_f.xaxis.axis_label = "Recall"
        prc_f.yaxis.axis_label = "Precision"

        roc_f = figure(plot_width=700, plot_height=400,
                       title=dropdown_value + " - ROC Curve")

        roc_f.xaxis.axis_label = "False Positive Rate"
        roc_f.yaxis.axis_label = "True Positive Rate"

        prc_source = ColumnDataSource(prc_dict)
        roc_source = ColumnDataSource(roc_dict)

        roc_f.line(x="fpr", y="tpr", source=roc_source)
        prc_f.line(x="recall", y="precision", source=prc_source)
        roc_f.line(x=np.arange(0, 1.1, 0.1), y=np.arange(0, 1.1, 0.1), line_dash="dashed")
        prc_f.line(x=np.arange(0, 1.1, 0.1), y=np.arange(1.0, -0.1, -0.1), line_dash="dashed")

        def update(attr, old, new):
            dropdown_value = self.model_dropdown.value
            model_curves = self.curves[dropdown_value]

            prc_precision = model_curves["precision_recall_curve"][0]
            prc_recall = model_curves["precision_recall_curve"][1]
            prc_dict = {'precision': prc_precision, 'recall': prc_recall}

            roc_fpr = model_curves["roc_curve"][0]
            roc_tpr = model_curves["roc_curve"][1]
            roc_dict = {'fpr': roc_fpr, 'tpr': roc_tpr}

            prc_source.data = prc_dict
            roc_source.data = roc_dict

            roc_f.title.text = dropdown_value + " - ROC Curve"
            prc_f.title.text = dropdown_value + " - Precision Recall Curve"

        self.model_dropdown.on_change('value', update)

        return row(roc_f, prc_f)

    def assemble_score_distributions(self):
        xgb_test_dist = self.score_distributions_test["XGBClassifier"]
        xgb_train_dist = self.score_distributions_train["XGBClassifier"]

        rf_test_dist = self.score_distributions_test["RandomForestClassifier"]
        rf_train_dist = self.score_distributions_train["RandomForestClassifier"]

        test_distribution_f = figure(plot_width=700, plot_height=400,
                                     title="Histogram for scores over the test set (log10 scale)")
        train_distribution_f = figure(plot_width=700, plot_height=400,
                                      title="Histogram for scores over the train set (log10 scale)")

        xgb_test_hist, xgb_test_edges = np.histogram(xgb_test_dist, density=True, bins=100)
        rf_test_hist, rf_test_edges = np.histogram(rf_test_dist, density=True, bins=100)
        xgb_train_hist, xgb_train_edges = np.histogram(xgb_train_dist, density=True, bins=100)
        rf_train_hist, rf_train_edges = np.histogram(rf_train_dist, density=True, bins=100)

        test_distribution_f.quad(top=np.abs(np.log10(xgb_test_hist)), bottom=0, left=xgb_test_edges[:-1],
                                 right=xgb_test_edges[1:],
                                 fill_color="navy", legend="XGBClassifier")
        test_distribution_f.quad(top=np.abs(np.log10(rf_test_hist)), bottom=0, left=rf_test_edges[:-1],
                                 right=rf_test_edges[1:],
                                 fill_color="red", alpha=0.5, legend="RandomForestClassifier")

        train_distribution_f.quad(top=np.abs(np.log10(xgb_train_hist)), bottom=0, left=xgb_train_edges[:-1],
                                  right=xgb_train_edges[1:],
                                  fill_color="navy", legend="XGBClassifier")
        train_distribution_f.quad(top=np.abs(np.log10(rf_train_hist)), bottom=0, left=rf_train_edges[:-1],
                                  right=rf_train_edges[1:],
                                  fill_color="red", alpha=0.5, legend="RandomForestClassifier")

        return row(test_distribution_f, train_distribution_f)

    def assemble_tsne(self):
        tsne_test = figure(plot_width=700, plot_height=400, title="TSNE for the test set")
        tsne_train = figure(plot_width=700, plot_height=400, title="TSNE for the train set")

        for label, color in zip(self.test_tsne["labels"].unique(), ["red", "blue"]):
            tsne_test_on_label = self.test_tsne[self.test_tsne["labels"] == label]
            tsne_train_on_label = self.train_tsne[self.train_tsne["labels"] == label]
            tsne_test.circle(x=tsne_test_on_label["comp1"], y=tsne_test_on_label["comp2"], legend=str(label),
                             color=color, size=1)
            tsne_train.circle(x=tsne_train_on_label["comp1"], y=tsne_train_on_label["comp2"], legend=str(label),
                             color=color, size=1)

        return row(tsne_test, tsne_train)

    def assemble_analysis_dashboard(self):
        global_widgets = row(self.model_dropdown, self.threshold_slider)
        importances_layout = self.assemble_feature_importances_tab()
        feature_importances_tab = Panel(child=importances_layout, title="Feature importances")
        curves_layout = self.assemble_curves_tab()
        curves_tab = Panel(child=curves_layout, title="Curves: Precision and Recall, ROC")
        daily_precision_recall_layout = self.assemble_precision_recall_tab()
        daily_precision_recall_tab = Panel(child=daily_precision_recall_layout, title="Daily Precision and Recall")
        score_dist_layout = self.assemble_score_distributions()
        score_dist_tab = Panel(child=score_dist_layout, title="Score distribution")
        tsne_layout = self.assemble_tsne()
        tsne_tab = Panel(child=tsne_layout, title="TSNE Scatter Plots")
        curdoc().add_root(column(global_widgets,
                                 Tabs(tabs=[feature_importances_tab,
                                            curves_tab,
                                            daily_precision_recall_tab,
                                            score_dist_tab,
                                            tsne_tab]),
                                 sizing_mode="scale_width"))

    def run(self):
        self.assemble_analysis_dashboard()
