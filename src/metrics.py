import pickle
from collections import OrderedDict

from sklearn import metrics
import numpy as np
import pandas as pd


def pfbeta_np(labels, preds, beta=1):
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        return (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
    return 0.0

def get_best_f1(labels, preds, beta=1):
    # preds = np.power(preds, 0.5)
    percentiles = np.arange(0, 100, 0.1)
    threshs = [np.percentile(preds, perc) for perc in percentiles]
    best_score = 0
    best_thr = 0
    best_perc = 0
    scores = []
    for perc in percentiles:
        thresh = np.percentile(preds, perc)
        score = pfbeta_np(labels, preds > thresh, beta)
        if score > best_score:
            best_score = score
            best_thr = thresh
            best_perc = perc
        scores.append(score)

    try:
        rec_1 = metrics.recall_score(labels, preds > best_thr)
        rec_0 = metrics.recall_score(labels, preds > best_thr, pos_label=0)
        prc_1 = metrics.precision_score(labels, preds > best_thr)
        prc_0 = metrics.precision_score(labels, preds > best_thr, pos_label=0)
        roc_auc = metrics.roc_auc_score(labels, preds)
    except:
        roc_auc = 0.5
        rec_0, rec_1, prc_0, prc_1 = 0, 0, 0, 0

    res = OrderedDict(
        [
            ("F1", round(best_score, 4)),
            ("thresh", round(best_thr, 4)),
            ("ROC_AUC", round(roc_auc, 4)),
            ("rec_1", round(rec_1, 4)),
            ("prc_1", round(prc_1, 4)),
        ]
    )

    return res

def agg_default(df_test):
    df_sub = df_test.groupby("prediction_id")[["cancer"]].mean()
    df_trg = df_test.groupby("prediction_id")[["target"]].mean()
    agg_labels = df_trg.target.tolist()
    agg_preds = df_sub.cancer.tolist()

    agg_labels, agg_preds = np.array(agg_labels), np.array(agg_preds)
    return agg_labels, agg_preds

def agg_custom_1(df_tests):
    df_sub = df_test.groupby(["prediction_id", "view"])[["cancer"]].mean()
    df_trg = df_test.groupby(["prediction_id", "view"])[["target"]].max()
    df_sub = df_sub.groupby("prediction_id")[["cancer"]].max()
    df_trg = df_trg.groupby("prediction_id")[["target"]].max()
    agg_labels = df_trg.target.tolist()
    agg_preds = df_sub.cancer.tolist()
    agg_labels, agg_preds = np.array(agg_labels), np.array(agg_preds)
    return agg_labels, agg_preds

def save_pickle(name, data):
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(data, f)

def calc_metrics(df_test, beta=1):
    agg_labels, agg_preds = agg_default(df_test)

    labels = np.array(df_test.target.tolist())
    preds = np.array(df_test.cancer.tolist())

    res_single = get_best_f1(labels, preds, beta=1)
    res_agg = get_best_f1(agg_labels, agg_preds, beta=1)

    agg_str = "_agg"
    for key in res_agg:
        res_single[key + agg_str] = res_agg[key]

    return res_single

def calc_all_metrics(df_test, beta=1):
    overall = calc_metrics(df_test, beta)
    site1 = calc_metrics(df_test[df_test.site_id == 1], beta)
    site2 = calc_metrics(df_test[df_test.site_id == 2], beta)

    agg_str = "s1_"
    for key in site1:
        overall[agg_str + key] = site1[key]

    agg_str = "s2_"
    for key in site2:
        overall[agg_str + key] = site2[key]

    return overall


class AverageMeter:
    def __init__(self):
        self.metrics = None
        self.num_samples = 0

    def add(self, metrics, num_samples):
        self.num_samples += num_samples

        if isinstance(metrics, dict):
            if self.metrics is None:
                self.metrics = OrderedDict([(metric, 0) for metric in metrics])
            for metric in metrics:
                self.metrics[metric] += metrics[metric].item() * num_samples
        else:
            if self.metrics is None:
                self.metrics = 0
            self.metrics += metrics.item() * num_samples

    def get(self):
        if isinstance(self.metrics, dict):
            for metric in self.metrics:
                self.metrics[metric] = round(self.metrics[metric] / self.num_samples, 4)
        else:
            self.metrics = round(self.metrics / self.num_samples, 4)
        return self.metrics


class MetricCalculator:
    def __init__(self):
        self.preds = []
        self.target = []
        self.ids = []
        self.views = []
        self.site_ids = []

    def add(self, preds, target, ids, views, site_id):
        self.target.extend(target)
        self.ids.extend(ids)
        self.views.extend(views)
        self.site_ids.extend(site_id)

        if isinstance(preds, dict):
            if len(self.preds) == 0:
                self.preds = OrderedDict([(key, list(preds[key])) for key in preds])
            else:
                for key in self.preds:
                    self.preds[key].extend(preds[key])
        else:
            self.preds.extend(preds)

    def get(self):
        df_test = pd.DataFrame({"prediction_id": self.ids, "view": self.views, "cancer": self.preds, "target": self.target, "site_id": self.site_ids})
        result = calc_all_metrics(df_test)
        return result
