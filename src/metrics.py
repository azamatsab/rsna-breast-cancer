from collections import OrderedDict

from sklearn import metrics
import numpy as np


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
    threshs = np.arange(0, 1, 0.02)
    best_score = 0
    best_thr = 0
    scores = []
    for thresh in threshs:
        score = pfbeta_np(labels, preds > thresh, beta)
        if score > best_score:
            best_score = score
            best_thr = thresh
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

    # for thr, score in zip(threshs, scores):
    #     res[f"{round(thr, 3)}"] = round(score, 3)

    return res

def cacl_all_metrics(labels, preds, ids, beta=1):
    preds_dict = {pred_id: [] for pred_id in ids}
    targ_dict = {pred_id: [] for pred_id in ids}

    for pred_id, pred, target in zip(ids, preds, labels):
        preds_dict[pred_id].append(pred)
        targ_dict[pred_id].append(target)
    
    agg_labels, agg_preds = [], []
    for pred_id in ids:
        agg_preds.append(np.max(preds_dict[pred_id]))
        agg_labels.append(np.max(targ_dict[pred_id]))
    agg_labels, agg_preds = np.array(agg_labels), np.array(agg_preds)

    res_single = get_best_f1(labels, preds, beta=1)
    res_agg = get_best_f1(agg_labels, agg_preds, beta=1)

    agg_str = "_agg"
    for key in res_agg:
        res_single[key + agg_str] = res_agg[key]

    # thr = res_single["thresh"]
    # res_single[f"F1_{thr}"] = round(pfbeta_np(agg_labels, agg_preds > thr, beta), 4)
    return res_single


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

    def add(self, preds, target, ids):
        self.target.extend(target)
        self.ids.extend(ids)

        if isinstance(preds, dict):
            if len(self.preds) == 0:
                self.preds = OrderedDict([(key, list(preds[key])) for key in preds])
            else:
                for key in self.preds:
                    self.preds[key].extend(preds[key])
        else:
            self.preds.extend(preds)

    def get(self):
        if isinstance(self.preds, dict):
            result = OrderedDict()
            for key in self.preds:
                result[key] = cacl_all_metrics(np.array(self.target), np.array(self.preds[key], self.ids))
        else:
            result = cacl_all_metrics(np.array(self.target), np.array(self.preds), self.ids)
        return result
