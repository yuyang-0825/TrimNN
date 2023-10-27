import os
import sys
import json
import numpy as np
from utils import get_best_epochs, compute_mae, compute_rmse, compute_p_r_f1, compute_tp, compute_recall_precision, \
    compute_confusion_matrix, compute_f1_mcc

import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    model_dir = '../dumps/ablation/'
    with open(os.path.join(model_dir, "split4_test89.json"), "r") as f:
        results = json.load(f)
        results["data"]["pattern"] = []
        results["data"]["node"] = []
        for i in range(len(results["data"]["id"])):
            results["data"]["pattern"].append('_'.join(results["data"]["id"][i].split("-")[0].split("_")[0:3]))
            results["data"]["node"].append(results["data"]["id"][i].split("G_")[1].split("_")[0])
        pattern_type = set(results["data"]["pattern"])
        node_type = set(results["data"]["node"])
        result_data = pd.DataFrame(results["data"])
        for pattern in pattern_type:
            for node in node_type:
                result_data_sub = result_data[(result_data["pattern"]==pattern) & (result_data["node"]==node)]
                pred = np.array(result_data_sub["pred"])
                counts = np.array(result_data_sub["counts"])
                exist = np.array(np.where(result_data_sub["counts"] > 0, 1.0, 0.0))
                pred = np.where(pred > 0, 1.0, 0.0)
                precision, recall = compute_recall_precision(pred, exist)
                f1, mcc = compute_f1_mcc(pred, exist)
                if len(pred) != 0:
                    tn, fp, fn, tp = compute_confusion_matrix(pred, exist)
                else:
                    tn, fp, fn, tp = [0,0,0,0]

                # print("pattern: "+ pattern+"\t"+"node: "+ node+"\t"+"precision: %.4f\trecall: %.4f\ttest-F1_Zero: %.4f\ttest-F1_NonZero: %.4f "% (
                #     precision, recall,
                #     compute_p_r_f1(pred < 0.5, counts < 0.5)[2], compute_p_r_f1(pred > 0.5, counts > 0.5)[2]))
                print(
                    "pattern: " + pattern + "\t" + "node: " + node + "\t" + "tn: %d\t fp: %d\t fn: %d\t tp: %d\t precision: %.4f\t recall: %.4f\t f1: %.4f\t mcc: %.4f" % (
                        tn, fp, fn, tp, precision, recall, f1, mcc))
        print("test-Time: %.4f" %(results["time"]["total"]))