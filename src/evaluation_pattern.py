import os
import sys
import json
import numpy as np
from utils import get_best_epochs, compute_mae, compute_rmse, compute_p_r_f1, compute_tp
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    model_dir = '../dumps/generate/'
    with open(os.path.join(model_dir, "test93.json"), "r") as f:
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
                print("pattern: "+ pattern+"\t"+"node: "+ node+"\t"+"test-RMSE: %.4f\ttest-MAE: %.4f\ttest-F1_Zero: %.4f\ttest-F1_NonZero: %.4f "% (
                    compute_rmse(pred, counts), compute_mae(pred, counts),
                    compute_p_r_f1(pred < 0.5, counts < 0.5)[2], compute_p_r_f1(pred > 0.5, counts > 0.5)[2]))
        print("test-Time: %.4f" %(results["time"]["total"]))