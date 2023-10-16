import json, os, time
import numpy as np


def oneshotstl(y, T):
    with open(r"D:/project/python/demo38_lsmdecomposition/algorithm/utils/input.json", "r") as f:
        jd = json.load(f)
    jd["ts"] = y.tolist()
    jd["period"] = T
    jd["trainTestSplit"] = 5 * T
    jd["trend"] = []
    jd["seasonal"] = []
    jd["residual"] = []

    with open(r"D:/project/python/demo38_lsmdecomposition/algorithm/utils/input.json", "w") as f:
        json.dump(jd, f)

    cmd = r'java -jar D:/project/python/demo38_lsmdecomposition/algorithm/utils/OneShotSTL/OneShotSTL.jar --method OneShotSTL --task decompose --shiftWindow 0 --in D:/project/python/demo38_lsmdecomposition/algorithm/utils/input.json --out D:/project/python/demo38_lsmdecomposition/algorithm/utils/output.json'

    _ = os.system(cmd)

    with open(r"D:/project/python/demo38_lsmdecomposition/algorithm/utils/output.json", "r") as f:
        jd = json.load(f)
    trend = jd["trend"]
    seasonal = jd["seasonal"]
    residual = jd["residual"]

    return trend, seasonal, residual


def oneshotstl_flush_impute(x):
    # head and tail
    size = len(x) - 1
    start_i, end_i = 0, size
    while np.isnan(x[start_i]):
        start_i += 1
    for i in range(start_i):
        x[i] = x[start_i]
    while np.isnan(x[end_i]):
        end_i -= 1
    for i in range(end_i + 1, size):
        x[i] = x[end_i]

    # body
    for i in range(start_i, end_i):
        if np.isnan(x[i]):
            left_i = i - 1
            while np.isnan(x[i]):
                i += 1
            right_i = i
            for j in range(left_i + 1, right_i):
                for _ in range(4):  #
                    x[j] = (x[right_i] - x[left_i]) / (right_i - left_i) * \
                           (j - left_i) + x[left_i]

    return x


def oneshotstl_query_concat(xs, t_maxs, t_mins, isnans):
    x_concat = np.array(xs[0])
    size = len(xs)
    for i in range(1, size):
        # adjacent
        if t_maxs[i - 1] + 1 == t_mins[i]:
            # 1. x,isnan: direct concat
            x_concat = np.concatenate((x_concat, xs[i]))
        # disjoint
        elif t_maxs[i - 1] + 1 < t_mins[i]:
            # 1. x: concat with imputation
            x_interval_impute = np.zeros(t_mins[i] - t_maxs[i - 1] - 1)
            # 1.3 impute
            for idx, ts in enumerate(range(t_maxs[i - 1] + 1, t_mins[i])):
                x_interval_impute[idx] = (x_concat[-1] + xs[i][0]) / 2

            x_concat = np.concatenate((x_concat, x_interval_impute, xs[i]))
        # overlapped
        else:
            # 1. reconstruct the concat result
            x_concat_temp = np.zeros(t_maxs[i] - t_mins[0] + 1)
            for idx, ts in enumerate(range(t_mins[0], t_maxs[i] + 1)):
                if ts < t_mins[i]:  # pre
                    x_concat_temp[idx] = x_concat[ts - t_mins[0]]
                elif ts > t_maxs[i - 1]:  # now
                    x_concat_temp[idx] = xs[i][ts - t_mins[i]]
                # pre not nan and now nan
                elif i > 0 and isnans[i][ts - t_mins[i]] and not isnans[i - 1][ts - t_mins[i - 1]]:
                    x_concat_temp[idx] = x_concat[ts - t_mins[0]]
                else:
                    x_concat_temp[idx] = xs[i][ts - t_mins[i]]
            x_concat = x_concat_temp
    return x_concat
