import time

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from algorithm.LSMSTLAlg import *
from algorithm.OneShotSTLAlg import oneshotstl
from algorithm.RobustSTLAlg import robuststl
from algorithm.Preprocessing import *

# from observe import batch_lsm, batch_pre

fontsize = 16


def generate_syn1(x_size=1000):
    t = np.arange(x_size)
    trend = t * 0.1

    seasonal = np.sin(t / 6 * np.pi)

    residual = np.zeros(x_size)
    residual[80] -= 12
    y = seasonal + trend + residual
    T = 12

    return y, t, T, trend, seasonal, residual


def generate_syn2(x_size=1000):
    t = np.arange(x_size)
    trend = -t * 0.1
    # trend = -t * 0

    seasonal = np.tile(np.concatenate([np.full(6, 1.), np.full(6, -1.)]), x_size // 12 + 1)[:x_size]
    # seasonal = signal.square(t / 6 * np.pi)

    for i in range(len(seasonal)):
        if i % 12 == 9:
            seasonal[i] += 2.
    residual = np.zeros(x_size)
    # residual = np.random.random(x_size) * 0.01
    y = seasonal + trend + residual
    T = 12

    return y, t, T, trend, seasonal, residual


def draw_pic_combined(tau1, s1, r1, tau2, s2, r2, tau3, s3, r3, tau4, s4, r4, save):
    figsize = (25.6, 4.8)
    fig, ax = plt.subplots(3, 4, figsize=figsize)

    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(tau1, color="#000000")
    plt.title("(a) Truth", fontsize=fontsize + 2)
    plt.xticks([], fontsize=fontsize)
    plt.ylabel("Trend", fontsize=fontsize)
    if save == "syn1":
        plt.yticks([2, 6, 10], fontsize=fontsize)
    else:
        plt.yticks([-6, -9, -12], fontsize=fontsize)

    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(tau2, color="#000000")
    plt.title("(b) LSM-STL", fontsize=fontsize + 2)
    plt.xticks([], fontsize=fontsize)
    if save == "syn1":
        plt.yticks([2, 6, 10], fontsize=fontsize)
    else:
        plt.yticks([-6, -9, -12], fontsize=fontsize)

    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(tau3, color="#000000")
    plt.title("(c) OneShotSTL", fontsize=fontsize + 2)
    plt.xticks([], fontsize=fontsize)
    if save == "syn1":
        plt.yticks([2, 6, 10], fontsize=fontsize)
    else:
        plt.yticks([-6, -9, -12], fontsize=fontsize)

    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(tau4, color="#000000")
    plt.title("(d) RobustSTL", fontsize=fontsize + 2)
    plt.xticks([], fontsize=fontsize)
    if save == "syn1":
        plt.yticks([2, 6, 10], fontsize=fontsize)
    else:
        plt.yticks([-6, -9, -12], fontsize=fontsize)

    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(s1, color="#000000")
    plt.xticks([], fontsize=fontsize)
    plt.ylabel("Seasonal", fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-9, -3, 3], fontsize=fontsize)
    else:
        plt.yticks([-1, 0, 1], fontsize=fontsize)

    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(s2, color="#000000")
    plt.xticks([], fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-9, -3, 3], fontsize=fontsize)
    else:
        plt.yticks([-1, 0, 1], fontsize=fontsize)

    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(s3, color="#000000")
    plt.xticks([], fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-9, -3, 3], fontsize=fontsize)
    else:
        plt.yticks([-1, 0, 1], fontsize=fontsize)

    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(s4, color="#000000")
    plt.xticks([], fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-9, -3, 3], fontsize=fontsize)
    else:
        plt.yticks([-1, 0, 1], fontsize=fontsize)

    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(r1, color="#000000")
    plt.xticks(fontsize=fontsize)
    plt.ylabel("Residual", fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-15, -5, 5], fontsize=fontsize)
    else:
        plt.yticks([-5, 0, 5], fontsize=fontsize)

    ax10 = plt.subplot(3, 4, 10)
    ax10.plot(r2, color="#000000")
    plt.xticks(fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-15, -5, 5], fontsize=fontsize)
    else:
        plt.yticks([-5, 0, 5], fontsize=fontsize)
    ax11 = plt.subplot(3, 4, 11)
    ax11.plot(r3, color="#000000")
    plt.xticks(fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-15, -5, 5], fontsize=fontsize)
    else:
        plt.yticks([-5, 0, 5], fontsize=fontsize)
    ax12 = plt.subplot(3, 4, 12)
    ax12.plot(r4, color="#000000")
    plt.xticks(fontsize=fontsize)
    if save == "syn1":
        plt.yticks([-15, -5, 5], fontsize=fontsize)
    else:
        plt.yticks([-5, 0, 5], fontsize=fontsize)

    # fig.show()
    fig.savefig('./figures/exp_effect_' + save + '.eps', dpi=900, format='eps', bbox_inches='tight')


def write_to_file(string):
    f = open("./input/synthetic.txt", "a")
    f.write(string)
    f.close()


def main_syn():
    page_size = 30000

    # 1. data acquisition
    for dataset_idx in [1, 2]:
        x_size = 300
        if dataset_idx == 1:
            x, t, M, tau_truth, s_truth, r_truth = generate_syn1(x_size)
            save = "syn1"
            # lsm
            lambda_t, lambda_s = 100000.0, 1.6
            # robust
            reg1, reg2, K, H, dn1, dn2, ds1, ds2 = 0.5, 10, 20, 0, 1., 1., 50., 5.8
        else:
            x, t, M, tau_truth, s_truth, r_truth = generate_syn2(x_size)
            save = "syn2"
            # lsm
            lambda_t, lambda_s = 100000.0, 1.
            # robust
            # robust
            reg1, reg2, K, H, dn1, dn2, ds1, ds2 = .000001, 1000.5, 20, 50000, 1., 1., 5., .4

        print(save, "==========================")
        with open("./results/" + save + ".txt", "w") as f:
            for i in x:
                f.write(str(i) + "\n")

        # query range
        x, t, tau_truth, s_truth, r_truth = x, t, tau_truth[5 * M:], s_truth[5 * M:], r_truth[5 * M:]
        q_b, q_e = 5 * M, x_size

        # 2. data noising
        x, t, size = missing(x, t, missing_rate, missing_length)
        x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

        # 3. cold start
        x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
        cs = ColdStart(x=x_cs, M=M)
        v = cs.return_v()

        # 4. ldlt-decomposition
        coefficients = ldlt(N=x_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t, lambda_s=lambda_s)
        head_impute_size = coefficients.rtn_converge_size() + 2

        # 5. forward substitution
        z_pages, x_pages, t_max_pages, t_min_pages, isnan_pages = [], [], [], [], []
        insert_time_sum = 0.0
        for page_idx in range(len(x) // page_size + 1):
            x_page, t_page = reconstruct(x[page_idx * page_size:(page_idx + 1) * page_size],
                                         t[page_idx * page_size:(page_idx + 1) * page_size])

            # calculate
            insert_time_start = time.time()
            forward = ForwardSubstitution(coefficients=coefficients, x=x_page, t=t_page, v=v, M=M,
                                          hi_size=head_impute_size)
            insert_time_end = time.time()

            insert_time_sum += insert_time_end - insert_time_start

            # record
            z_pages.append(forward.return_z())
            x_pages.append(forward.return_x_imputed())
            t_max_pages.append(np.max(t_page))
            t_min_pages.append(np.min(t_page))
            isnan_pages.append(forward.return_isnan())

        # 6. concatenation
        q_b, q_e, pn_b, pn_e = return_page_query_range(q_b=q_b, q_e=q_e, t_min_pages=t_min_pages,
                                                       t_max_pages=t_max_pages)

        query_time_start = time.time()
        concatenation = Concatenation(coefficients=coefficients, zs=z_pages[pn_b:pn_e + 1],
                                      xs=x_pages[pn_b:pn_e + 1],
                                      t_maxs=t_max_pages[pn_b:pn_e + 1], t_mins=t_min_pages[pn_b:pn_e + 1], v=v,
                                      M=M,
                                      isnans=isnan_pages)
        x_concat = concatenation.return_x_concat()
        z_concat = concatenation.return_z_concat()
        t_concat = np.arange(t_min_pages[pn_b], t_max_pages[pn_e] + 1)

        backward = BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                        M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                        epsilon_2=1e-5)

        query_time_end = time.time()
        s_lsm = backward.return_s()
        bias = np.mean(s_lsm)
        s_lsm -= bias
        tau_lsm = backward.return_tau() + bias
        r_lsm = x_concat[5 * M:] - tau_lsm - s_lsm

        tau_rmse = np.sqrt(np.square(tau_lsm - tau_truth).mean())
        s_rmse = np.sqrt(np.square(s_lsm - s_truth).mean())
        r_rmse = np.sqrt(np.square(r_lsm - r_truth).mean())

        print("LSM-STL trend:", tau_rmse, "seasonal:", s_rmse, "residual:", r_rmse, "time:",
              query_time_end - query_time_start)

        # oneshot
        with open("./data/trend" + str(dataset_idx) + ".txt", "r") as f:
            tau_oneshot = [float(i) for i in f.read().strip().split("\n")]
        with open("./data/seasonal" + str(dataset_idx) + ".txt", "r") as f:
            s_oneshot = [float(i) for i in f.read().strip().split("\n")]
        with open("./data/residual" + str(dataset_idx) + ".txt", "r") as f:
            r_oneshot = [float(i) for i in f.read().strip().split("\n")]

        time_oneshot = 0.0

        # query_time_start = time.time()
        # tau_oneshot, s_oneshot, r_oneshot = oneshotstl(y=y_oneshot, T=M)
        # query_time_end = time.time()

        tau_rmse = np.sqrt(np.square(tau_oneshot - tau_truth).mean())
        s_rmse = np.sqrt(np.square(s_oneshot - s_truth).mean())
        r_rmse = np.sqrt(np.square(r_oneshot - r_truth).mean())
        #
        print("OneShotSTL trend:", tau_rmse, "seasonal:", s_rmse, "residual:", r_rmse, "time:",
              time_oneshot)

        query_time_start = time.time()
        y_robust = np.array(x_concat[q_b - t_min_pages[pn_b]:q_e - t_min_pages[pn_b] + 1])
        tau_robust, s_robust, r_robust = robuststl(y=y_robust, T=M, reg1=reg1, reg2=reg2, K=K, H=H, dn1=dn1, dn2=dn2,
                                                   ds1=ds1, ds2=ds2)
        query_time_end = time.time()

        tau_rmse = np.sqrt(np.square(tau_robust - tau_truth).mean())
        s_rmse = np.sqrt(np.square(s_robust - s_truth).mean())
        r_rmse = np.sqrt(np.square(r_robust - r_truth).mean())

        print("RobustSTL trend:", tau_rmse, "seasonal:", s_rmse, "residual:", r_rmse, "time:",
              query_time_end - query_time_start)

        tau_truth, s_truth, r_truth, tau_lsm, s_lsm, r_lsm, tau_oneshot, s_oneshot, r_oneshot, tau_robust, s_robust, r_robust = \
            tau_truth[:60], s_truth[:60], r_truth[:60], tau_lsm[:60], s_lsm[:60], r_lsm[:60], \
            tau_oneshot[:60], s_oneshot[:60], r_oneshot[:60], tau_robust[:60], s_robust[:60], r_robust[:60],

        draw_pic_combined(tau_truth, s_truth, r_truth, tau_lsm, s_lsm, r_lsm, tau_oneshot, s_oneshot, r_oneshot,
                          tau_robust,
                          s_robust, r_robust, save=save)


def main_threshold_ldlt():
    print("main_threshold_ldlt()")

    # 1. data acquisition
    for dataset_idx in [1, 2]:
        x_size = 150
        if dataset_idx == 1:
            x, t, M, tau_truth, s_truth, r_truth = generate_syn1(x_size)
        else:
            x, t, M, tau_truth, s_truth, r_truth = generate_syn2(x_size)

    # query range
    size = len(x)
    q_b, q_e = 0, len(x) - 1

    # 3. cold start
    x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
    cs = ColdStart(x=x_cs, M=M)
    v = cs.return_v()

    # write_to_file(dataset_name + " recal 1e-1,1e-2,1e-3,1e-4,1e-5\n")
    space, mse = [], []
    for epsilon_ldlt in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        # 4. ldlt-decomposition
        coefficients = ldlt(N=data_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
        head_impute_size = coefficients.rtn_converge_size() + 2

        space.append((head_impute_size + 2) * 5 * 4)

        print(epsilon_ldlt)
        # 5. forward substitution
        z_pages, x_pages, t_max_pages, t_min_pages, isnan_pages = [], [], [], [], []
        insert_time_sum = 0.0
        for page_idx in range(size // page_size + 1):
            if size % page_size == 0 and page_idx == size // page_size:
                break

            x_page, t_page = reconstruct(x[page_idx * page_size:(page_idx + 1) * page_size],
                                         t[page_idx * page_size:(page_idx + 1) * page_size])
            # calculate
            forward = ForwardSubstitution(coefficients=coefficients, x=x_page, t=t_page, v=v, M=M,
                                          hi_size=head_impute_size)

            # record
            z_pages.append(forward.return_z())
            x_pages.append(forward.return_x_imputed())
            t_max_pages.append(np.max(t_page))
            t_min_pages.append(np.min(t_page))
            isnan_pages.append(forward.return_isnan())

        # write_to_file(str(round(insert_time_sum, 3)) + ",")

        # 6. concatenation
        q_b, q_e, pn_b, pn_e = return_page_query_range(q_b=q_b, q_e=q_e, t_min_pages=t_min_pages,
                                                       t_max_pages=t_max_pages)
        concatenation = Concatenation(coefficients=coefficients, zs=z_pages[pn_b:pn_e + 1],
                                      xs=x_pages[pn_b:pn_e + 1],
                                      t_maxs=t_max_pages[pn_b:pn_e + 1], t_mins=t_min_pages[pn_b:pn_e + 1], v=v,
                                      M=M,
                                      isnans=isnan_pages[pn_b:pn_e + 1], epsilon_2=epsilon_recal)
        x_concat = concatenation.return_x_concat()
        z_concat = concatenation.return_z_concat()
        t_concat = np.arange(t_min_pages[pn_b], t_max_pages[pn_e] + 1)

        # 7. backward substitution
        backward = BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                        M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                        epsilon_2=epsilon_recal)

        tau_rtn = backward.return_tau()
        s_rtn = backward.return_s()
        r_rtn = x_concat - s_rtn - tau_rtn

        tau_mse = np.sqrt(np.square(tau_rtn - tau_truth).mean())
        s_mse = np.sqrt(np.square(s_rtn - s_truth).mean())
        r_mse = np.sqrt(np.square(r_rtn - r_truth).mean())

        mse.append((tau_mse + s_mse + r_mse) / 3)
        # write_to_file(str(round(query_time_end - query_time_start, 3)) + ",")
    print("mse:", "\n", ",".join([str(i) for i in mse]))
    print("space:", "\n", ",".join([str(i) for i in space]))


def main_threshold_recal():
    print("main_threshold_recal()")

    # 1. data acquisition
    for dataset_idx in [1, 2]:
        x_size = 10000
        if dataset_idx == 1:
            x, t, M, tau_truth, s_truth, r_truth = generate_syn1(x_size)
        else:
            x, t, M, tau_truth, s_truth, r_truth = generate_syn2(x_size)

    page_size = 50
    missing_rate = 10.0
    missing_length = 5
    out_of_order_rate = 10.0
    out_of_order_length = 5

    # 2. data noising
    # x, t, size = missing(x, t, missing_rate, missing_length)
    size = len(x)
    x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

    # query range
    q_b, q_e = 0, len(x) - 1

    # 3. cold start
    x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
    cs = ColdStart(x=x_cs, M=M)
    v = cs.return_v()

    # 4. ldlt-decomposition
    coefficients = ldlt(N=data_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
    head_impute_size = coefficients.rtn_converge_size() + 2

    mse, timec = [], []

    # write_to_file(dataset_name + " recal 1e-1,1e-2,1e-3,1e-4,1e-5\n")
    for epsilon_recal in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        print(epsilon_recal)
        # 5. forward substitution
        z_pages, x_pages, t_max_pages, t_min_pages, isnan_pages = [], [], [], [], []
        insert_time_sum = 0.0
        for page_idx in range(size // page_size + 1):
            if size % page_size == 0 and page_idx == size // page_size:
                break

            x_page, t_page = reconstruct(x[page_idx * page_size:(page_idx + 1) * page_size],
                                         t[page_idx * page_size:(page_idx + 1) * page_size])
            # calculate
            forward = ForwardSubstitution(coefficients=coefficients, x=x_page, t=t_page, v=v, M=M,
                                          hi_size=head_impute_size)

            # record
            z_pages.append(forward.return_z())
            x_pages.append(forward.return_x_imputed())
            t_max_pages.append(np.max(t_page))
            t_min_pages.append(np.min(t_page))
            isnan_pages.append(forward.return_isnan())

        # 6. concatenation
        q_b, q_e, pn_b, pn_e = return_page_query_range(q_b=q_b, q_e=q_e, t_min_pages=t_min_pages,
                                                       t_max_pages=t_max_pages)
        query_time_start = time.time()
        concatenation = Concatenation(coefficients=coefficients, zs=z_pages[pn_b:pn_e + 1],
                                      xs=x_pages[pn_b:pn_e + 1],
                                      t_maxs=t_max_pages[pn_b:pn_e + 1], t_mins=t_min_pages[pn_b:pn_e + 1], v=v,
                                      M=M,
                                      isnans=isnan_pages[pn_b:pn_e + 1], epsilon_2=epsilon_recal)
        x_concat = concatenation.return_x_concat()
        z_concat = concatenation.return_z_concat()
        t_concat = np.arange(t_min_pages[pn_b], t_max_pages[pn_e] + 1)

        # 7. backward substitution
        backward = BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                        M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                        epsilon_2=epsilon_recal)
        query_time_end = time.time()

        timec.append(query_time_end - query_time_start)

        tau_rtn = backward.return_tau()
        s_rtn = backward.return_s()
        r_rtn = x_concat - s_rtn - tau_rtn

        tau_mse = np.sqrt(np.square(tau_rtn - tau_truth).mean())
        s_mse = np.sqrt(np.square(s_rtn - s_truth).mean())
        r_mse = np.sqrt(np.square(r_rtn - r_truth).mean())

        mse.append((tau_mse + s_mse + r_mse) / 3)

    print("mse:", "\n", ",".join([str(i) for i in mse]))
    print("time:", "\n", ",".join([str(i) for i in timec]))

    # tau = backward.return_tau()
    # s = backward.return_s()
    # r = x_concat[q_b:q_e + 1] - tau - s
    # draw_pic(trend=tau, seasonal=s, resid=r)


if __name__ == '__main__':
    cold_start_size = 5
    lambda_t = 1.0

    dataset_name = "power"  # voltage
    method_name = "lsm"  # oneshot online

    epsilon_ldlt = 1e-5
    epsilon_recal = 1e-3

    data_size = 200000
    page_size = 30000
    query_size = 10000
    missing_rate = 0.0
    missing_length = 5
    out_of_order_rate = 0.0
    out_of_order_length = 5

    np.random.seed(666)

    y, t, T, trend, seasonal, residual = generate_syn2(100)

    main_syn()
    # main_threshold_ldlt()
    # main_threshold_recal()
