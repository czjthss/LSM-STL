import pandas as pd
from statsmodels.datasets import co2

from algorithm.LSMSTLAlg import *
from algorithm.OneShotSTLAlg import *
from algorithm.Preprocessing import *

fontsize = 22


def print_matrix(m, interval=5):
    for idx, items in enumerate(m):
        print(idx, ":", end=" ")
        for item in items:
            num = np.around(item, interval)
            print(num, end=(2 * interval - 1 - len(str(num))) * " ")
        print()


def acquisition(dataset=None, size=-1):
    if dataset == "power":
        if size == -1:
            size = 52416
        data = pd.read_csv("./data/power_5241600.csv", low_memory=True)
        x = data["value"].to_numpy()[:size]
        M = 144
    elif dataset == "voltage":
        if size == -1:
            size = 2282400
        data = pd.read_csv("./data/voltage_22825440.csv", low_memory=True)
        x = data["value"].to_numpy()[:size]
        M = 1440
    else:
        data_co2 = co2.load().data
        data_co2 = data_co2.resample('M').mean().ffill()
        x = data_co2["co2"].to_numpy()
        M = 12
    t = np.arange(len(x))
    return x, t, M


def write_to_file(string, file_idx=1):
    if file_idx == 0 or file_idx == 1:
        f = open("./results/insert.txt", "a")
        f.write(string)
        f.close()
    if file_idx == 0 or file_idx == 2:
        f = open("./results/query.txt", "a")
        f.write(string)
        f.close()


def main_scala():
    print("main_scala()\n")

    # 1. data acquisition
    for dataset_name in ["power", "voltage"]:
        if dataset_name == "power":
            write_to_file(dataset_name + " scala 10k,20k,30k,40k,50k\n", file_idx=0)
            data_range = range(10000, 50001, 10000)
        else:
            write_to_file(dataset_name + " scala 0.4m,0.8m,1.2m,1.6m,2m\n", file_idx=0)
            data_range = range(400000, 2000001, 400000)
        for method_name in ["lsm", "oneshot"]:
            for data_size in data_range:
                print(dataset_name + "  " + method_name + "  " + str(data_size))

                x_raw, t_raw, M = acquisition(dataset=dataset_name, size=data_size)
                # query range
                q_b, q_e = 0, data_size // 10 - 1

                # 2. data noising
                x, t, size = missing(x_raw.copy(), t_raw.copy(), missing_rate, missing_length)
                x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

                # 3. cold start
                x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
                cs = ColdStart(x=x_cs, M=M)
                v = cs.return_v()

                # 4. ldlt-decomposition
                coefficients = ldlt(N=data_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
                head_impute_size = coefficients.rtn_converge_size() + 2

                # 5. forward substitution
                z_pages, x_pages, t_max_pages, t_min_pages, isnan_pages = [], [], [], [], []
                insert_time_sum = 0.0
                for page_idx in range(size // page_size + 1):
                    if size % page_size == 0 and page_idx == size // page_size:
                        break

                    x_page, t_page = reconstruct(x[page_idx * page_size:(page_idx + 1) * page_size],
                                                 t[page_idx * page_size:(page_idx + 1) * page_size])
                    # calculate
                    insert_time_start = time.time()
                    forward = ForwardSubstitution(coefficients=coefficients, x=x_page, t=t_page, v=v, M=M,
                                                  hi_size=head_impute_size)
                    insert_time_end = time.time()

                    if method_name == "oneshot":
                        insert_time_start = time.time()
                        oneshotstl_flush_impute(x=x_page)
                        insert_time_end = time.time()

                    insert_time_sum += insert_time_end - insert_time_start

                    # record
                    z_pages.append(forward.return_z())
                    x_pages.append(forward.return_x_imputed())
                    t_max_pages.append(np.max(t_page))
                    t_min_pages.append(np.min(t_page))
                    isnan_pages.append(forward.return_isnan())

                write_to_file(str(round(insert_time_sum, 3)) + ",", 1)

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
                if method_name == "lsm":
                    BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                         M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                         epsilon_2=epsilon_recal)
                elif method_name == "oneshot":
                    query_time_start = time.time()
                    x_concat = oneshotstl_query_concat(xs=x_pages[pn_b:pn_e + 1], t_maxs=t_max_pages[pn_b:pn_e + 1],
                                                       t_mins=t_min_pages[pn_b:pn_e + 1],
                                                       isnans=isnan_pages[pn_b:pn_e + 1])
                    y_oneshot = np.array(x_concat[q_b - t_min_pages[pn_b]:q_e - t_min_pages[pn_b]])
                    oneshotstl(y=y_oneshot, T=M)
                query_time_end = time.time()

                write_to_file(str(round(query_time_end - query_time_start, 3)) + ",", 2)

            write_to_file("\n", 0)


def main_psize():
    print("main_psize()\n")

    # 1. data acquisition
    for dataset_name in ["power", "voltage"]:
        x_raw, t_raw, M = acquisition(dataset=dataset_name, size=data_size)
        # query range
        q_b, q_e = 0, query_size - 1

        # 2. data noising
        x, t, size = missing(x_raw.copy(), t_raw.copy(), missing_rate, missing_length)
        x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

        # 3. cold start
        x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
        cs = ColdStart(x=x_cs, M=M)
        v = cs.return_v()

        # 4. ldlt-decomposition
        coefficients = ldlt(N=data_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
        head_impute_size = coefficients.rtn_converge_size() + 2

        write_to_file(dataset_name + " psize 10k,30k,50k,70k,90k\n", file_idx=0)
        for method_name in ["lsm", "oneshot"]:
            for page_size in range(10000, 90001, 20000):
                print(dataset_name + "  " + method_name + "  " + str(page_size))
                # 5. forward substitution
                z_pages, x_pages, t_max_pages, t_min_pages, isnan_pages = [], [], [], [], []
                insert_time_sum = 0.0
                for page_idx in range(size // page_size + 1):
                    if size % page_size == 0 and page_idx == size // page_size:
                        break

                    x_page, t_page = reconstruct(x[page_idx * page_size:(page_idx + 1) * page_size],
                                                 t[page_idx * page_size:(page_idx + 1) * page_size])
                    # calculate
                    insert_time_start = time.time()
                    forward = ForwardSubstitution(coefficients=coefficients, x=x_page, t=t_page, v=v, M=M,
                                                  hi_size=head_impute_size)
                    insert_time_end = time.time()

                    if method_name == "oneshot":
                        insert_time_start = time.time()
                        oneshotstl_flush_impute(x=x_page)
                        insert_time_end = time.time()

                    insert_time_sum += insert_time_end - insert_time_start

                    # record
                    z_pages.append(forward.return_z())
                    x_pages.append(forward.return_x_imputed())
                    t_max_pages.append(np.max(t_page))
                    t_min_pages.append(np.min(t_page))
                    isnan_pages.append(forward.return_isnan())

                write_to_file(str(round(insert_time_sum, 3)) + ",", 1)

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
                if method_name == "lsm":
                    BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                         M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                         epsilon_2=epsilon_recal)
                elif method_name == "oneshot":
                    query_time_start = time.time()
                    x_concat = oneshotstl_query_concat(xs=x_pages[pn_b:pn_e + 1], t_maxs=t_max_pages[pn_b:pn_e + 1],
                                                       t_mins=t_min_pages[pn_b:pn_e + 1],
                                                       isnans=isnan_pages[pn_b:pn_e + 1])
                    y_oneshot = np.array(x_concat[q_b - t_min_pages[pn_b]:q_e - t_min_pages[pn_b]])
                    oneshotstl(y=y_oneshot, T=M)
                query_time_end = time.time()

                write_to_file(str(round(query_time_end - query_time_start, 3)) + ",", 2)

            write_to_file("\n", 0)


def main_qsize():
    print("main_qsize()\n")

    # 1. data acquisition
    for dataset_name in ["power", "voltage"]:
        x_raw, t_raw, M = acquisition(dataset=dataset_name, size=data_size)

        # 2. data noising
        x, t, size = missing(x_raw.copy(), t_raw.copy(), missing_rate, missing_length)
        x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

        # 3. cold start
        x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
        cs = ColdStart(x=x_cs, M=M)
        v = cs.return_v()

        # 4. ldlt-decomposition
        coefficients = ldlt(N=data_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
        head_impute_size = coefficients.rtn_converge_size() + 2

        write_to_file(dataset_name + " qsize 8k,16k,24k,32k,40k\n", file_idx=2)
        for method_name in ["lsm", "oneshot"]:
            for query_size in range(8000, 40001, 8000):
                # query range
                q_b, q_e = 0, query_size - 1

                print(dataset_name + "  " + method_name + "  " + str(query_size))
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
                                              isnans=isnan_pages[pn_b:pn_e + 1])
                x_concat = concatenation.return_x_concat()
                z_concat = concatenation.return_z_concat()
                t_concat = np.arange(t_min_pages[pn_b], t_max_pages[pn_e] + 1)

                # 7. backward substitution
                if method_name == "lsm":
                    backward = BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                                    M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                                    epsilon_2=1e-3)
                elif method_name == "oneshot":
                    query_time_start = time.time()
                    y_oneshot = np.array(x_concat[q_b - t_min_pages[pn_b]:q_e - t_min_pages[pn_b]])
                    oneshotstl(y=y_oneshot, T=M)
                query_time_end = time.time()

                write_to_file(str(round(query_time_end - query_time_start, 3)) + ",", 2)
            # print("query time:", query_time_end - query_time_start)
            write_to_file("\n", 2)


def main_missing_rate():
    print("main_missing_rate()\n")

    # 1. data acquisition
    for dataset_name in ["power", "voltage"]:
        x_raw, t_raw, M = acquisition(dataset=dataset_name, size=data_size)

        # 4. ldlt-decomposition
        coefficients = ldlt(N=data_size + 4, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
        head_impute_size = coefficients.rtn_converge_size() + 2

        # query range
        q_b, q_e = 0, query_size - 1

        write_to_file(dataset_name + " mrate 1,2,3,4,5\n", file_idx=1)
        for method_name in ["lsm", "oneshot"]:
            for missing_rate in range(1, 6, 1):
                missing_rate = float(missing_rate)

                # 2. data noising
                x, t, size = missing(x_raw.copy(), t_raw.copy(), missing_rate, missing_length)

                x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

                # 3. cold start
                x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
                cs = ColdStart(x=x_cs, M=M)
                v = cs.return_v()

                print(dataset_name + "  " + method_name + "  " + str(missing_rate))
                # 5. forward substitution
                insert_time_sum = 0.0
                for page_idx in range(size // page_size + 1):
                    if size % page_size == 0 and page_idx == size // page_size:
                        break

                    x_page, t_page = reconstruct(x[page_idx * page_size:(page_idx + 1) * page_size],
                                                 t[page_idx * page_size:(page_idx + 1) * page_size])
                    # calculate
                    insert_time_start = time.time()
                    if method_name == "lsm":
                        ForwardSubstitution(coefficients=coefficients, x=x_page, t=t_page, v=v, M=M,
                                            hi_size=head_impute_size)
                    else:
                        oneshotstl_flush_impute(x=x_page)
                    insert_time_end = time.time()

                    insert_time_sum += insert_time_end - insert_time_start

                write_to_file(str(round(insert_time_sum, 3)) + ",", 1)

            write_to_file("\n", 1)


def main_missing_length():
    print("main_missing_length()\n")
    missing_rate = 0.5

    # 1. data acquisition
    for dataset_name in ["power", "voltage"]:
        write_to_file(dataset_name + " mlength 20,40,60,80,100\n", file_idx=2)

        x_raw, t_raw, M = acquisition(dataset=dataset_name, size=data_size)

        # 4. ldlt-decomposition
        coefficients = ldlt(N=data_size + 4, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
        head_impute_size = coefficients.rtn_converge_size() + 2

        # query range
        q_b, q_e = 0, query_size - 1

        for method_name in ["lsm", "oneshot"]:
            for missing_length in range(20, 101, 20):
                print(dataset_name + "  " + method_name + "  " + str(missing_length))

                # 2. data noising
                x, t, size = missing(x_raw.copy(), t_raw.copy(), missing_rate, 1)
                x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

                # 3. cold start
                x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
                cs = ColdStart(x=x_cs, M=M)
                v = cs.return_v()

                # 5. forward substitution
                z_pages, x_pages, t_max_pages, t_min_pages, isnan_pages = [], [], [], [], []
                for page_idx in range(size // (page_size + missing_length) + 1):
                    # for page_idx in range(size // page_size + 1):
                    if size % (page_size + missing_length) == 0 and page_idx == size // (page_size + missing_length):
                        # if size % page_size == 0 and page_idx == size // page_size:
                        break

                    x_page, t_page = reconstruct(x[page_idx * (page_size + missing_length):(page_idx + 1) * (page_size + missing_length)],
                                                 t[page_idx * (page_size + missing_length):(page_idx + 1) * (page_size + missing_length)])
                    # calculate
                    forward = ForwardSubstitution(coefficients=coefficients, x=x_page, t=t_page, v=v, M=M,
                                                  hi_size=head_impute_size)

                    if method_name == "oneshot":
                        oneshotstl_flush_impute(x=x_page)

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
                if method_name == "lsm":
                    BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                         M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                         epsilon_2=epsilon_recal)
                elif method_name == "oneshot":
                    query_time_start = time.time()
                    x_concat = oneshotstl_query_concat(xs=x_pages[pn_b:pn_e + 1], t_maxs=t_max_pages[pn_b:pn_e + 1],
                                                       t_mins=t_min_pages[pn_b:pn_e + 1],
                                                       isnans=isnan_pages[pn_b:pn_e + 1])
                    y_oneshot = np.array(x_concat[q_b - t_min_pages[pn_b]:q_e - t_min_pages[pn_b]])
                    oneshotstl(y=y_oneshot, T=M)
                query_time_end = time.time()

                write_to_file(str(round(query_time_end - query_time_start, 3)) + ",", 2)

            write_to_file("\n", 2)


def main_out_of_order_rate():
    print("main_out_of_order_rate()\n")

    # 1. data acquisition
    for dataset_name in ["power", "voltage"]:
        write_to_file(dataset_name + " orate 2,4,6,8,10\n", file_idx=2)

        x_raw, t_raw, M = acquisition(dataset=dataset_name, size=data_size)
        # query range
        q_b, q_e = 0, query_size - 1

        for method_name in ["lsm", "oneshot"]:
            for out_of_order_rate in range(2, 11, 2):
                print(dataset_name + "  " + method_name + "  " + str(out_of_order_rate))

                out_of_order_rate = float(out_of_order_rate)
                # 2. data noising
                x, t, size = missing(x_raw.copy(), t_raw.copy(), missing_rate, missing_length)
                x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

                # 3. cold start
                x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
                cs = ColdStart(x=x_cs, M=M)
                v = cs.return_v()

                # 4. ldlt-decomposition
                coefficients = ldlt(N=data_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
                head_impute_size = coefficients.rtn_converge_size() + 2

                # 5. forward substitution
                z_pages, x_pages, t_max_pages, t_min_pages, isnan_pages = [], [], [], [], []
                insert_time_sum = 0.0
                for page_idx in range(size // page_size + 1):
                    if size % page_size == 0 and page_idx == size // page_size:
                        break
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
                                              isnans=isnan_pages[pn_b:pn_e + 1], epsilon_2=epsilon_recal)
                x_concat = concatenation.return_x_concat()
                z_concat = concatenation.return_z_concat()
                t_concat = np.arange(t_min_pages[pn_b], t_max_pages[pn_e] + 1)

                # 7. backward substitution
                if method_name == "lsm":
                    backward = BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                                    M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                                    epsilon_2=epsilon_recal)
                elif method_name == "oneshot":
                    query_time_start = time.time()
                    y_oneshot = np.array(x_concat[q_b - t_min_pages[pn_b]:q_e - t_min_pages[pn_b]])
                    oneshotstl(y=y_oneshot, T=M)
                query_time_end = time.time()

                write_to_file(str(round(query_time_end - query_time_start, 3)) + ",", 2)

            write_to_file("\n", 2)


def main_out_of_order_length():
    print("main_out_of_order_length()\n")
    out_of_order_rate = 2.0

    # 1. data acquisition
    for dataset_name in ["power", "voltage"]:
        write_to_file(dataset_name + " olength 0.1k,0.2k,0.3k,0.4k,0.5k\n", file_idx=2)

        x_raw, t_raw, M = acquisition(dataset=dataset_name, size=data_size)
        # query range
        q_b, q_e = 0, query_size - 1

        for method_name in ["lsm", "oneshot"]:
            for out_of_order_length in range(100, 501, 100):
                print(dataset_name + "  " + method_name + "  " + str(out_of_order_length))

                # 2. data noising
                x, t, size = missing(x_raw.copy(), t_raw.copy(), missing_rate, missing_length)
                x, t = out_of_order(x, t, size, out_of_order_rate, out_of_order_length)

                # 3. cold start
                x_cs, t_cs = reconstruct(x[:cold_start_size * M], t[:cold_start_size * M])
                cs = ColdStart(x=x_cs, M=M)
                v = cs.return_v()

                # 4. ldlt-decomposition
                coefficients = ldlt(N=data_size, epsilon_1=epsilon_ldlt, lambda_t=lambda_t)
                head_impute_size = coefficients.rtn_converge_size() + 2

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
                if method_name == "lsm":
                    backward = BackwardSubstitution(coefficients=coefficients, z=z_concat, x=x_concat, t=t_concat, v=v,
                                                    M=M, q_b=q_b - t_min_pages[pn_b], q_e=q_e - t_min_pages[pn_b],
                                                    epsilon_2=epsilon_recal)
                elif method_name == "oneshot":
                    y_oneshot = np.array(x_concat[q_b - t_min_pages[pn_b]:q_e - t_min_pages[pn_b]])
                    oneshotstl(y=y_oneshot, T=M)
                query_time_end = time.time()

                write_to_file(str(round(query_time_end - query_time_start, 3)) + ",", 2)

            write_to_file("\n", 2)


if __name__ == '__main__':
    cold_start_size = 5
    lambda_t = 1.0

    dataset_name = "power"  # voltage
    method_name = "lsm"  # oneshot online

    epsilon_ldlt = 1e-5
    epsilon_recal = 1e-3

    data_size = 200000
    page_size = 30000
    query_size = 70000
    missing_rate = 0.5
    missing_length = 1
    out_of_order_rate = 0.5
    out_of_order_length = 100

    np.random.seed(666)

    # main_scala()
    # main_psize()
    # main_qsize()
    main_missing_rate()
    # main_missing_length()
    # main_out_of_order_rate()
    # main_out_of_order_length()
    write_to_file("\n", 0)
