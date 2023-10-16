import numpy as np


def missing(x, t, rate, length):
    size = len(x)
    i = 0
    while i < len(x):
        if np.random.rand() * 100 < rate:
            d = np.random.randint(1, length * 2)
            for j in range(i, min(i + d, len(x) - 1)):
                x[j] = np.nan
                size -= 1
            i += d
        else:
            i += 1
    return x, t, size


def out_of_order(x, t, size, rate, length):
    x_sl, t_sl = np.full(size, -1.0), np.full(size, -1)
    idx_sl = 0
    for idx in range(len(x)):
        if not np.isnan(x[idx]):  # not missing
            if np.random.rand() * 100 < rate:  # out-of-order
                d = np.random.randint(1, length * 2)
                while idx_sl + d < size and t_sl[idx_sl + d] != -1:
                    d += 1
                if idx_sl + d < size:
                    x_sl[idx_sl + d] = x[idx]
                    t_sl[idx_sl + d] = t[idx]
                else:
                    while t_sl[idx_sl] != -1:  # except out-of-order
                        idx_sl += 1
                    x_sl[idx_sl] = x[idx]
                    t_sl[idx_sl] = t[idx]
                    idx_sl += 1
            else:
                while t_sl[idx_sl] != -1:  # except out-of-order
                    idx_sl += 1
                x_sl[idx_sl] = x[idx]
                t_sl[idx_sl] = t[idx]
                idx_sl += 1
    return x_sl, t_sl


def reconstruct(x, t):
    t_min, t_max = np.min(t), np.max(t)
    x_rc, t_rc = np.full(t_max - t_min + 1, np.nan), np.arange(t_min, t_max + 1)
    for idx, ts in enumerate(t):
        x_rc[ts - t_min] = x[idx]
    return x_rc, t_rc


def return_page_query_range(q_b, q_e, t_min_pages, t_max_pages):
    pn_b, pn_e = 0, 0
    if q_b > q_e:
        raise Exception("q_b should not exceed q_e.")
    # page number range
    if q_b <= t_min_pages[0]:
        q_b = t_min_pages[0]
        pn_b = 0
    else:
        for i in range(len(t_min_pages)):
            if t_min_pages[i] > q_b:
                pn_b = i - 1
                break
    if q_e >= t_max_pages[len(t_max_pages) - 1]:
        q_e = t_max_pages[len(t_max_pages) - 1]
        pn_e = len(t_max_pages) - 1
    else:
        for i in range(len(t_max_pages) - 1, -1, -1):
            if t_max_pages[i] < q_e:
                pn_e = i + 1
                break
    return q_b, q_e, pn_b, pn_e


if __name__ == '__main__':
    np.random.seed(666)
    x = np.arange(100).astype(float)
    t = np.arange(100)
    x, t, size = missing(x, t, 50, 2)
    x, t = out_of_order(x, t, size, 0.0, 5.0)
    cnt = 0
    for i in x:
        print(i)
        if not np.isnan(i):
            cnt += 1
    print(cnt, size)
