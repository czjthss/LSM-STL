import numpy as np
import time


def onlinestl(s, period):
    s = np.array(s)
    size = len(s) - period
    gamma = 0.3
    trend, seasonal = np.zeros(size), np.zeros(size)
    # initial

    start_time = time.time()
    for i in range(size):
        if i < period:
            trend[i], seasonal[i] = np.nan, np.nan
        else:
            num = 0.0
            rate_sum = 0.0
            for j in range(i - period, i + period):
                rate = (1 - (abs(i - j) / period) ** 3) ** 3
                num += rate * s[j]
                rate_sum += rate

            trend[i] = num / rate_sum
            # trend[i] = tf(s, i, period, i - period, i)

    detrend = s[:size] - trend
    for i in range(period, size):
        if i < 2 * period:
            seasonal[i] = detrend[i]
        else:
            seasonal[i] = gamma * detrend[i] + (1 - gamma) * seasonal[i - period]

    residual = detrend - seasonal
    end_time = time.time()

    return end_time - start_time
