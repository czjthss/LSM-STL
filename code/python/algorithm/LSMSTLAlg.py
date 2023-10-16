import numpy as np
from statsmodels.tsa.seasonal import STL
from observe import batch_pre


class ldlt:
    def __init__(self, N, epsilon_1=1e-5, lambda_t=1.0, lambda_s=1.0):
        self.__N = N
        self.__epsilon_1 = epsilon_1

        self.__construct_a(lambda_t, lambda_s)
        self.__decompose_a()

    def __construct_a(self, lambda_t, lambda_s):
        start_lines1 = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])
        start_lines2 = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]) * lambda_s
        start_lines3 = np.array([[1, 0, -2, 0, 1], [0, 0, 0, 0, 0], [-2, 0, 5, 0, -4], [0, 0, 0, 0, 0]]) * lambda_t
        start_lines = start_lines1 + start_lines2 + start_lines3

        mid_lines1 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]])
        mid_lines2 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]) * lambda_s
        mid_lines3 = np.array([[1, 0, -4, 0, 6], [0, 0, 0, 0, 0]]) * lambda_t
        mid_lines = mid_lines1 + mid_lines2 + mid_lines3

        end_lines1 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1]])
        end_lines2 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]) * lambda_s
        end_lines3 = np.array([[1, 0, -4, 0, 5], [0, 0, 0, 0, 0], [1, 0, -2, 0, 1], [0, 0, 0, 0, 0]]) * lambda_t
        end_lines = end_lines1 + end_lines2 + end_lines3

        self.__a_start_lines = start_lines
        self.__a_mid_lines = mid_lines
        self.__a_end_lines = end_lines

    def __query_a(self, i, j):
        if i < j:
            temp = i
            i = j
            j = temp

        if abs(i - j) > 4:
            return 0.0
        if i < 4:
            return self.__a_start_lines[i][j]
        elif 4 <= i < 2 * self.__N - 4:
            return self.__a_mid_lines[i % 2][j - i + 4]
        else:
            return self.__a_end_lines[i - 2 * self.__N + 4][j - i + 4]

    def __decompose_a(self):
        L = np.zeros((2 * self.__N, 4))
        D = np.zeros(2 * self.__N)
        C = np.zeros((2, 5))
        converge = False

        for i in range(2 * self.__N):
            if converge and i < 2 * self.__N - 4:
                D[i] = C[i % 2][4]
                for j in range(i + 1, i + 5):
                    L[j][4 - j + i] = C[i % 2][j - i - 1]
            else:
                D[i] = self.__query_a(i, i) - np.sum(
                    [D[k] * L[i][k + 4 - i] * L[i][k + 4 - i] for k in range(max(0, i - 4), i)])
                for j in range(i + 1, min(2 * self.__N, i + 5)):
                    L[j][4 - j + i] = (self.__query_a(j, i) - np.sum(
                        [L[j][k + 4 - j] * D[k] * L[i][k + 4 - i] for k in range(max(0, j - 4), i)])) / D[i]
                if i >= 4 and np.sum([
                    np.abs(D[h] - D[h - 2]) +
                    np.sum([np.abs(L[h][k] - L[h - 2][k]) for k in range(4)]) for h in [i - 1, i]]) < self.__epsilon_1:
                    converge = True
                    self.__converge_size = (i + 1) // 2
                    for j in range(i + 1, i + 5):
                        C[i % 2][j - i - 1] = L[j][4 - j + i]
                        C[(i - 1) % 2][j - i - 1] = L[j - 1][4 - j + i]
                    C[i % 2][4] = D[i]
                    C[(i - 1) % 2][4] = D[i - 1]

        self.__L = L
        self.__D = D

    def print_l(self):
        for i in range(len(self.__L)):
            for j in range(len(self.__L)):
                print(self.query_l(j, i), end="  ")
            print()
        print()

    def print_d(self):
        for i in range(len(self.__L) - 1, -1, -1):
            print(self.query_d(i, i), end="  ")
        print()

    def query_l(self, i, j):
        if i == j:
            return 1.0
        elif j > i or i - j > 4:
            return 0.0
        else:
            return self.__L[i][j - i + 4]

    def query_d(self, i, j):
        if i != j:
            return 0.0
        else:
            return self.__D[i]

    def query_end_l(self, i, j, N_new):
        # if i < 2 * N_new - 4 or j < 2 * N_new - 4:
        #     raise Exception("Query is exceed the end.")
        if i == j:
            return 1.0
        elif j > i or i - j > 4:
            return 0.0
        else:
            return self.__L[i - 2 * N_new + 2 * self.__N][j - i + 4]

    def query_end_d(self, i, j, N_new):
        # if i < 2 * N_new - 4 or j < 2 * N_new - 4:
        #     raise Exception("Query is exceed the end.")
        if i != j:
            return 0.0
        else:
            return self.__D[i - 2 * N_new + 2 * self.__N]

    def rtn_converge_size(self):
        return self.__converge_size


class ColdStart:
    def __init__(self, x, M):
        self.__x = x
        self.__impute()

        self.__M = M

        # self.__v = self.__stl()
        self.__v = self.__batch()

    def __impute(self):
        # head and tail
        start_i, end_i = 0, len(self.__x) - 1
        while np.isnan(self.__x[start_i]):
            start_i += 1
        for i in range(start_i):
            self.__x[i] = self.__x[start_i]
        while np.isnan(self.__x[end_i]):
            end_i -= 1
        for i in range(end_i + 1, len(self.__x)):
            self.__x[i] = self.__x[end_i]

        # body
        for i in range(start_i, end_i):
            if np.isnan(self.__x[i]):
                left_i = i - 1
                while np.isnan(self.__x[i]):
                    i += 1
                right_i = i
                for j in range(left_i + 1, right_i):
                    self.__x[j] = (self.__x[right_i] - self.__x[left_i]) / (right_i - left_i) * \
                                  (j - left_i) + self.__x[left_i]

    def __batch(self):
        _, seasonal, _ = batch_pre(x=self.__x, T=self.__M)
        bias = np.mean(seasonal)
        seasonal -= bias

        v = []
        for i in range(self.__M):
            v.append(np.average(seasonal[i::self.__M]))
        return v

    def __stl(self):
        stl = STL(self.__x, period=self.__M).fit()
        seasonal = stl.seasonal

        v = []
        for i in range(self.__M):
            v.append(np.average(seasonal[i::self.__M]))
        return v

    def return_v(self):
        return self.__v


class ForwardSubstitution:
    def __init__(self, coefficients, x, t, v, M, hi_size):
        self.__v = v
        self.__M = M
        self.__t = t
        self.__isnan = np.full(len(x), False)
        self.__b = self.__generate_b(self.__head_impute(x, hi_size))
        self.__coefficients = coefficients

        self.__forward_substitution()

    def __generate_b(self, x):
        b = np.repeat(x, repeats=2)
        for idx, ts in enumerate(self.__t):
            b[idx * 2 + 1] += self.__v[ts % self.__M]
        return b

    def __forward_substitution(self):
        self.__z = np.zeros(len(self.__b))
        for i in range(len(self.__b)):
            # impute
            if np.isnan(self.__b[i]):
                self.__body_impute(i)
            # calculate
            res = 0.0
            for k in range(max(i - 4, 0), i):
                res += self.__coefficients.query_l(i, k) * self.__z[k]
            self.__z[i] = self.__b[i] - res

    def __head_impute(self, x, size):
        # head and tail
        size = min(len(x) - 1, size)
        start_i, end_i = 0, size
        while np.isnan(x[start_i]):
            self.__isnan[start_i] = True
            start_i += 1
        for i in range(start_i):
            x[i] = x[start_i]
        while np.isnan(x[end_i]):
            self.__isnan[end_i] = True
            end_i -= 1
        for i in range(end_i + 1, size):
            x[i] = x[end_i]

        # body
        for i in range(start_i, end_i):
            if np.isnan(x[i]):
                left_i = i - 1
                while np.isnan(x[i]):
                    self.__isnan[i] = True
                    i += 1
                right_i = i
                for j in range(left_i + 1, right_i):
                    x[j] = (x[right_i] - x[left_i]) / (right_i - left_i) * \
                           (j - left_i) + x[left_i]

        return x

    def __body_impute(self, i_impute):
        # tail forward
        self.__isnan[i_impute // 2] = True
        z_temp = self.__z.copy()
        for i in range(i_impute - 4, i_impute):
            res = 0.0
            for k in range(max(i - 4, 0), i):
                res += self.__coefficients.query_end_l(i, k, N_new=i_impute // 2) * z_temp[k]
            z_temp[i] = self.__b[i] - res

        # backward
        tau_impute = z_temp[i_impute - 2] / self.__coefficients.query_end_d(i_impute - 2, i_impute - 2,
                                                                            N_new=i_impute // 2) - \
                     z_temp[i_impute - 1] / self.__coefficients.query_end_d(i_impute - 1, i_impute - 1,
                                                                            N_new=i_impute // 2)
        v_impute = self.__v[self.__t[i_impute // 2] % self.__M]  # probably not from the beginning

        # impute
        self.__b[i_impute] = tau_impute + v_impute
        self.__b[i_impute + 1] = tau_impute + 2 * v_impute

    def return_z(self):
        return self.__z

    def return_x_imputed(self):
        return self.__b[::2]

    def return_isnan(self):
        return self.__isnan


class Concatenation:
    def __init__(self, coefficients, zs, xs, t_maxs, t_mins, isnans, v, M, epsilon_2=1e-3):
        self.__coefficients = coefficients
        self.__v = v
        self.__M = M
        self.__epsilon_2 = epsilon_2

        self.__size = len(xs)
        self.__zs = zs
        self.__xs = xs
        self.__t_maxs = t_maxs
        self.__t_mins = t_mins
        self.__isnans = isnans
        self.__z_concat, self.__x_concat, self.__isnan_concat = zs[0], xs[0], isnans[0]

        for i in range(1, self.__size):
            self.__concat(i)

    def __generate_b(self, x, t):
        b = np.repeat(x, repeats=2)
        for idx, ts in enumerate(t):
            b[idx * 2 + 1] += self.__v[ts % self.__M]
        return b

    def __concat(self, i):
        # adjacent
        if self.__t_maxs[i - 1] + 1 == self.__t_mins[i]:
            x_last_point_pre = len(self.__x_concat)
            # 1. x,isnan: direct concat
            self.__x_concat = np.concatenate((self.__x_concat, self.__xs[i]))
            self.__isnan_concat = np.concatenate((self.__isnan_concat, self.__isnans[i]))
            # 2. z: converge concat in adjacent point
            self.__z_concat = np.concatenate((self.__z_concat, self.__zs[i]))
            self.__converge_concat(x_last_point_pre)
        # disjoint
        elif self.__t_maxs[i - 1] + 1 < self.__t_mins[i]:
            x_last_point_pre = len(self.__x_concat)
            # 1. x: concat with imputation
            x_interval_impute = np.zeros(self.__t_mins[i] - self.__t_maxs[i - 1] - 1)
            # 1.1 tail forward
            z_temp = self.__z_concat.copy()
            i_impute = len(z_temp)
            x_impute = len(z_temp) // 2
            b_temp = [self.__x_concat[x_impute - 2],
                      self.__x_concat[x_impute - 2] + self.__v[(self.__t_maxs[i - 1] - 1) % self.__M],
                      self.__x_concat[x_impute - 1],
                      self.__x_concat[x_impute - 1] + self.__v[(self.__t_maxs[i - 1]) % self.__M]]
            for idx in range(i_impute - 4, i_impute):
                res = 0.0
                for k in range(max(idx - 4, 0), idx):
                    res += self.__coefficients.query_end_l(idx, k, N_new=x_impute) * z_temp[k]
                z_temp[idx] = b_temp[idx - i_impute + 4] - res
            # 1.2 backward
            tau_impute = z_temp[i_impute - 2] / self.__coefficients.query_end_d(i_impute - 2, i_impute - 2,
                                                                                N_new=x_impute) - \
                         z_temp[i_impute - 1] / self.__coefficients.query_end_d(i_impute - 1, i_impute - 1,
                                                                                N_new=x_impute)
            # 1.3 impute
            for idx, ts in enumerate(range(self.__t_maxs[i - 1] + 1, self.__t_mins[i])):
                v_impute = self.__v[ts % self.__M]  # probably not from the beginning
                x_interval_impute[idx] = tau_impute + v_impute

            self.__x_concat = np.concatenate((self.__x_concat, x_interval_impute, self.__xs[i]))
            # 2. isnan: missing points are marked as True
            isnan_interval_impute = np.full(self.__t_mins[i] - self.__t_maxs[i - 1] - 1, True)
            self.__isnan_concat = np.concatenate((self.__isnan_concat, isnan_interval_impute, self.__isnans[i]))
            # 3. z: converge concat in end of (i-1)-th page
            z_interval_impute = np.zeros(2 * (self.__t_mins[i] - self.__t_maxs[i - 1] - 1))
            self.__z_concat = np.concatenate((self.__z_concat, z_interval_impute, self.__zs[i]))
            self.__converge_concat(x_last_point_pre)
        # overlapped
        else:
            # 1. reconstruct the concat result
            x_concat_temp = np.zeros(self.__t_maxs[i] - self.__t_mins[0] + 1)
            isnan_concat_temp = np.full(self.__t_maxs[i] - self.__t_mins[0] + 1, False)
            z_concat_temp = np.zeros(2 * (self.__t_maxs[i] - self.__t_mins[0] + 1))
            converge_idxs = []
            for idx, ts in enumerate(range(self.__t_mins[0], self.__t_maxs[i] + 1)):
                if ts < self.__t_mins[i]:  # pre
                    x_concat_temp[idx] = self.__x_concat[ts - self.__t_mins[0]]
                    isnan_concat_temp[idx] = self.__isnan_concat[ts - self.__t_mins[0]]
                    z_concat_temp[2 * idx] = self.__z_concat[2 * (ts - self.__t_mins[0])]
                    z_concat_temp[2 * idx + 1] = self.__z_concat[2 * (ts - self.__t_mins[0]) + 1]
                elif ts > self.__t_maxs[i - 1]:  # now
                    x_concat_temp[idx] = self.__xs[i][ts - self.__t_mins[i]]
                    isnan_concat_temp[idx] = self.__isnans[i][ts - self.__t_mins[i]]
                    z_concat_temp[2 * idx] = self.__zs[i][2 * (ts - self.__t_mins[i])]
                    z_concat_temp[2 * idx + 1] = self.__zs[i][2 * (ts - self.__t_mins[i]) + 1]
                # pre not nan and now nan
                elif self.__isnans[i][ts - self.__t_mins[i]] and not self.__isnan_concat[ts - self.__t_mins[0]]:
                    x_concat_temp[idx] = self.__x_concat[ts - self.__t_mins[0]]
                    isnan_concat_temp[idx] = self.__isnan_concat[ts - self.__t_mins[0]]
                    z_concat_temp[2 * idx] = self.__z_concat[2 * (ts - self.__t_mins[0])]
                    z_concat_temp[2 * idx + 1] = self.__z_concat[2 * (ts - self.__t_mins[0]) + 1]
                else:
                    converge_idxs.append(idx - self.__t_mins[0])
                    x_concat_temp[idx] = self.__xs[i][ts - self.__t_mins[i]]
                    isnan_concat_temp[idx] = self.__isnans[i][ts - self.__t_mins[i]]
                    z_concat_temp[2 * idx] = self.__zs[i][2 * (ts - self.__t_mins[i])]
                    z_concat_temp[2 * idx + 1] = self.__zs[i][2 * (ts - self.__t_mins[i]) + 1]
            self.__x_concat = x_concat_temp
            self.__isnan_concat = isnan_concat_temp
            self.__z_concat = z_concat_temp
            # 2. z: converge concat
            for converge_idx in converge_idxs:
                self.__converge_concat(converge_idx)

    def __converge_concat(self, u_b):
        b_temp = self.__generate_b(self.__x_concat,
                                   np.arange(self.__t_mins[0], self.__t_mins[0] + len(self.__x_concat)))
        z_pre = 0x3f3f3f3f
        for i in range(2 * u_b, len(self.__z_concat)):
            res = 0.0
            for k in range(max(i - 4, 0), i):
                res += self.__coefficients.query_l(i, k) * self.__z_concat[k]
            z_new = b_temp[i] - res
            # converge
            if i > 0 and np.abs(self.__z_concat[i] - z_new) + np.abs(self.__z_concat[i - 1] - z_pre) < self.__epsilon_2:
                # print("concatenation converge:", i - 2 * u_b, "/", len(self.__z_concat) - 2 * u_b)
                break
            self.__z_concat[i], z_pre = z_new, z_new

    def return_x_concat(self):
        return self.__x_concat

    def return_z_concat(self):
        return self.__z_concat


class BackwardSubstitution:
    def __init__(self, coefficients, z, x, t, v, M, q_b=0, q_e=None, epsilon_2=1e-3):
        self.__coefficients = coefficients
        self.__epsilon_2 = epsilon_2

        if q_e is None:
            q_e = len(z) // 2
        self.__z = z[q_b * 2:q_e * 2 + 2]
        self.__v = v
        self.__M = M
        self.__b = self.__generate_b(x[q_b:q_e + 1], t[q_b:q_e + 1])
        self.__N_query = q_e - q_b + 1

        self.__head_recalculation()
        self.__tail_recalculation()
        self.__backward_substitution()

    def __generate_b(self, x, t):
        b = np.repeat(x, repeats=2)
        for idx, ts in enumerate(t):
            b[idx * 2 + 1] += self.__v[ts % self.__M]
        return b

    def __head_recalculation(self):
        z_pre = 0x3f3f3f3f
        for i in range(2 * self.__N_query):
            res = 0.0
            for k in range(max(i - 4, 0), i):
                res += self.__coefficients.query_l(i, k) * self.__z[k]
            z_new = self.__b[i] - res
            # converge
            if i > 0 and np.abs(self.__z[i] - z_new) + np.abs(self.__z[i - 1] - z_pre) < self.__epsilon_2:
                # print("head recalculation converge:", i, "/", 2 * self.__N_query)
                break
            self.__z[i], z_pre = z_new, z_new

    def __tail_recalculation(self):
        for i in range(2 * self.__N_query - 4, 2 * self.__N_query):
            res = 0.0
            for k in range(max(i - 4, 0), i):
                res += self.__coefficients.query_end_l(i, k, N_new=self.__N_query) * self.__z[k]
            self.__z[i] = self.__b[i] - res

    def __backward_substitution(self):
        self.__y = np.zeros(2 * self.__N_query)
        for i in range(2 * self.__N_query - 1, -1, -1):
            res = 0.0
            for k in range(min(i + 4, 2 * self.__N_query - 1), i, -1):
                # last 4 rows
                if i >= 2 * self.__N_query - 4:
                    res += self.__coefficients.query_end_l(k, i, N_new=self.__N_query) * self.__y[k]
                else:
                    res += self.__coefficients.query_l(k, i) * self.__y[k]
            # last 4 rows
            if i >= 2 * self.__N_query - 4:
                self.__y[i] = self.__z[i] / self.__coefficients.query_end_d(i, i, N_new=self.__N_query) - res
            else:
                self.__y[i] = self.__z[i] / self.__coefficients.query_d(i, i) - res

    def return_tau(self):
        return self.__y[::2]

    def return_s(self):
        return self.__y[1::2]

    def return_y(self):
        return self.__y


def query_a(N, i, j, lambda_t=1.0, lambda_s=1.0):
    start_lines1 = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0]])
    start_lines2 = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]) * lambda_s
    start_lines3 = np.array([[1, 0, -2, 0, 1], [0, 0, 0, 0, 0], [-2, 0, 5, 0, -4], [0, 0, 0, 0, 0]]) * lambda_t
    start_lines = start_lines1 + start_lines2 + start_lines3

    mid_lines1 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]])
    mid_lines2 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]) * lambda_s
    mid_lines3 = np.array([[1, 0, -4, 0, 6], [0, 0, 0, 0, 0]]) * lambda_t
    mid_lines = mid_lines1 + mid_lines2 + mid_lines3

    end_lines1 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1]])
    end_lines2 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]) * lambda_s
    end_lines3 = np.array([[1, 0, -4, 0, 5], [0, 0, 0, 0, 0], [1, 0, -2, 0, 1], [0, 0, 0, 0, 0]]) * lambda_t
    end_lines = end_lines1 + end_lines2 + end_lines3

    a_start_lines = start_lines
    a_mid_lines = mid_lines
    a_end_lines = end_lines

    if i < j:
        temp = i
        i = j
        j = temp

    if abs(i - j) > 4:
        return 0.0
    if i < 4:
        return a_start_lines[i][j]
    elif 4 <= i < 2 * N - 4:
        return a_mid_lines[i % 2][j - i + 4]
    else:
        return a_end_lines[i - 2 * N + 4][j - i + 4]


def decompose_a_1(N):
    L = np.zeros((2 * N, 4))
    D = np.zeros(2 * N)
    C = np.zeros((2, 5))
    converge = False

    for i in range(2 * N):
        if converge and i < 2 * N - 4:
            D[i] = C[i % 2][4]
            for j in range(i + 1, i + 5):
                L[j][4 - j + i] = C[i % 2][j - i - 1]
        else:
            D[i] = query_a(N, i, i) - np.sum(
                [D[k] * L[i][k + 4 - i] * L[i][k + 4 - i] for k in range(max(0, i - 4), i)])
            for j in range(i + 1, min(2 * N, i + 5)):
                L[j][4 - j + i] = (query_a(N, j, i) - np.sum(
                    [L[j][k + 4 - j] * D[k] * L[i][k + 4 - i] for k in range(max(0, j - 4), i)])) / D[i]
            if i >= 4 and np.sum([
                np.abs(D[h] - D[h - 2]) +
                np.sum([np.abs(L[h][k] - L[h - 2][k]) for k in range(4)]) for h in [i - 1, i]]) < 1e-5:
                converge = True
                for j in range(i + 1, i + 5):
                    C[i % 2][j - i - 1] = L[j][4 - j + i]
                    C[(i - 1) % 2][j - i - 1] = L[j - 1][4 - j + i]
                C[i % 2][4] = D[i]
                C[(i - 1) % 2][4] = D[i - 1]
    return L, D


def decompose_a_2(N):
    L = np.zeros((2 * N, 4))
    D = np.zeros(2 * N)
    converge = False

    for i in range(2 * N):
        if converge and i < 2 * N - 4:
            D[i] = D[i - 2]
            for j in range(i + 1, i + 5):
                L[j][4 - j + i] = L[j - 2][4 - j + i]
        else:
            D[i] = query_a(N, i, i) - np.sum(
                [D[k] * L[i][k + 4 - i] * L[i][k + 4 - i] for k in range(max(0, i - 4), i)])
            for j in range(i + 1, min(2 * N, i + 5)):
                L[j][4 - j + i] = (query_a(N, j, i) - np.sum(
                    [L[j][k + 4 - j] * D[k] * L[i][k + 4 - i] for k in range(max(0, j - 4), i)])) / D[i]
            if i >= 4 and np.sum([
                np.abs(D[h] - D[h - 2]) +
                np.sum([np.abs(L[h][k] - L[h - 2][k]) for k in range(4)]) for h in [i - 1, i]]) < 1e-5:
                converge = True
    return L, D


if __name__ == '__main__':
    L1, D1 = decompose_a_1(100)
    L2, D2 = decompose_a_2(100)

    s = 0

    for i in range(len(L1)):
        for j in range(len(L1[0])):
            if round(L1[i][j], 5) != round(L2[i][j], 5):
                # print("hei")
                s += 1
    for i in range(len(D1)):
        if round(D1[i], 5) != round(D2[i], 5):
            # print("hei")
            s += 1
    print(s)
    print("ok")
