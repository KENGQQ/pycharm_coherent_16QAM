
# %%import
# import sys
import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
from datetime import timedelta

start_time = time.time()

# %%-------------loading/transform data by path + filename----------------#

# tx_xi=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\TxXI0km.txt',skip_header=8)[0:,1] #convert raw data to numerical data
# tx_xq=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\TxXQ0km.txt',skip_header=8)[0:,1]
# tx_yi=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\TxYI0km.txt',skip_header=8)[0:,1] #convert raw data to numerical data
# tx_yq=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\TxYQ0km.txt',skip_header=8)[0:,1]
# rx_xi=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\RxXI0km.txt',skip_header=8)[0:,1]
# rx_xq=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\RxXQ0km.txt',skip_header=8)[0:,1]
# rx_yi=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\RxYI0km.txt',skip_header=8)[0:,1]
# rx_yq=np.genfromtxt(r'D:\新增資料夾\QPSK\no synch\RxYQ0km.txt',skip_header=8)[0:,1]
# tx_xi = np.genfromtxt(r'C:\Users\user\Desktop\QAMtest\80\tx_xi.txt', skip_header=8)[0:, 1]
# tx_xq = np.genfromtxt(r'C:\Users\user\Desktop\QAMtest\80\tx_xq.txt', skip_header=8)[0:, 1]
# rx_xi = np.genfromtxt(r'C:\Users\user\Desktop\QAMtest\80\rx_xi.txt', skip_header=8)[0:, 1]
# rx_xq = np.genfromtxt(r'C:\Users\user\Desktop\QAMtest\80\rx_xq.txt', skip_header=8)[0:, 1]


# %%-----------------subfunction------------------#

def sum(memory_length):
    s = 0
    for n in range(0, memory_length + 1):
        s = s + n
    return s


def rx_datanromalize(origin_data, PAM_order):
    amplitude_factor = (max(origin_data) - min(origin_data)) / 2
    mean_factor = np.mean(origin_data)
    shift_data = origin_data - mean_factor
    normalize_data = shift_data / amplitude_factor
    return normalize_data


def datanromalize(origin_data, PAM_order):
    mean_factor = np.mean(origin_data)
    shift_data = origin_data - mean_factor
    amplitude_factor = np.mean(np.abs(shift_data)) * 2 / PAM_order;
    normalize_data = shift_data / amplitude_factor
    return normalize_data  # ,mean_factor,amplitude_factor


def Sym2Bit(Tx_signal_normalized, PAM_order):  # amplitude level classification
    Tx_signal_symbol = np.zeros(len(Tx_signal_normalized))
    level_value = np.linspace(-PAM_order + 1, PAM_order - 1, PAM_order, endpoint=True)
    level_classification = np.zeros(PAM_order + 1)
    for i in range(PAM_order + 1):
        if i == 0:
            level_classification[i] = float('-Inf')
        elif i == PAM_order:
            level_classification[i] = float('Inf')
        else:
            level_classification[i] = -PAM_order + 2 * i
    for i in range(len(Tx_signal_normalized)):
        for k in range(PAM_order):
            if level_classification[k] <= Tx_signal_normalized[i] and Tx_signal_normalized[i] < level_classification[
                k + 1]:
                Tx_signal_symbol[i] = level_value[k]
            else:
                continue
    return Tx_signal_symbol


def dn_sample(x, dn_num):  # down sampling needed input data in x and down sample number in dn_num
    raw_data = x
    N1 = len(raw_data)
    if N1 % dn_num == 0:
        N2 = N1 // dn_num + 1
        dn_data = np.zeros(N2 - 1)
        yy = np.arange(0, N2 - 1, 1)
    else:
        N2 = N1 // dn_num
        dn_data = np.zeros(N2 + 1)
        yy = np.arange(0, N2 + 1, 1)
    for i in yy:
        if i == 0:
            dn_data[0] = raw_data[0]
        else:
            dn_data[i] = raw_data[i * dn_num]
    return dn_data


def Tx_constellation(i_signal, q_signal):  # constellatiion classification & gray mapping
    I = i_signal
    Q = q_signal
    if len(I) < len(Q):
        N = len(I)
    else:
        N = len(Q)
    result_signal = np.zeros(N, dtype=complex)
    for i in range(N):
        p = I[i]
        imp = Q[i]
        result_signal[i] = complex(p, imp) * (-1)  # *-1 is "gray code convert"
    return result_signal


def data_synch(tx_signal, rx_signal, prbs, dtype=complex):
    tx = tx_signal
    rx = rx_signal
    start = 1  # 2
    window_length = 7000
    end = start + window_length - 1
    tx_window = tx[start - 1:start - 1 + window_length]  # tx[start-1:end]
    scan = 2 ** prbs - 1  # 8192
    if len(rx_signal) > scan:
        scan = len(rx_signal) - window_length
    y = np.zeros((scan, 2), dtype=complex)
    for i in np.arange(0, scan, 1):  # np.linspace(0,len(rx)-window_length,len(rx)-window_length,dtype=int):
        r_positive = np.corrcoef(rx[i:i + window_length], tx_window)[0, 1]
        r_negative = np.corrcoef(-1 * rx[i:i + window_length], tx_window)[0, 1]
        y[i, 0] = r_positive
        y[i, 1] = r_negative
    if max(y[:, 0]) > max(y[:, 1]):
        max_corr = max(y[:, 0])
        shift = np.argmax(y[:, 0])
        rx_1 = rx[shift::]
        print('calculate data: rx')
    else:
        max_corr = max(y[:, 1])
        shift = np.argmax(y[:, 1])
        rx_1 = -1 * rx[shift::]
        print('calculate data:-rx')
    print(shift)
    tx_1 = tx[start - 1:len(rx_1)]
    return max_corr, shift, tx_1, rx_1, y


def corr_synch(tx_signal, rx_signal, prbs, dtype=complex):
    tx = tx_signal
    rx = rx_signal
    start = 2
    window_length = 150
    end = start + window_length - 1
    tx_window = tx[start - 1:start - 1 + window_length]  # tx[start-1:end]
    scan = 2 ** prbs  # 8192
    y = np.zeros((scan, 2), dtype=complex)
    for i in np.arange(0, scan, 1):  # np.linspace(0,len(rx)-window_length,len(rx)-window_length,dtype=int):
        r_positive = np.corrcoef(rx[i:i + window_length], tx_window)[0, 1]
        r_negative = np.corrcoef(-1 * rx[i:i + window_length], tx_window)[0, 1]
        y[i, 0] = r_positive
        y[i, 1] = r_negative
    if max(y[:, 0]) > max(y[:, 1]):
        max_corr = max(y[:, 0])
        shift = np.argmax(y[:, 0])
        rx_1 = rx[shift::]
        print('calculate data: rx')
    else:
        max_corr = max(y[:, 1])
        shift = np.argmax(y[:, 1])
        rx_1 = -1 * rx[shift::]
        print('calculate data:-rx')
    print(shift)
    tx_1 = tx[start - 1:len(rx_1)]
    return max_corr, shift, tx_1, rx_1, y


def total_tap_number(D=16, L=80, lemda=1.55e-6, Baudrate=25e9):  # least tap number requirement in time domain
    Ts = 1 / Baudrate
    T = Ts / 2
    c = 2.997e8
    N = 2 * (math.ceil((D * L * 1e-3 * lemda * lemda / (2 * c * T * T))) - 1) + 1  # tap number
    print('D =', D, 'ps/nm*km,', 'distance =', L, 'km,', 'wavelength =', lemda * 1e9, 'nm,', 'Baudrate =',
          Baudrate / 1e9, 'GBaud')
    print('total tap number requirement: ', N, 'taps')
    return N


def vol_(train_length, memory_length, tx, rx, order):
    train_rx = rx[0:train_length]
    test_rx = rx[train_length::]
    train_tx = tx[0:train_length]
    test_tx = tx[int(train_length)::]
    test_length = len(tx) - train_length
    tap_number = int((memory_length - 1) / 2)
    feature_1st = memory_length
    feature_2nd = int((memory_length + 1) * memory_length / 2)
    # ----------test sequency----------
    train_zero = np.hstack((np.zeros(tap_number), test_rx, np.zeros(tap_number)))
    vol_bias = np.ones([test_length - 1, 1])
    vol_1st = np.zeros([test_length - 1, feature_1st])
    # vol_2nd=np.zeros([test_length-1,feature_2nd])
    if order >= 1:
        for i in range(test_length - 1):
            vol_1st[i, :] = train_zero[i:i + feature_1st]
    # if order >= 2:
    #   for k in range(test_length-1):
    #      for i in range(memory_length):
    #         for j in range(memory_length):
    #            if i <= j:
    #               vol_2nd[k,j+sum(memory_length-1)-sum(memory_length-1-i)]=train_zero[i+k]*train_zero[j+k]
    if order == 2:
        F_test = np.hstack((vol_bias, vol_1st))
    else:
        F_test = np.hstack((vol_bias, vol_1st))
    # ----------train sequency----------
    train_zero = np.hstack((np.zeros(tap_number), train_rx, np.zeros(tap_number)))
    vol_bias = np.ones([train_length, 1])
    vol_1st = np.zeros([train_length, feature_1st])
    # vol_2nd=np.zeros([train_length,feature_2nd])
    if order >= 1:
        for i in range(train_length):
            vol_1st[i, :] = train_zero[i:i + feature_1st]
    # if order >= 2:
    #   for k in range(train_length-1):
    #      for i in range(memory_length):
    #         for j in range(memory_length):
    #            if i <= j:
    #               vol_2nd[k,j+sum(memory_length-1)-sum(memory_length-1-i)]=train_zero[i+k]*train_zero[j+k]
    if order == 2:
        F = np.hstack((vol_bias, vol_1st))
    else:
        F = np.hstack((vol_bias, vol_1st))
    F_T = F.T
    F_T_inv = np.linalg.inv(F_T.dot(F))
    F_T_t = F_T.dot(train_tx)
    wiener = F_T_inv.dot(F_T_t)
    # wiener=np.linalg.inv((F.T).dot(F)).dot((F.T).dot(train_tx))
    rx_vol = F_test.dot(wiener)
    return train_zero, F, wiener, F_test, rx_vol, test_tx


def vol_c(train_length, memory_length, tx, rx, order, title='zero-forcing(Volterra) MMSE equalizer', dtype=complex):
    # memory_length=39
    train_rx = rx[0:int(train_length)]
    test_rx = rx[int(train_length)::]
    train_tx = tx[0:int(train_length)]
    test_tx = tx[int(train_length)::]
    test_length = len(test_rx)  # -int(train_length)
    tap_number = int((memory_length - 1) / 2)
    feature_1st = int(memory_length)
    feature_2nd = int((memory_length + 1) * memory_length / 2)
    memory_length_u = 7
    tap_u = int((memory_length_u - 1) / 2)
    feature_3rd = int(memory_length_u ** 2)
    # ----------test sequency----------
    train_zero = np.hstack((np.zeros(tap_number, dtype=complex), test_rx, np.zeros(tap_number, dtype=complex)))
    train_zero_u = np.hstack((np.zeros(tap_u, dtype=complex), test_rx, np.zeros(tap_u, dtype=complex)))
    vol_bias = np.ones([test_length, 1], dtype=complex)
    vol_1st = np.zeros([test_length, feature_1st], dtype=complex)
    # vol_2nd=np.zeros([test_length,feature_2nd],dtype=complex)
    # vol_3rd=np.zeros([test_length,feature_3rd],dtype=complex)
    # F3=np.zeros([memory_length_u,memory_length_u],dtype=complex)
    if order >= 1:
        for i in range(test_length):
            vol_1st[i, :] = train_zero[i:i + feature_1st]
    # if order >= 2:
    #   for k in range(test_length-1):
    #      for i in range(int(memory_length)):
    #         for j in range(int(memory_length)):
    #            if i <= j:
    #               vol_2nd[k,j+sum(int(memory_length)-1)-sum(int(memory_length)-1-i)]=train_zero[i+k]*train_zero[j+k]
    # if order >= 3:
    #   for i in range(test_length-1):
    #      temp=train_zero_u[i:i+memory_length_u]
    #     for j in range(memory_length_u):
    #        for k in range(memory_length_u):
    #           F3[j,k]=temp[j]*temp[j]*np.conj(temp[k])
    #  vol_3rd[i,:]=F3.reshape(1,-1)

    if order == 3:
        F_test = np.hstack((vol_1st))
    else:
        F_test = np.hstack((vol_bias, vol_1st))
    # ----------train sequency----------
    train_zero = np.hstack((np.zeros(tap_number, dtype=complex), train_rx, np.zeros(tap_number, dtype=complex)))
    train_zero_u = np.hstack((np.zeros(tap_u, dtype=complex), train_rx, np.zeros(tap_u, dtype=complex)))
    vol_bias = np.ones([train_length, 1], dtype=complex)
    vol_1st = np.zeros([train_length, feature_1st], dtype=complex)
    # vol_2nd=np.zeros([train_length,feature_2nd],dtype=complex)
    # vol_3rd=np.zeros([train_length,feature_3rd],dtype=complex)
    if order >= 1:
        for i in range(train_length):
            vol_1st[i, :] = train_zero[i:i + feature_1st]
    # if order >= 2:
    #   for k in range(train_length-1):
    #      for i in range(int(memory_length)):
    #         for j in range(int(memory_length)):
    #            if i <= j:
    #               vol_2nd[k,j+sum(int(memory_length)-1)-sum(int(memory_length)-1-i)]=train_zero[i+k]*train_zero[j+k]
    # if order >= 3:
    #   for i in range(train_length-1):
    #      temp=train_zero_u[i:i+memory_length_u]
    #     for j in range(memory_length_u):
    #        for k in range(memory_length_u):
    #           F3[j,k]=temp[j]*temp[j]*np.conj(temp[k])
    #  vol_3rd[i,:]=F3.reshape(1,-1)

    if order == 3:
        F = np.hstack((vol_1st))
    else:
        F = np.hstack((vol_bias, vol_1st))
    F_H = np.conj(F.T)
    F_H_inv = np.linalg.inv(F_H.dot(F))
    F_H_t = F_H.dot(train_tx)
    wiener = F_H_inv.dot(F_H_t)
    # wiener=np.linalg.inv((F.T).dot(F)).dot((F.T).dot(train_tx))
    rx_vol = F_test.dot(wiener)
    constellation_plot(rx_vol, title=title, bins=75)
    return train_zero, F, wiener, F_test, rx_vol, test_tx


def vol_pre(train_length, memory_length, tx, rx, order, dtype=complex):
    # memory_length=39
    train_rx = rx[0:int(train_length)]
    test_rx = rx[int(train_length)::]
    train_tx = tx[0:int(train_length)]
    test_tx = tx[int(train_length)::]
    test_length = len(test_rx)
    tap_number = int(memory_length - 1)
    feature_1st = int(memory_length)
    # ----------test sequency----------
    train_zero = np.hstack((np.zeros(tap_number, dtype=complex), test_rx))
    vol_bias = np.ones([test_length, 1], dtype=complex)
    vol_1st = np.zeros([test_length, feature_1st], dtype=complex)
    for i in range(test_length):
        for j in range(feature_1st):
            vol_1st[i, j] = train_zero[feature_1st - 1 + i - j]
    F_test = np.hstack((np.conj(vol_1st), vol_bias, vol_1st))
    # ----------train sequency----------
    train_zero = np.hstack((np.zeros(tap_number, dtype=complex), train_rx))
    vol_bias = np.ones([train_length, 1], dtype=complex)
    vol_1st = np.zeros([train_length, feature_1st], dtype=complex)
    for i in range(train_length):
        for j in range(feature_1st):
            vol_1st[i, j] = train_zero[feature_1st - 1 + i - j]
    F = np.hstack((np.conj(vol_1st), vol_bias, vol_1st))
    F_H = np.conj(F.T)
    F_H_inv = np.linalg.inv(F_H.dot(F))
    F_H_t = F_H.dot(train_tx)
    wiener = F_H_inv.dot(F_H_t)
    # wiener=np.linalg.inv((F.T).dot(F)).dot((F.T).dot(train_tx))
    rx_vol = F_test.dot(wiener)
    return train_zero, F, wiener, F_test, rx_vol, test_tx


def vol_non0(train_length, memory_length, tx, rx, order, dtype=complex):
    # memory_length=39
    train_rx = rx[0:int(train_length)]
    test_rx = rx[int(train_length)::]
    test_tx = tx[int(train_length)::]
    test_length = len(test_rx)
    tap_number = int((memory_length - 1) / 2)
    feature_1st = int(memory_length)
    feature_2nd = int((memory_length + 1) * memory_length / 2)
    memory_length_u = 7
    tap_u = int((memory_length_u - 1) / 2)
    feature_3rd = int(memory_length_u ** 2)
    train_tx = tx[0:int(train_length) - feature_1st + 1]
    # ----------test sequency----------
    train_zero = test_rx
    train_zero_u = np.hstack((np.zeros(tap_u, dtype=complex), test_rx, np.zeros(tap_u, dtype=complex)))
    vol_bias = np.ones([test_length - feature_1st + 1, 1], dtype=complex)
    vol_1st = np.zeros([test_length - feature_1st + 1, feature_1st], dtype=complex)
    vol_2nd = np.zeros([test_length - feature_1st + 1, feature_2nd], dtype=complex)
    vol_3rd = np.zeros([test_length - feature_1st + 1, feature_3rd], dtype=complex)
    F3 = np.zeros([memory_length_u, memory_length_u], dtype=complex)
    if order >= 1:
        for i in range(test_length - feature_1st + 1):
            # print(i,test_rx[i:i+feature_1st].shape)
            vol_1st[i, :] = test_rx[i:i + feature_1st]
    # if order >= 2:
    #   for k in range(test_length-1):
    #      for i in range(int(memory_length)):
    #         for j in range(int(memory_length)):
    #            if i <= j:
    #               vol_2nd[k,j+sum(int(memory_length)-1)-sum(int(memory_length)-1-i)]=train_zero[i+k]*train_zero[j+k]
    if order >= 3:
        for i in range(test_length - 1):
            temp = train_zero_u[i:i + memory_length_u]
            for j in range(memory_length_u):
                for k in range(memory_length_u):
                    F3[j, k] = temp[j] * temp[j] * np.conj(temp[k])
            vol_3rd[i, :] = F3.reshape(1, -1)

    if order == 3:
        F_test = np.hstack((vol_1st, vol_3rd))
    else:
        F_test = np.hstack((vol_bias, vol_1st))  # np.conj(vol_1st)
    # ----------train sequency----------
    train_zero = train_rx
    train_zero_u = np.hstack((np.zeros(tap_u, dtype=complex), train_rx, np.zeros(tap_u, dtype=complex)))
    vol_bias = np.ones([train_length - feature_1st + 1, 1], dtype=complex)
    vol_1st = np.zeros([train_length - feature_1st + 1, feature_1st], dtype=complex)
    vol_2nd = np.zeros([train_length - feature_1st + 1, feature_2nd], dtype=complex)
    vol_3rd = np.zeros([train_length - feature_1st + 1, feature_3rd], dtype=complex)
    if order >= 1:
        for i in range(train_length - feature_1st + 1):
            vol_1st[i, :] = train_rx[i:i + feature_1st]
    # if order >= 2:
    #   for k in range(train_length-1):
    #      for i in range(int(memory_length)):
    #         for j in range(int(memory_length)):
    #            if i <= j:
    #               vol_2nd[k,j+sum(int(memory_length)-1)-sum(int(memory_length)-1-i)]=train_zero[i+k]*train_zero[j+k]
    if order >= 3:
        for i in range(train_length - 1):
            temp = train_zero_u[i:i + memory_length_u]
            for j in range(memory_length_u):
                for k in range(memory_length_u):
                    F3[j, k] = temp[j] * temp[j] * np.conj(temp[k])
            vol_3rd[i, :] = F3.reshape(1, -1)

    if order == 3:
        F = np.hstack((vol_1st, vol_3rd))
    else:
        F = np.hstack((vol_bias, vol_1st))
    F_H = np.conj(F.T)
    F_H_inv = np.linalg.inv(F_H.dot(F))
    F_H_t = F_H.dot(train_tx)
    wiener = F_H_inv.dot(F_H_t)
    # wiener=np.linalg.inv((F.T).dot(F)).dot((F.T).dot(train_tx))
    rx_vol = F_test.dot(wiener)[tap_number:len(F_test.dot(wiener)) - tap_number]
    return train_rx, F, wiener, F_test, rx_vol, test_tx[tap_number:len(F_test.dot(wiener)) - tap_number]


def LeastMeanSquAlgorithm(tx, rx, tap_numb, mu=1.37e-3, iterate=50, title='Least Mean Square Error Algorithm'):
    T = len(rx)
    sig = rx
    N = tap_numb
    Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
    X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
    for i in range(Lp):
        X[:, i] = (np.flipud(sig[i:i + N + 1]).T)
    e = np.zeros(Lp, dtype=complex);  # used to save instant error
    f = np.zeros(N + 1, dtype=complex);  # 0.001
    for k in range(iterate):
        for i in range(Lp):
            e[i] = tx[i] - np.dot(np.conj(f.T), X[:, i])  # *np.dot(np.conj(X[:,i]),f)     # original instant error, R=2
            f = f + mu * np.conj(e[i]) * X[:, i]  # original weight update
            # f[P]=1
    sb = 1 * np.dot(np.conj(f), X)
    constellation_plot(sb, title=title, bins=75)
    return sb


def NormaLMS(tx, rx, mu, tap_numb=1, iterate=50, title='normalized Least Mean Square Algorithm'):
    T = len(rx)
    sig = rx
    N = tap_numb
    Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
    X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
    for i in range(Lp):
        X[:, i] = (np.flipud(sig[i:i + N + 1]).T)
    e = np.zeros(Lp, dtype=complex);  # used to save instant error
    f = np.zeros(N + 1, dtype=complex);  # 0.001
    for k in range(iterate):
        for i in range(Lp):
            e[i] = tx[i] - np.dot(np.conj(f.T), X[:, i])  # *np.dot(np.conj(X[:,i]),f)     # original instant error, R=2
            f = f + mu * np.conj(e[i]) * X[:, i] / abs(X[:, i]) ** 2  # original weight update
            # f[P]=1
    sb = 1 * np.dot(np.conj(f), X)
    constellation_plot(sb, title=title, bins=75)
    return


def Onetap_NLMS(tx, rx, mu, iterate=50, title='1-tap NLMS algorithm for phase recovery'):
    T = len(rx)
    sig = rx
    Lp = T;  # remove several first samples to avoid 0 or negative subscript
    e = np.zeros(Lp, dtype=complex);  # used to save instant error
    f = np.zeros(1, dtype=complex);  # 0.001
    for k in range(iterate):
        for i in range(Lp):
            e[i] = tx[i] - f * sig[i]
            f = f + mu * e[i] * np.conj(sig[i]) / abs(sig[i]) ** 2  # original weight update
            # f[P]=1
    sb = f * sig
    constellation_plot(sb, title=title, bins=75)
    return


def ConstModulusAlgorithm(rx, tap_numb, mu, const, iterate=50, title='Constant Modulus Algorithm'):
    T = len(rx)
    N = tap_numb;  # smoothing length N+1
    # Lh=5;  # channel length = Lh+1
    P = round((N) / 2);  # equalization delay
    sig = rx
    Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
    X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
    x0 = np.hstack((np.zeros(1), sig))
    for i in range(Lp):
        # X[:,i]=np.conj(x0[i+N+1:i:-1].T)
        X[:, i] = np.flipud(x0[i + N + 1:i:-1]).T

    e = np.zeros(Lp, dtype=complex);  # used to save instant error
    f = np.zeros(N + 1, dtype=complex);
    f[P] = 1;  # initial condition
    R2 = const  # np.sqrt(2);                  # constant modulas of QPSK symbols
    mu = mu  # 0.000271;      # parameter to adjust convergence and steady error 16QAM is samller than QAM

    for k in range(iterate):
        for i in range(Lp):
            y = np.dot(np.conj(f.T), X[:, i])  # update y
            # e[i]=(abs((y.real-2*np.sign(y.real))+1j*(y.imag-2*np.sign(y.imag)))**2-R2)*(y-2*np.sign(y.real)-1j*2*np.sign(y.imag))     # ModifiedCMA cost function, R2=2
            e[i] = y * (abs(y) ** 2 - R2)  # original instant error, R=2/10 for QAM/16QAM
            # e[i]=y.real*(abs(y.real)**2-R2)+1j*y.imag*(abs(y.imag)**2-R2)     # original instant error, R=2/10 for QAM/16QAM
            f = f - mu * np.conj(e[i].T) * X[:, i]  # original update equalizer coefficiency
            # f[P]=1
        # print(f)
    sb = 1 * np.dot(np.conj(f.T), X)
    constellation_plot(sb, title=title, bins=75)

    return sb


def MIMO_CMA(rx_signal_x, rx_signal_y, tap_numb, mu=2.31e-6, const=2, iterate=50):
    T = len(rx_signal_x)
    N = tap_numb;  # smoothing length N+1
    Lh = 0;  # channel length = Lh+1
    P = round((N + Lh) / 2);  # equalization delay
    sig_x = rx_signal_x
    sig_y = rx_signal_y

    Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
    X_x = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
    x0_x = np.hstack((np.zeros(1), sig_x))
    X_y = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
    x0_y = np.hstack((np.zeros(1), sig_y))
    for i in range(Lp):
        X_x[:, i] = np.flipud(x0_x[i + N + 1:i:-1]).T
        X_y[:, i] = np.flipud(x0_y[i + N + 1:i:-1]).T

    e_x = np.zeros(Lp, dtype=complex);  # used to save instant error
    e_y = np.zeros(Lp, dtype=complex)
    f_xx = np.zeros(N + 1, dtype=complex);
    f_xx[P] = 1;  # initial condition
    f_xy = np.zeros(N + 1, dtype=complex);
    f_xy[P] = 0;  # initial condition
    f_yx = np.zeros(N + 1, dtype=complex);
    f_yx[P] = 0;  # initial condition
    f_yy = np.zeros(N + 1, dtype=complex);
    f_yy[P] = 1;  # initial condition

    R2 = const  # np.sqrt(2);                  # constant modulas of QPSK symbols
    mu = mu;  # parameter to adjust convergence and steady error 16QAM is samller than QAM

    for k in range(iterate):
        for i in range(Lp):
            y_x = np.dot(np.conj(f_xx.T), X_x[:, i]) + np.dot(np.conj(f_xy.T), X_y[:, i])  # update y
            y_y = np.dot(np.conj(f_yx.T), X_x[:, i]) + np.dot(np.conj(f_yy.T), X_y[:, i])
            e_x[i] = y_x * (abs(y_x) ** 2 - R2)  # original instant error, R=2
            e_y[i] = y_y * (abs(y_y) ** 2 - R2)
            # e[i]=y.real*(abs(y.real)**2-R2)+1j*y.imag*(abs(y.imag)**2-R2)     # original instant error, R=1
            f_xx = f_xx - mu * np.conj(e_x[i].T) * X_x[:, i]  # original update equalizer coefficiency
            f_xy = f_xy - mu * np.conj(e_x[i].T) * X_y[:, i]
            f_yx = f_yx - mu * np.conj(e_y[i].T) * X_x[:, i]
            f_yy = f_yy - mu * np.conj(e_y[i].T) * X_y[:, i]
            # f[P]=1
    sb_x = np.dot(np.conj(f_xx.T), X_x) + np.dot(np.conj(f_xy.T), X_y)
    sb_y = np.dot(np.conj(f_yx.T), X_x) + np.dot(np.conj(f_yy.T), X_y)

    x = sb_x.real  # [10000:16000]#rx_signal_x.real
    y = sb_x.imag  # [10000:16000]#rx_signal_x.imag
    plt.hist2d(x, y, bins=75, range=np.array([(min(x) - 0.1, max(x) + 0.1), (min(y) - 0.1, max(y) + 0.1)]),
               cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    x = sb_y.real  # [10000:16000]#rx_signal_x.real
    y = sb_y.imag  # [10000:16000]#rx_signal_x.imag
    plt.hist2d(x, y, bins=75, range=np.array([(min(x) - 0.1, max(x) + 0.1), (min(y) - 0.1, max(y) + 0.1)]),
               cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    return sb_x, sb_y


def FreqOffsetComp(rx, fsamp=2.5e9, fres=1e5,
                   title='Coarse Frequency Compensation'):  # CoarseFrequencyOffset compensation based on prediogram method for M-QAM
    fr = fres;
    fs = fsamp;
    m_order = 4;  # if formate is QAM, morder is 4
    Nfft = 2 ** int(np.floor(np.log2(fs / fr)));
    # if Nfft>1e7:
    #    Nfft=2**23   # avoid memory error
    # elif Nfft<1e3:
    #    Nfft=2**10
    N = len(rx);
    # raiseSig=rx**m_order
    absFFTSig = abs(np.fft.fft(rx ** m_order, Nfft))
    # plt.plot(absFFTSig/max(absFFTSig));plt.xlabel('index');plt.ylabel('magnitude');plt.show();
    maxIndex = np.argmax(absFFTSig)
    if maxIndex > Nfft / 2:
        maxIndex = maxIndex - Nfft
    estFreqOffset = fs / Nfft * (maxIndex - 1) / m_order;
    print('Frequency offset=', estFreqOffset)
    rx = rx * np.exp(-estFreqOffset * np.linspace(1, N, N) * 2 * np.pi * 1j / fs)
    # constellation_plot(rx, title=title, bins=75)
    return estFreqOffset, rx


def FreqOffsetComp2(rx, fsamp=2.5e9, fres=1e5,
                    title='CoarseFrequencyCompensation'):  # CoarseFrequencyOffset compensation based on prediogram method for M-QAM
    fr = fres;
    fs = fsamp;
    m_order = 4;  # QAM is 4
    Nfft = 2 ** int(np.floor(np.log2(fs / fr)));
    N = len(rx);
    # if Nfft>1e6:
    #    Nfft=2**18
    # N=len(rx);
    # raiseSig=rx**m_order
    absFFTSig = abs(np.fft.fft(rx ** m_order, Nfft))
    # plt.plot(absFFTSig/max(absFFTSig));plt.xlabel('index');plt.ylabel('magnitude');plt.show();
    psd = np.fft.fftshift(absFFTSig)
    maxIndex = np.argmax(psd)
    offsetIndex = maxIndex - Nfft / 2
    estFreqOffset = fs / Nfft * (offsetIndex - 1) / m_order;
    print(estFreqOffset)
    rx = rx * np.exp(-estFreqOffset * np.linspace(1, N, N) * 2 * np.pi * 1j / fs)
    # plt.plot(rx.real,rx.imag,'.');plt.show()
    constellation_plot(rx, title=title, bins=75)
    return estFreqOffset, rx


def PLL_(rx_vol, bn, damp_factor, title='Carrier Phase Recovery'):  # PLL emulation by DDS_VCO
    Bn = bn  # 0.025 #0.01~0.05 is suitable range #larger bandwidth is more suitable & convergence to target
    damp_f = damp_factor  # 3 #0.707 is pre-value #larger damping factor is more suitable & convergence to target
    k0 = 1
    kp = 2  # 2 for QAM
    theta = Bn * (damp_f + (4 * damp_f) ** -1) ** -1
    d = 1 + 2 * theta * damp_f + theta ** 2
    g1 = 4 * (theta ** 2) / (d * k0 * kp)
    gp = 4 * damp_f * theta / (d * k0 * kp)
    previousSample = 0
    phase = 0
    loopfiltState = 0
    integfiltState = 0
    DDSpreInp = 0
    digitalSynchGain = -1
    output = np.zeros(len(rx_vol), dtype=complex)
    phaseCorrection = np.zeros(len(rx_vol))
    p = np.zeros(len(rx_vol))
    for i in range(len(rx_vol)):
        # phase detector
        phErr = np.sign(previousSample.real) * previousSample.imag - np.sign(previousSample.imag) * previousSample.real
        output[i] = rx_vol[i] * np.exp(1j * phase)
        p[i] = phErr
        # loop filter
        loopfiltout = phErr * g1 + loopfiltState
        loopfiltState = loopfiltout
        # DDS
        DDSout = DDSpreInp + integfiltState
        integfiltState = DDSout
        DDSpreInp = phErr * gp + loopfiltout
        phase = DDSout * digitalSynchGain
        phaseCorrection[i] = phase
        previousSample = output[i]

    constellation_plot(output, title=title, bins=75)
    return phaseCorrection, output, p


def SNRcal_(T, R):
    if len(T) < len(R):
        N = len(T)
    else:
        N = len(R)
    absR = np.zeros(N)
    absR_T = np.zeros(N)
    for i in range(N):
        absR[i] = abs(R[i]) ** 2
        absR_T[i] = abs(T[i] - R[i]) ** 2
    snr = 10 * np.log10(np.mean(absR) / (np.mean(absR_T)))
    print('snr=', snr)
    return snr


def EVM(T, R):
    s = 0
    s1 = 0
    if len(T) < len(R):
        N = len(T)
    else:
        N = len(R)
    for i in range(N):
        t = abs(R[i] - T[i]) ** 2
        t1 = abs(T[i]) ** 2
        s = s + t
        s1 = s1 + t1
    evm = np.sqrt(np.mean(s) / np.mean(s1))
    evm_dB = 10 * np.log10(np.mean(s) / np.mean(s1))
    # evm2snr=1/np.log2(16)*evm**-2
    print('evm,evm2snr=', evm, evm_dB)
    return evm, evm_dB


def SNR2BER(snr, data_format):
    snr_ratio = 10 ** (snr / 10)
    un = np.sqrt(1.5 * snr_ratio / (data_format - 1))
    BER = 2 * (1 - data_format ** (-0.5)) / np.log2(data_format) * (math.erfc(un) + math.erfc(3 * un))
    print('BER=', BER)
    return BER


def BER(T, R, PAM_order):
    count = 0
    R = Sym2Bit(R, PAM_order)
    for i in range(len(R)):
        if R[i] != T[i]:
            count = count + 1
    print('count=', count)
    return count


def BER_count(T, R, PAM_order):
    count = 0
    R_r = Sym2Bit(R.real, PAM_order)
    R_i = Sym2Bit(R.imag, PAM_order)
    R = R_r + 1j * R_i
    index = []
    if len(T) < len(R):
        N = len(T)
    else:
        N = len(R)
    for i in range(N):
        if R[i] != T[i]:
            count = count + 1
            index.append(i)
    print('error count=', count);  # print('error location',index)
    print('BER count=', count / N)
    return count, index, count / N


def constellation_plot(sig, title='signal', bins=75):
    x = sig.real  # [10000:16000]#rx_signal_x.real
    y = sig.imag  # [10000:16000]#rx_signal_x.imag
    plt.hist2d(x, y, bins=bins, range=np.array([(min(x) - 0.1, max(x) + 0.1), (min(y) - 0.1, max(y) + 0.1)]),
               cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title)
    plt.show()


def error_mark(sig, err_sig):
    err_length = len(err_sig)
    if err_length > 0:
        err_index = np.zeros(err_length, dtype=int)
        err_point = np.zeros(err_length, dtype=complex)
        for i in range(err_length):
            err_index[i] = err_sig[i]
            err_point[i] = sig[err_index[i]]
        p1 = plt.plot(sig.real, sig.imag, 'g+')
        p2 = plt.plot(err_point.real, err_point.imag, 'ro', mfc='none', label='error')
        plt.title('scatter plot w/ error point marker')
        plt.legend(loc=1)
        plt.show()
    else:
        print("\033[35m>>>>>>> there is no error point can plot...! <<<<<<<")
    return 0


end_time = time.time()
time_diff = end_time - start_time
print("\033[36mdata import & func. loading iime is: " + str(timedelta(seconds=int(round(time_diff)))))
# %%--------------------------setting enviroment---------------------------#

start_time = time.time()

cmath.sqrt(-1)
PAM_order = 4
Tx_setting_symbol_rate = 28e9
Rx_setting_sample_rate = 23 * 28e9
Rx_setting_upsample_number = 1
Rx_setting_sample_per_symbol = int(Rx_setting_sample_rate / Tx_setting_symbol_rate)
Rx_setting_sample_per_symbol_resample = Rx_setting_sample_rate / Tx_setting_symbol_rate * Rx_setting_upsample_number
eye_position = 11  # 39#is bessel
# start=1
# window_length=150
prbs = 14
train_length = 1750  # int(0.07*len(rx_signal_x))
memory_length = total_tap_number(D=16, L=80, lemda=1.55e-6, Baudrate=28e9)  # 127#159#83#163#233
Bn = 0.003214  # 0.001725#0.005214
damp_f = 133.707  # 133.707
# ---------------------DSP flow----------------------#

tx_i_signal = dn_sample(Sym2Bit(datanromalize(tx_xi, PAM_order), PAM_order)[eye_position - 1::],
                        Rx_setting_sample_per_symbol)
tx_q_signal = dn_sample(Sym2Bit(datanromalize(tx_xq, PAM_order), PAM_order)[eye_position - 1::],
                        Rx_setting_sample_per_symbol)
tx_signal_x = Tx_constellation(tx_i_signal, tx_q_signal)
rx_i_signal = dn_sample(datanromalize(rx_xi, PAM_order)[eye_position - 1::], Rx_setting_sample_per_symbol)
rx_q_signal = dn_sample(datanromalize(rx_xq, PAM_order)[eye_position - 1::], Rx_setting_sample_per_symbol)
rx_signal_x = Tx_constellation(rx_i_signal, rx_q_signal)
# tx_i_signal=dn_sample(Sym2Bit(datanromalize(tx_yi,PAM_order),PAM_order)[eye_position-1::],Rx_setting_sample_per_symbol)
# tx_q_signal=dn_sample(Sym2Bit(datanromalize(tx_yq,PAM_order),PAM_order)[eye_position-1::],Rx_setting_sample_per_symbol)
# tx_signal_y=Tx_constellation(tx_i_signal,tx_q_signal)
# rx_i_signal=dn_sample(datanromalize(rx_yi,PAM_order)[eye_position-1::],Rx_setting_sample_per_symbol)
# rx_q_signal=dn_sample(datanromalize(rx_yq,PAM_order)[eye_position-1::],Rx_setting_sample_per_symbol)
# rx_signal_y=Tx_constellation(rx_i_signal,rx_q_signal)

plt.plot(rx_signal_x.real, rx_signal_x.imag, 'g+', tx_signal_x.real, tx_signal_x.imag, 'ro')  # plot scatter
plt.xlim(-1 + min(rx_signal_x.real), 1 + max(rx_signal_x.real))
plt.ylim(-1 + min(rx_signal_x.imag), 1 + max(rx_signal_x.imag))
plt.show()

# sb=ConstModulusAlgorithm(rx_signal_x,tap_numb=27,mu=0.0000023,const=2,iterate=50)     #qam test
# sb=ConstModulusAlgorithm(rx_signal_x,tap_numb=53,mu=0.000013,const=2,iterate=50)    #modified MCMA 16QAM
# phasecorrection,rx_pll,p=PLL_(sb,Bn,damp_f)
# rx_LMS=LeastMeanSquAlgorithm(tx_signal_x,rx_signal_x,tap_numb=20,mu=1.37e-3,iterate=50)
[mm, ss, aa, bb, yy] = data_synch(tx_signal_x, rx_signal_x, prbs)

# train_data_r,F_r,w_r,R_testr,rx_vol_R,test_dataR=vol_(train_length,memory_length,aa.real,bb.real,1)
# train_data_i,F_i,w_i,R_testi,rx_vol_I,test_dataI=vol_(train_length,memory_length,aa.imag,bb.imag,1)

# train_data,F,w,R_test,rx_vol,test_data=vol_pre(train_length,memory_length,aa,bb,1)
train_data, F, w, R_test, rx_vol, test_data = vol_c(train_length, 2 * memory_length + 1, aa, bb, 1)
phasecorrection, rx_pll, p = PLL_(rx_vol, Bn, damp_f)
rx_LMS = LeastMeanSquAlgorithm(test_data, rx_pll, tap_numb=2 * memory_length + 1, mu=3.327e-6, iterate=50)
# cmap=plt.cm.Reds,cmap=plt.cm.BuPu,cmap=plt.cm.jet,plt.cm.hot

evm = EVM(test_data[0:len(rx_vol)], rx_vol)
snr = SNRcal_(test_data[0:len(rx_vol)], rx_vol)
vol_BERcount = BER_count(test_data[0:len(rx_pll)], rx_vol, PAM_order)

evm = EVM(test_data[0:len(rx_pll)], rx_pll)
snr = SNRcal_(test_data[0:len(rx_pll)], rx_pll)
pll_BERcount = BER_count(test_data[0:len(rx_pll)], rx_pll, PAM_order)
# error_mark(rx_pll,pll_BERcount[1])

evm = EVM(test_data[0:len(rx_vol)], rx_LMS)
snr = SNRcal_(test_data[0:len(rx_vol)], rx_LMS)
LMS_BERcount = BER_count(test_data[0:len(rx_pll)], rx_LMS, PAM_order)
error_mark(rx_LMS, LMS_BERcount[1])
# sb=ConstModulusAlgorithm(rx_signal_x,tap_numb=93,mu=0.00000273,const=10,iterate=50)
# [mm,ss,aa,bb,yy]=data_synch(tx_signal_x,sb,prbs)
# train_data,F,w,R_test,rx_vol,test_data=vol_c(train_length,memory_length,aa,bb,1)
# phasecorrection,rx_pll,p=PLL_(rx_vol,Bn,damp_f)

end_time = time.time()
time_diff = end_time - start_time
print("\033[36mdata_calculate & plot Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
# %%----------CMA+volterra+PLL---------
sb = ConstModulusAlgorithm(rx_signal_x, tap_numb=memory_length, mu=1.7e-6, const=10, iterate=50)
[mm, ss, aa, bb, yy] = data_synch(tx_signal_x, sb, prbs);
train_data, F, w, R_test, rx_vol, test_data = vol_c(train_length, 2 * memory_length + 1, aa, bb, 1)
# rx_LMS=LeastMeanSquAlgorithm(test_data,rx_vol,tap_numb=113,mu=1.27e-5,iterate=50)
phasecorrection, rx_pll, p = PLL_(rx_vol, Bn, damp_f)
rx_LMS = LeastMeanSquAlgorithm(test_data, rx_pll, tap_numb=2 * memory_length + 1, mu=3.327e-6, iterate=50)

evm = EVM(test_data[0:len(rx_vol)], rx_vol)
snr = SNRcal_(test_data[0:len(rx_vol)], rx_vol)
vol_BERcount = BER_count(test_data[0:len(rx_pll)], rx_vol, PAM_order)

evm = EVM(test_data[0:len(rx_pll)], rx_pll)
snr = SNRcal_(test_data[0:len(rx_pll)], rx_pll)
pll_BERcount = BER_count(test_data[0:len(rx_pll)], rx_pll, PAM_order)

evm = EVM(test_data[0:len(rx_vol)], rx_LMS)
snr = SNRcal_(test_data[0:len(rx_vol)], rx_LMS)
LMS_BERcount = BER_count(test_data[0:len(rx_pll)], rx_LMS, PAM_order)
error_mark(rx_LMS, LMS_BERcount[1])

end_time = time.time()
time_diff = end_time - start_time
print("\033[36mdata_calculate & plot Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
# %%----------CMA+CFC+volterra+PLL (DSP flow for LO asynch & delta frequency != 0)---------
start_time = time.time()

tx_xi = np.genfromtxt(r'data\KENG_optsim_py\20201130_DATA_2Laser_final\500KLW_1GFO_14Noise_80KM_initialphase225_50GBW_0dBLO\TxYI.txt')
tx_xq = np.genfromtxt(r'data\KENG_optsim_py\20201130_DATA_2Laser_final\500KLW_1GFO_14Noise_80KM_initialphase225_50GBW_0dBLO\TxYQ.txt')
rx_xi = np.genfromtxt(r'data\KENG_optsim_py\20201130_DATA_2Laser_final\500KLW_1GFO_14Noise_80KM_initialphase225_50GBW_0dBLO\RxYI.txt')
rx_xq = np.genfromtxt(r'data\KENG_optsim_py\20201130_DATA_2Laser_final\500KLW_1GFO_14Noise_80KM_initialphase225_50GBW_0dBLO\RxYQ.txt')
# %%--------------
cmath.sqrt(-1)
PAM_order = 4
Tx_setting_symbol_rate = 56e9
Rx_setting_sample_rate = 32 * 56e9
Rx_setting_upsample_number = 1
Rx_setting_sample_per_symbol = int(Rx_setting_sample_rate / Tx_setting_symbol_rate)
Rx_setting_sample_per_symbol_resample = Rx_setting_sample_rate / Tx_setting_symbol_rate * Rx_setting_upsample_number
eye_position = 25  # must try eye decision first in each simulation case
# start=1
# window_length=150
prbs = 13
train_length = 1750  # int(0.07*len(rx_signal_x))
memory_length = total_tap_number(D=16, L=80, lemda=1.55e-6, Baudrate=56e9)
# Bn = 0.003514  # 0.001725#0.005214
# damp_f = 133.707  # 133.707
Bn = 0.001  # 0.001725#0.005214
damp_f = 0.707  # 133.707
# ---------------------DSP flow CMA->FOC->corr->volterra->MaxmLikehood PLL->DD LMS----------------------#

tx_i_signal = dn_sample(Sym2Bit(datanromalize(tx_xi, PAM_order), PAM_order)[eye_position - 1::],
                        Rx_setting_sample_per_symbol)
tx_q_signal = dn_sample(Sym2Bit(datanromalize(tx_xq, PAM_order), PAM_order)[eye_position - 1::],
                        Rx_setting_sample_per_symbol)
tx_signal_x = Tx_constellation(tx_i_signal, tx_q_signal)
rx_i_signal = dn_sample(datanromalize(rx_xi, PAM_order)[eye_position - 1::], Rx_setting_sample_per_symbol)
rx_q_signal = dn_sample(datanromalize(rx_xq, PAM_order)[eye_position - 1::], Rx_setting_sample_per_symbol)
rx_signal_x = Tx_constellation(rx_i_signal, rx_q_signal)
constellation_plot(rx_signal_x, title='received signal', bins=75)

sb = ConstModulusAlgorithm(rx_signal_x, tap_numb=57, mu=1e-6, const=10, iterate=50)
frequency_offset, rx_foc = FreqOffsetComp(sb, fsamp=56e9, fres=1e5)  # fsamp=5.35e9,fres=2e5
[mm, ss, aa, bb, yy] = data_synch(tx_signal_x, rx_foc, prbs);
train_data, F, w, R_test, rx_vol, test_data = vol_c(train_length, memory_length, aa, bb, 1)
phasecorrection, rx_pll, p = PLL_(rx_vol, Bn, damp_f)
rx_LMS = LeastMeanSquAlgorithm(test_data, rx_pll, tap_numb=memory_length, mu=3.327e-6, iterate=50)

evm = EVM(test_data[0:len(rx_vol)], rx_vol)
snr = SNRcal_(test_data[0:len(rx_vol)], rx_vol)
vol_BERcount = BER_count(test_data[0:len(rx_pll)], rx_vol, PAM_order)

evm = EVM(test_data[0:len(rx_pll)], rx_pll)
snr = SNRcal_(test_data[0:len(rx_pll)], rx_pll)
pll_BERcount = BER_count(test_data[0:len(rx_pll)], rx_pll, PAM_order)

evm = EVM(test_data[0:len(rx_pll)], rx_LMS)
snr = SNRcal_(test_data[0:len(rx_pll)], rx_LMS)
LMS_BERcount = BER_count(test_data[0:len(rx_pll)], rx_LMS, PAM_order)
error_mark(rx_LMS, LMS_BERcount[1])
# -----------------DSP flow CMA->FOC->MaxmLikehood PLL->corr->volterra->DD LMS(somtimes error if rx is after EDC)----------------------------#

phasecorrection, rx_pll, p = PLL_(rx_foc, Bn, damp_f)
[mm, ss, aa, bb, yy] = data_synch(tx_signal_x, rx_pll, prbs);
train_data, F, w, R_test, rx_vol, test_data = vol_c(train_length, memory_length, aa, bb, 1)
rx_LMS = LeastMeanSquAlgorithm(test_data, rx_vol, tap_numb=memory_length, mu=3.327e-6, iterate=50)

evm = EVM(test_data[0:len(rx_vol)], rx_vol)
snr = SNRcal_(test_data[0:len(rx_vol)], rx_vol)
vol_BERcount = BER_count(test_data[0:len(rx_pll)], rx_vol, PAM_order)

evm = EVM(test_data[0:len(rx_pll)], rx_LMS)
snr = SNRcal_(test_data[0:len(rx_pll)], rx_LMS)
LMS_BERcount = BER_count(test_data[0:len(rx_pll)], rx_LMS, PAM_order)
error_mark(rx_LMS, LMS_BERcount[1])

end_time = time.time()
time_diff = end_time - start_time
print("\033[36mdata_calculate & plot Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
# %%-------------eye position decision for no FOC condition-----------------
cmath.sqrt(-1)
PAM_order = 4
Tx_setting_symbol_rate = 28e9
Rx_setting_sample_rate = 23 * 28e9
Rx_setting_upsample_number = 1
Rx_setting_sample_per_symbol = int(Rx_setting_sample_rate / Tx_setting_symbol_rate)
Rx_setting_sample_per_symbol_resample = Rx_setting_sample_rate / Tx_setting_symbol_rate * Rx_setting_upsample_number
prbs = 13
train_length = 1750  # int(0.07*len(rx_signal_x))
memory_length = 2 * total_tap_number(D=16, L=80, lemda=1.55e-6, Baudrate=30e9) + 1  # 157#67#is bessel
Bn = 0.003214  # 0.001725#0.005214
damp_f = 1423.707  # 133.707
evm_test = np.zeros(int(Rx_setting_sample_rate / Tx_setting_symbol_rate))
BER_test = np.zeros(int(Rx_setting_sample_rate / Tx_setting_symbol_rate))
for i in range(int(Rx_setting_sample_rate / Tx_setting_symbol_rate)):
    eye_position = i + 1
    tx_i_signal = dn_sample(Sym2Bit(datanromalize(tx_xi, PAM_order), PAM_order)[eye_position - 1::],
                            Rx_setting_sample_per_symbol)
    tx_q_signal = dn_sample(Sym2Bit(datanromalize(tx_xq, PAM_order), PAM_order)[eye_position - 1::],
                            Rx_setting_sample_per_symbol)
    tx_signal_x = Tx_constellation(tx_i_signal, tx_q_signal)
    rx_i_signal = dn_sample(datanromalize(rx_xi, PAM_order)[eye_position - 1::], Rx_setting_sample_per_symbol)
    rx_q_signal = dn_sample(datanromalize(rx_xq, PAM_order)[eye_position - 1::], Rx_setting_sample_per_symbol)
    rx_signal_x = Tx_constellation(rx_i_signal, rx_q_signal)
    [mm, ss, aa, bb, yy] = data_synch(tx_signal_x, rx_signal_x, prbs)
    train_data, F, w, R_test, rx_vol, test_data = vol_c(train_length, memory_length, aa, bb, 1)
    phasecorrection, rx_pll, p = PLL_(rx_vol, Bn, damp_f)
    evm = EVM(test_data[0:len(rx_pll)], rx_pll)
    evm_test[i] = evm[0]
    pll_BERcount = BER_count(test_data[0:len(rx_pll)], rx_pll, PAM_order)
    BER_test[i] = pll_BERcount[0]
    print('eye position=', eye_position)

plt.plot(np.linspace(1, int(Rx_setting_sample_rate / Tx_setting_symbol_rate),
                     int(Rx_setting_sample_rate / Tx_setting_symbol_rate)), evm_test, 'o-')
plt.show()
plt.plot(np.linspace(1, int(Rx_setting_sample_rate / Tx_setting_symbol_rate),
                     int(Rx_setting_sample_rate / Tx_setting_symbol_rate)), BER_test, 'o-')
plt.show()
# sb=ConstModulusAlgorithm(rx_signal_x,tap_numb=13,mu=0.000005,const=10,iterate=1)
# [mm,ss,aa,bb,yy]=data_synch(tx_signal_x,sb,prbs)
# train_data,F,w,R_test,rx_vol,test_data=vol_c(train_length,memory_length,aa,bb,1)
# phasecorrection,rx_pll,p=PLL_(rx_vol,Bn,damp_f)

# %%------------
r = np.zeros(12, dtype=complex)
for i in range(12):
    r[i] = np.exp(1j * (np.pi * i / 2 + np.pi / 4)) * np.exp(1j * (i * 0.0073 + 0.13 * np.pi))
# plt.plot(r.real,r.imag,'g*')
for i in np.arange(11):
    if i < len(r):
        t = (r[i + 1] * np.conj(r[i])) ** 4
        ss = ss + t
s_mean = ss / 11
freq = np.arctan(s_mean.imag / s_mean.real) / 4
r1 = np.zeros(12, dtype=complex)
for i in range(12):
    r1[i] = r[i] * np.exp(-1j * (i * freq))
plt.plot(r.real, r.imag, 'g*', r1.real, r1.imag, 'r*')

# %%-------------PLL by DDS-vco----------

Bn = 0.001  # 0.01~0.05 is suitable range #larger bandwidth is more suitable & convergence to target
damp_f = 0.707  # 0.707 is pre-value #larger damping factor is more suitable & convergence to target
k0 = 1
kp = 2  # for QAM
theta = Bn * (damp_f + (4 * damp_f) ** -1) ** -1
d = 1 + 2 * theta * damp_f + theta ** 2
g1 = 4 * (theta ** 2) / (d * k0 * kp)
gp = 4 * damp_f * theta / (d * k0 * kp)
previousSample = 0
phase = 0
loopfiltState = 0
integfiltState = 0
DDSpreInp = 0
digitalSynchGain = -1
output = np.zeros(len(rx_vol_mod), dtype=complex)
phaseCorrection = np.zeros(len(rx_vol_mod))
p = np.zeros(len(rx_vol_mod))
for i in range(len(rx_vol_mod)):
    # phase detector
    phErr = np.sign(previousSample.real) * previousSample.imag - np.sign(previousSample.imag) * previousSample.real
    output[i] = rx_vol[i] * np.exp(1j * phase)
    p[i] = phErr
    # loop filter
    loopfiltout = phErr * g1 + loopfiltState
    loopfiltState = loopfiltout
    # DDS
    DDSout = DDSpreInp + integfiltState
    integfiltState = DDSout
    DDSpreInp = phErr * gp + loopfiltout
    phase = DDSout * digitalSynchGain
    phaseCorrection[i] = phase
    previousSample = output[i]

x = output.real  # rx_signal_x.real
y = output.imag  # rx_signal_x.imag
plt.plot(rx_vol_mod.real, rx_vol_mod.imag, 'r+', x, y, 'g*')
# plt.hist2d(x,y, bins=200, range=np.array([(min(x)-0.1, max(x)+0.1), (min(y)-0.1, max(y)+0.1)]), cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()

# %%----------eye plot-----
# for raw data
for i in range(335):
    plt.plot(np.arange(0, 46 * 2), rx_xi[i * 46 * 2:i * 46 * 2 + 46 * 2], 'g-')
# %%
# for RX
for i in range(335):
    plt.plot(np.arange(0, 3), rx_pll.real[i * 3:i * 3 + 3], 'g-')

# %%--------------------3D KDE plot-------------------------
# n_components = 3
# Define the borders
# deltaX = (max(x) - min(x))/10
# deltaY = (max(y) - min(y))/10
# xmin = min(x) - deltaX
# xmax = max(x) + deltaX
# ymin = min(y) - deltaY
# ymax = max(y) + deltaY
# print(xmin, xmax, ymin, ymax)
# Create meshgrid
# xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
# positions = np.vstack([xx.ravel(), yy.ravel()])
# values = np.vstack([x, y])
# kernel = st.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, xx.shape)
# fig = plt.figure(figsize=(8, 7))
# ax = plt.axes(projection='3d')
# surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('PDF')
# ax.set_title('Surface plot of Gaussian 2D KDE')
# fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
# ax.view_init(60, 35)

# --------------2d histgram-------------------
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mp
# from matplotlib.ticker import NullFormatter
# from matplotlib.ticker import MultipleLocator


# Load FCS file using loadFCS from fcm


# definitions for the axes
# left, width = 0.1, 0.65
# bottom, height = 0.1, 0.65
# bottom_h = left_h = left+width+0.02

# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom_h, width, 0.2]
# rect_histy = [left_h, bottom, 0.2, height]

# plt.figure(1, figsize=(7,7))


# axScatter = plt.axes(rect_scatter)
# axHistx = plt.axes(rect_histx)
# axHistx.xaxis.set_major_formatter(NullFormatter())
# axHistx.xaxis.set_minor_formatter(NullFormatter())
# axHisty = plt.axes(rect_histy)
# axHisty.yaxis.set_major_formatter(NullFormatter())
# axHisty.yaxis.set_minor_formatter(NullFormatter())

# axHistx.xaxis.set_major_locator( MultipleLocator(10) )
# axHistx.xaxis.set_minor_locator( MultipleLocator(10) )
# axHisty.yaxis.set_major_locator( MultipleLocator(10) )
# axHisty.yaxis.set_minor_locator( MultipleLocator(10) )

# the scatter plot:
# axScatter.scatter(x, y, edgecolors='none')


# now determine nice limits by hand:
# binwidth = 0.1
# xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
# lim = ( int(xymax/binwidth) + 1) * binwidth

# axScatter.set_xlim( (-lim, lim) )
# axScatter.set_ylim( (-lim, lim) )

# bins = np.arange(-lim, lim + binwidth, binwidth)
# axHistx.hist(x, bins=50, facecolor='blue', edgecolor='blue', histtype='stepfilled')
# axHisty.hist(y, bins=50, orientation='horizontal', facecolor='blue', edgecolor='blue',  histtype='stepfilled')

# axHistx.set_xlim( axScatter.get_xlim() )
# axHistx.set_xticks(())
# axHistx.set_yticks(())
# axHisty.set_ylim( axScatter.get_ylim() )
# axHisty.set_xticks(())
# axHisty.set_yticks(())

# plt.show()

# %%--------------------
pcb_s = np.genfromtxt(r'C:\Users\0668\Desktop\SYZ Sweep 1_DiffParams Diff SYZ 3.txt')
temp = np.zeros((501, 9))
temp[:, 0] = (pcb_s[:, 0] - 5e-09) * 1000000000
temp[:, 1] = pcb_s[:, 1] - 5e-009
temp[:, 2] = np.zeros(501)
temp[:, 3] = pcb_s[:, 2] - 5e-009
temp[:, 4] = np.zeros(501)
temp[:, 5] = pcb_s[:, 3] - 5e-009
temp[:, 6] = np.zeros(501)
temp[:, 7] = pcb_s[:, 4] - 5e-009
temp[:, 8] = np.zeros(501)
np.savetxt(r'C:\Users\0668\Desktop\test.txt', temp, fmt='%2.5e')

S_para = np.genfromtxt(r'C:\Users\0668\Desktop\test.txt')
plt.figure(figsize=(14, 6))
for i in np.arange(0, 4, 1):
    plt.subplot(2, 4, i + 1)
    plt.plot(S_para[0:, 0], S_para[0:, 2 * i + 1], 'g')
    plt.title('S')
    plt.subplot(2, 4, i + 5)
    plt.plot(S_para[0:, 0], S_para[0:, 2 * i + 2])
    plt.title('phase')
# ----------------------
train_rx = bb.real[0:train_length]
test_rx = bb.real[train_length::]
train_tx = aa.real[0:train_length]
train_length = len(train_rx)
test_length = len(bb.real) - train_length
tap_number = int((memory_length - 1) / 2)
feature_1st = memory_length
feature_2nd = int((memory_length + 1) * memory_length / 2)
# ----------train sequency----------
xx = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0]
train_length = 8
memory_length = 5
tap_number = int((memory_length - 1) / 2)
train_zero = np.hstack((np.zeros(tap_number), train_rx, np.zeros(tap_number)))
vol_bias = np.ones([train_length, 1])
vol_1st = np.zeros([train_length, feature_1st])
vol_2nd = np.zeros([train_length, feature_2nd])
test_matrix = np.zeros([memory_length, memory_length])
for i in range(memory_length):
    for j in range(memory_length):
        if i < j:
            test_matrix[i, j] = 0
        else:
            test_matrix[i, j] = train_zero[i] * train_zero[j]
for i in range(train_length):
    vol_1st[i, :] = train_zero[i:i + feature_1st]
for k in range(train_length):
    for i in range(memory_length):
        for j in range(memory_length):
            test_matrix(i, j) = train_zero[i + k] * train_zero[j + k]
    vol_2nd[k, :] = np.triu(test_matrix, 0).reshape(1, -1)
F = np.hstack((vol_bias, vol_1st, vol_2nd))
# F=np.hstack((vol_bias,vol_1st))
wiener = np.linalg.inv((F.T).dot(F)).dot((F.T).dot(train_tx))
rx_vol = F.dot(wiener)

test_length = 8
memory_length = 5
feature_2nd = int((memory_length + 1) * memory_length / 2)
xx = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0]
test_1 = np.ones([test_length, 1])
test_F = np.zeros([test_length, memory_length])
test_m = np.zeros([memory_length, memory_length])
vol_2nd = np.zeros([test_length, feature_2nd])
for i in range(test_length):
    test_F[i, :] = xx[i:i + memory_length]
for k in range(test_length):
    for i in range(memory_length):
        for j in range(memory_length):
            test_m[i, j] = xx[i + k] * xx[j + k]
    vol_2nd[k, :] = np.triu(test_m, 0).reshape(1, -1)
FF = np.hstack((test_1, test_F))

# %%----------eye plot-----

for i in range(335):
    plt.plot(np.arange(0, 5), rx_vol.real[i * 5:i * 5 + 5], 'r-')

# %%-------------------PLL ex1----------------
# PLL in an SDR

# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

k = 150.1
N = 1
K_0 = 1  # fixed
Bn_Fs = 0.05
Dp_factor = 1 / np.sqrt(2)
K_D = 0.5  # complex input=amplitude, real input=amplitude/2
K_p = 4 * Dp_factor * Bn_Fs / ((K_D * K_0) * (Dp_factor + (4 * Dp_factor) ** -1))  # 0.2667
K_i = 4 * Bn_Fs ** 2 / ((K_D * K_0) * (Dp_factor + 0.25 * (Dp_factor) ** -1) ** 2)  # 0.0178

input_signal = np.zeros(150)

for n in range(149):
    input_signal[n] = np.cos(2 * np.pi * (k / N) * n + np.pi / 1)

integrator_out = 0
phase_estimate = np.zeros(150)
e_D = []  # phase-error output
e_F = []  # loop filter output
sin_out = np.zeros(150)
cos_out = np.ones(150)

for n in range(149):
    # input_signal[n] = np.cos(2*np.pi*(k/N)*n + np.pi)

    # phase detector
    try:
        e_D.append(input_signal[n] * sin_out[n])
    except IndexError:
        e_D.append(0)

    # loop filter
    integrator_out += K_i * e_D[n]
    e_F.append(K_p * e_D[n] + integrator_out)

    # NCO
    try:
        phase_estimate[n + 1] = phase_estimate[n] + K_0 * e_F[n]
    except IndexError:
        phase_estimate[n + 1] = K_0 * e_F[n]

    sin_out[n + 1] = -np.sin(2 * np.pi * (k / N) * (n + 1) + phase_estimate[n])
    cos_out[n + 1] = np.cos(2 * np.pi * (k / N) * (n + 1) + phase_estimate[n])

# Create a Figure
fig = plt.figure()

# Set up Axes
ax1 = fig.add_subplot(211)
ax1.plot(cos_out, label='PLL Output')
# ax1.plot(e_D, label='e_D')
plt.grid()
ax1.plot(input_signal, label='Input Signal')
# ax1.plot(e_F, label='e_F')
plt.legend()
ax1.set_title('Waveforms')

# Show the plot
# plt.show()

ax2 = fig.add_subplot(212)
ax2.plot(phase_estimate)
# ax2.plot(e_F)
plt.grid()
ax2.set_title('Filtered Error')
plt.show()

# %%-------------------PLL ex2 complex signal----------------

# PLL in an SDR

# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

k = 25e9
N = 1
K_0 = 1  # fixed
Bn_Fs = 0.05
Dp_factor = 0.45  # 1/np.sqrt(2)
K_D = 1  # complex input=amplitude, real input=amplitude/2
K_p = 4 * Dp_factor * Bn_Fs / ((K_D * K_0) * (Dp_factor + (4 * Dp_factor) ** -1))  # 0.2667
K_i = 4 * Bn_Fs ** 2 / ((K_D * K_0) * (Dp_factor + 0.25 * (Dp_factor) ** -1) ** 2)  # 0.0178

input_signal = np.zeros(200, dtype=complex)
out_signal = np.zeros(200, dtype=complex)
sin_out = np.zeros(200)
cos_out = np.ones(200)

for n in range(199):
    input_signal[n] = np.complex(np.cos(2 * np.pi * (k / N) * n + np.pi * 1 / 2), np.sin(
        2 * np.pi * (k / N) * n + np.pi * 1 / 2))  # if phase is pi*N cant be convergence
    out_signal[n] = np.complex(cos_out[n], -sin_out[n])

integrator_out = 0
phase_estimate = np.zeros(200)
e_D = []  # phase-error output
e_F = []  # loop filter output

for n in range(199):
    # input_signal[n] = np.cos(2*np.pi*(k/N)*n + np.pi)

    # phase detector
    try:
        temp = input_signal.real[n] * out_signal.real[n] - input_signal.imag[n] * out_signal.imag[n] + 1j * (
                input_signal.imag[n] * out_signal.real[n] + input_signal.real[n] * out_signal.imag[n])
        # e_D.append(np.arcsin(temp.imag/abs(temp)))
        e_D.append(np.arctan(temp.imag / temp.real))
    except IndexError:
        e_D.append(0)

    # loop filter
    integrator_out += K_i * e_D[n]
    e_F.append(K_p * e_D[n] + integrator_out)

    # NCO
    try:
        phase_estimate[n + 1] = phase_estimate[n] + K_0 * e_F[n]
    except IndexError:
        phase_estimate[n + 1] = K_0 * e_F[n]

    sin_out[n + 1] = -np.sin(2 * np.pi * (k / N) * (n + 1) + phase_estimate[n])
    cos_out[n + 1] = np.cos(2 * np.pi * (k / N) * (n + 1) + phase_estimate[n])
    out_signal[n + 1] = np.complex(cos_out[n + 1], sin_out[n + 1])
phase_estimate = phase_estimate * 180 / np.pi

# Create a Figure
fig = plt.figure()

# Set up Axes
ax1 = fig.add_subplot(211)
ax1.plot(e_D, label='e_D')
plt.grid()
ax1.plot(e_F, label='e_F')
plt.legend()
ax1.set_title('Waveforms')

# Show the plot
# plt.show()

ax2 = fig.add_subplot(212)
ax2.plot(phase_estimate)
# ax2.plot(e_F)
plt.grid()
ax2.set_title('Filtered Error')
plt.show()

plt.plot(rx_signal_x.real, rx_signal_x.imag, 'g+', label='Rx signal')
plt.plot(bb.real, bb.imag, 'y+', label='after synch')
plt.plot(rx_vol.real, rx_vol.imag, 'r+', label='after VOL')
plt.plot(rx_pll.real, rx_pll.imag, 'b*', label='after PLL')
plt.plot(tx_signal_x.real, tx_signal_x.imag, 'ko', label='Tx target')
plt.legend()
plt.title('100G QPSK constellation')
# %%-----------------------CMA test

from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt


def datanromalize(origin_data, PAM_order):
    mean_factor = np.mean(origin_data)
    shift_data = origin_data - mean_factor
    amplitude_factor = np.mean(np.abs(shift_data)) * 2 / PAM_order;
    normalize_data = shift_data / amplitude_factor
    return normalize_data  # ,mean_factor,amplitude_factor


T = 15000;
dB = 25  # 35;
N = 50;  # smoothing length N+1
Lh = 5;  # channel length = Lh+1
P = round((N + Lh) / 2) + 1;  # equalization delay
h = np.array(
    [0.0545 + 1j * 0.05, 0.2832 - 0.1197 * 1j, -0.7676 + 0.2788 * 1j, -0.0641 - 0.0576 * 1j, 0.0566 - 0.2275 * 1j,
     0.4063 - 0.0739 * 1j]);
# h=randn(1,Lh+1)+sqrt(-1)*randn(1,Lh+1);   % channel (complex)
h = h / LA.norm(h)

# tx=np.around(np.random.rand(T))*2-1             #QAM symbol
# tx=tx+1j*(np.around(np.random.rand(T))*2-1)     #QAM symbol
# tx_xi=np.genfromtxt(r'C:\Users\Lenovo\Desktop\sr.txt')
# tx_xq=np.genfromtxt(r'C:\Users\Lenovo\Desktop\si.txt')
# rx_xi=np.genfromtxt(r'C:\Users\Lenovo\Desktop\xr.txt')
# rx_xq=np.genfromtxt(r'C:\Users\Lenovo\Desktop\xi.txt')
# tx=tx_xi+1j*tx_xq
# sig=rx_xi+1j*rx_xq

tx = ((np.around(np.random.rand(T)) + 2 * np.around(np.random.rand(T))) - 1.5) * 2  # 16QAM symbol
tx = tx + 1j * (((np.around(np.random.rand(T)) + 2 * np.around(np.random.rand(T))) - 1.5) * 2)  # 16QAM symbol

sig = signal.lfilter(h, 1, tx)
vn = np.random.rand(T) + 1j * np.random.rand(T)
vn = vn / LA.norm(vn) * 10 ** (-dB / 20) * LA.norm(sig)
SNR = 20 * np.log10(LA.norm(sig) / LA.norm(vn))
sig = sig + vn
# sig=datanromalize(sig,4)
Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
x0 = np.hstack((np.zeros(1), sig))
for i in range(Lp):
    X[:, i] = (x0[i + N + 1:i:-1].T)

e = np.zeros(Lp, dtype=complex);  # used to save instant error
f = np.zeros(N + 1, dtype=complex);
f[P] = 1;  # initial condition
# %%-----------------------adaptive blind equlizer(4QAM)
R2 = 2  # np.sqrt(2);                  # constant modulas of QPSK symbols
mu = 0.001  # 0.000271;      # parameter to adjust convergence and steady error 16QAM is samller than QAM
for k in range(50):
    for i in range(Lp):
        y = np.dot(np.conj(f.T), X[:, i])  # update y
        # yr=y.real-4*np.sign(y.real)-2*np.sign(y.real-4*np.sign(y.real))     #modified CMA
        # yi=y.imag-4*np.sign(y.imag)-2*np.sign(y.imag-4*np.sign(y.imag))     #modified CMA
        # e[i]=(abs(yr)**2-R2)*yr+1j*(abs(yi)**2-R2)*yi           #modified CMA
        # e[i]=y*(abs(y)**2-R2)                  # original instant error, R=2
        e[i] = y.real * (abs(y.real) ** 2 - R2) + 1j * y.imag * (abs(y.imag) ** 2 - R2)  # original instant error, R=1
        f = f - mu * np.conj(e[i].T) * X[:, i]  # original update equalizer coefficiency
        # f[P]=1
for k in range(50):
    for i in range(Lp):
        y = np.dot(np.conj(f.T), X[:, i])  # update y
        # yr=y.real-4*np.sign(y.real)-2*np.sign(y.real-4*np.sign(y.real))     #modified CMA
        # yi=y.imag-4*np.sign(y.imag)-2*np.sign(y.imag-4*np.sign(y.imag))     #modified CMA
        # e[i]=(abs(yr)**2-R2)*yr+1j*(abs(yi)**2-R2)*yi           #modified CMA
        e[i] = (abs(y) ** 2 - R2)  # original instant error, R=2
        # e[i]=y.real*(abs(y.real)**2-R2)+1j*y.imag*(abs(y.imag)**2-R2)     # original instant error, R=1
        # f=f-mu*np.conj(e[i].T)*X[:,i]     # original update equalizer coefficiency
        f = f - mu * 2 * e[i] * X[:, i] * np.dot(np.conj(X[:, i].T), f)
        f[P] = 1

sb = 1 * np.dot(np.conj(f.T), X)
plt.plot(sb[0:Lp - 1].real, sb[0:Lp - 1].imag, '.', tx.real, tx.imag, 'o')
# %%-----------------------adaptive blind equlizer(16QAM) CMA adjoint MCMA
R = [2, 10, 18];
R2 = 10
mu = 0.00000273  # 0.000000323;

for j in range(150):
    for i in range(Lp - 1):
        y = np.dot(f.T, X[:, i])  # update y
        index = np.argmin([abs(abs(y) ** 2 - R[0]), abs(abs(y) ** 2 - R[1]), abs(abs(y) ** 2 - R[2])])  # M-QAM used
        # e[i]=y*(abs(y)**2-R[index])     #CMA cost function
        e[i] = y.real * (abs(y.real) ** 2 - R[index]) + 1j * y.imag * (
                abs(y.imag) ** 2 - R[index])  # MCMA cost function
        f = f - mu * np.conj(X[:, i].T) * e[i]  # original update equalizer coefficiency
        # f[P]=1
for j in range(150):
    for i in range(Lp - 1):
        y = np.dot(f.T, X[:, i])  # update y
        index = np.argmin([abs(abs(y) ** 2 - R[0]), abs(abs(y) ** 2 - R[1]), abs(abs(y) ** 2 - R[2])])  # M-QAM used
        e[i] = y * (abs(y) ** 2 - R[index])  # CMA cost function
        # e[i]=y.real*(abs(y.real)**2-R[index])+1j*y.imag*(abs(y.imag)**2-R[index])     # MCMA cost function
        f = f - mu * np.conj(X[:, i].T) * e[i]  # original update equalizer coefficiency
        # f[P]=1
    # sb=0.75*np.dot(np.conj(f.T),X)
    # print(EVM(tx[30:],sb))
sb = np.dot(f.T, X)
plt.plot(sb[0:Lp - 1].real, sb[0:Lp - 1].imag, '.', tx.real, tx.imag, 'o')
# %%-----------------------adaptive blind equlizer(16QAM)  Modified CMA
R = [2, 10, 18];
R2 = 2
mu = 0.0000273  # 0.000000323;

for k in range(150):
    for i in range(Lp - 1):
        y = np.dot(np.conj(f.T), X[:, i])  # update y
        # index=np.argmin([abs(abs(y)**2-R[0]),abs(abs(y)**2-R[1]),abs(abs(y)**2-R[2])])
        e[i] = (abs((y.real - 2 * np.sign(y.real)) + 1j * (y.imag - 2 * np.sign(y.imag))) ** 2 - R2) * (
                    y - 2 * np.sign(y.real) - 1j * 2 * np.sign(y.imag))  # MCMA cost function R=R2 or R[index] for test
        f = f - mu * np.conj(e[i].T) * X[:, i]  # original update equalizer coefficiency
        # f[P]=1
    # sb=0.75*np.dot(np.conj(f.T),X)
    # print(EVM(tx[30:],sb))
sb = np.dot(np.conj(f.T), X)
plt.plot(sb[0:Lp - 1].real, sb[0:Lp - 1].imag, '.', tx.real, tx.imag, 'o')
# %%-----------------------adaptive blind equlizer(16QAM) Directed Discion + Modified CMA
R = [2, 10, 18];
symbol = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j, 1 + 3j, 1 - 3j, -1 + 3j, -1 - 3j, 3 + 1j, 3 - 1j, -3 + 1j, -3 - 1j, 3 + 3j,
          3 - 3j, -3 + 3j, -3 - 3j]
muc = 0.00000273
mud = 0.00000273
fc = np.zeros(N + 1, dtype=complex);
fc[P] = 1;
ec = np.zeros(Lp, dtype=complex);
fd = np.zeros(N + 1, dtype=complex);  # fd[P]=1;
ed = np.zeros(Lp, dtype=complex);
sdd = np.zeros(16, dtype=complex)  # y to symbol box

for k in range(70):
    for i in range(Lp - 1):
        y = np.dot(fc.T, X[:, i]) + np.dot(fd.T, X[:, i])  # update y
        index = np.argmin([abs(abs(y) ** 2 - R[0]), abs(abs(y) ** 2 - R[1]), abs(abs(y) ** 2 - R[2])])
        for l in range(16):
            sdd[l] = y - symbol[l]
        s1 = symbol[np.argmin(abs(sdd))]
        ec[i] = (abs((y.real - 2 * np.sign(y.real)) + 1j * (y.imag - 2 * np.sign(y.imag))) ** 2 - R[index]) * (
                y - 2 * np.sign(y.real) - 1j * 2 * np.sign(y.imag))  # MCMA cost function
        fc = fc - muc * np.conj(X[:, i].T) * ec[i]  # original update equalizer coefficiency
        y_adj = np.dot(fc.T, X[:, i]) + np.dot(fd.T, X[:, i])
        for l in range(16):
            sdd[l] = y_adj - symbol[l]
        s2 = symbol[np.argmin(abs(sdd))]
        if k > 50:
            if s1 == s2:
                ed[i] = abs(y - s1)
                fd = fd - mud * np.conj(X[:, i].T) * ed[i]
            else:
                fd = fd
        # f[P]=1
    # sb=0.75*np.dot(np.conj(f.T),X)
    # print(EVM(tx[30:],sb))
sb = np.dot(fc.T, X) + np.dot(fd.T, X) * 3 / 5
plt.plot(sb[0:Lp - 1].real, sb[0:Lp - 1].imag, '.', tx.real, tx.imag, 'o')
# x=sb[0:Lp-1].real#[10000:16000]#rx_signal_x.real
# y=sb[0:Lp-1].imag#[10000:16000]#rx_signal_x.imag

# plt.hist2d(x,y, bins=65, range=np.array([(min(x)-0.1, max(x)+0.1), (min(y)-0.1, max(y)+0.1)]), cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()
# %%-------------MIMO CMA----------
from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt

T = 15000;
dB = 30;
N = 20;  # smoothing length N+1
Lh = 5;  # channel length = Lh+1
P = round((N + Lh) / 2);  # equalization delay
h_x = np.array(
    [0.0545 + 1j * 0.05, 0.2832 - 0.1197 * 1j, -0.7676 + 0.2788 * 1j, -0.0641 - 0.0576 * 1j, 0.0566 - 0.2275 * 1j,
     0.4063 - 0.0739 * 1j]);
# h=randn(1,Lh+1)+sqrt(-1)*randn(1,Lh+1);   % channel (complex)
# h_x=np.random.rand(Lh+1)+1j*np.random.rand(Lh+1);   # channel (complex)
h_x = h_x / LA.norm(h_x)
h_y = h_x

tx_x = np.around(np.random.rand(T)) * 2 - 1  # QAM symbol
tx_x = tx_x + 1j * (np.around(np.random.rand(T)) * 2 - 1)  # QAM symbol
tx_y = np.around(np.random.rand(T)) * 2 - 1  # QAM symbol
tx_y = tx_y + 1j * (np.around(np.random.rand(T)) * 2 - 1)  # QAM symbol

# tx=((np.around(np.random.rand(T))+2*np.around(np.random.rand(T)))-1.5)*2             #16QAM symbol
# tx=tx+1j*(((np.around(np.random.rand(T))+2*np.around(np.random.rand(T)))-1.5)*2)     #16QAM symbol

sig_x = signal.lfilter(h_x, 1, tx_x)
sig_y = signal.lfilter(h_y, 1, tx_y)
vn_x = np.random.rand(T) + 1j * np.random.rand(T)
vn_y = np.random.rand(T) + 1j * np.random.rand(T)
vn_x = vn_x / LA.norm(vn_x) * 10 ** (-dB / 20) * LA.norm(sig_x)
vn_y = vn_y / LA.norm(vn_y) * 10 ** (-dB / 20) * LA.norm(sig_y)
SNR = 20 * np.log10(LA.norm(sig_x) / LA.norm(vn_x))
sig_x = sig_x + vn_x
sig_y = sig_y + vn_y

Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
X_x = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
x0_x = np.hstack((np.zeros(1), sig_x))
X_y = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
x0_y = np.hstack((np.zeros(1), sig_y))
for i in range(Lp - 1):
    X_x[:, i] = np.conj(x0_x[i + N + 1:i:-1].T)
    X_y[:, i] = np.conj(x0_y[i + N + 1:i:-1].T)

e_x = np.zeros(Lp, dtype=complex);  # used to save instant error
e_y = np.zeros(Lp, dtype=complex)
f_xx = np.zeros(N + 1, dtype=complex);
f_xx[P] = 1;  # initial condition
f_xy = np.zeros(N + 1, dtype=complex);
f_xy[P] = 1;  # initial condition
f_yx = np.zeros(N + 1, dtype=complex);
f_yx[P] = 1;  # initial condition
f_yy = np.zeros(N + 1, dtype=complex);
f_yy[P] = 1;  # initial condition

R2 = 2  # np.sqrt(2);                  # constant modulas of QPSK symbols
mu = 0.000271;  # parameter to adjust convergence and steady error 16QAM is samller than QAM

for k in range(50):
    for i in range(Lp - 1):
        y_x = np.dot(np.conj(f_xx.T), X_x[:, i]) + np.dot(np.conj(f_xy.T), X_y[:, i])  # update y
        y_y = np.dot(np.conj(f_yx.T), X_x[:, i]) + np.dot(np.conj(f_yy.T), X_y[:, i])
        e_x[i] = y_x * (abs(y_x) ** 2 - R2)  # original instant error, R=2
        e_y[i] = y_y * (abs(y_y) ** 2 - R2)
        # e[i]=y.real*(abs(y.real)**2-R2)+1j*y.imag*(abs(y.imag)**2-R2)     # original instant error, R=1
        f_xx = f_xx - mu * np.conj(e_x[i].T) * X_x[:, i]  # original update equalizer coefficiency
        f_xy = f_xy - mu * np.conj(e_x[i].T) * X_y[:, i]
        f_yx = f_yx - mu * np.conj(e_y[i].T) * X_x[:, i]
        f_yy = f_yy - mu * np.conj(e_y[i].T) * X_y[:, i]
        # f_xx[P]=1; f_xy[P]=1; f_yx[P]=1; f_yy[P]=1
sb_x = np.dot(np.conj(f_xx.T), X_x) + np.dot(np.conj(f_xy.T), X_y)
sb_y = np.dot(np.conj(f_yx.T), X_x) + np.dot(np.conj(f_yy.T), X_y)
plt.plot(sb_x[0:Lp - 1].real, sb_x[0:Lp - 1].imag, '.', tx_x.real, tx_x.imag, 'o')
# %%-----------------central FIR CMA------------------
from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt

T = 25000  # 15000;
dB = 20  # 35;
tap = 20;  # smoothing length N+1
Lh = 5;  # channel length = Lh+1
c = tap + 1  # P=round((N+Lh)/2)+1;  # equalization delay
memory = 2 * tap + 1
h = np.array(
    [0.0545 + 1j * 0.05, 0.2832 - 0.1197 * 1j, -0.7676 + 0.2788 * 1j, -0.0641 - 0.0576 * 1j, 0.0566 - 0.2275 * 1j,
     0.4063 - 0.0739 * 1j]);
# h=randn(1,Lh+1)+sqrt(-1)*randn(1,Lh+1);   % channel (complex)
h = h / LA.norm(h)
tx = np.around(np.random.rand(T)) * 2 - 1  # QAM symbol
tx = tx + 1j * (np.around(np.random.rand(T)) * 2 - 1)  # QAM symbol
sig = signal.lfilter(h, 1, tx)
vn = np.random.rand(T) + 1j * np.random.rand(T)
vn = vn / LA.norm(vn) * 10 ** (-dB / 20) * LA.norm(sig)
SNR = 20 * np.log10(LA.norm(sig) / LA.norm(vn))
sig = sig + vn
X = np.zeros([memory, len(sig) - 2 * tap], dtype=complex);  # sample vectors (each column is a sample vector)
for i in range(len(sig) - 2 * tap):
    X[:, i] = sig[i:i + memory].T
e = np.zeros(len(sig) - 2 * tap, dtype=complex);  # used to save instant error
f = np.zeros(memory, dtype=complex);
f[c - 1] = 1;  # initial condition
R2 = 2  # np.sqrt(2);                  # constant modulas of QPSK symbols
mu = 0.00065
for i in range(int((len(sig) - 2 * tap) * 0.07)):
    y = np.dot(np.conj(f), X[:, i])  # update y
    e[i] = (abs(y) ** 2 - R2)  # original instant error, R=2
    f = f - mu * 2 * e[i] * X[:, i] * np.dot(np.conj(X[:, i]), f)
    f[c - 1] = 1
sb = 1 * np.dot(np.conj(f), X)
plt.plot(sb.real, sb.imag, '.', tx.real, tx.imag, 'o')
plt.show()
phasecorrection, rx_pll, p = PLL_(sb, 0.00351,
                                  0.707)  # phase correct after blind EQ maybe correlation is -rx & soft decision used rx_pll
H = np.zeros([memory, memory + Lh], dtype=complex);
for i in range(memory):
    H[i, i:i + Lh + 1] = h  # channel matrix
fh = np.dot(np.conj(f), H)  # composite channel+equalizer response should be delta-like
temp = np.argmax(abs(fh));  # find the max of the composite response
sb1 = sb / fh[temp];  # scale the output
sb1 = np.sign(sb1.real) + 1j * np.sign(sb1.imag);  # signal soft decision
for i in range(tap + 1):
    start = i;  # carefully find the corresponding begining point
    sb2 = sb1 - tx[start:start + len(sb1)];  # find error symbols
    BERcount = 0
    location = []
    for i in range(len(sb2)):
        if sb2[i] != 0:
            BERcount = BERcount + 1
            location.append(i)
    print('error count=', BERcount, '\n', 'BER count', len(location) / len(sb2))
pos_y = []
neg_y = []
for i in range(len(rx_pll) - 1500):
    pos_y.append(np.corrcoef(tx[i:i + 1500], rx_pll[0:1500])[0, 1])
    neg_y.append(np.corrcoef(tx[0:1500], rx_pll[i:i + 1500])[0, 1])
if max(pos_y) > max(neg_y):
    shift1 = np.argmax(pos_y);
    value1 = max(pos_y)
else:
    shift1 = np.argmax(neg_y);
    value1 = max(neg_y)
rpos_y = []
rneg_y = []
for i in range(len(rx_pll) - 1500):
    rpos_y.append(np.corrcoef(tx[i:i + 1500], -rx_pll[0:1500])[0, 1])
    rneg_y.append(np.corrcoef(tx[0:1500], -rx_pll[i:i + 1500])[0, 1])
if max(rpos_y) > max(rneg_y):
    shift2 = np.argmax(rpos_y);
    value2 = max(rpos_y)
else:
    shift2 = np.argmax(rneg_y);
    value2 = max(rneg_y)

if value1 > value2:
    shift = shift1;
    value = value1
else:
    shift = shift2;
    value = value2
print(value, shift)
train_data, F, w, R_test, rx_vol, test_data = vol_c(1500, 33, tx[shift:shift + len(rx_pll)], rx_pll, 1)
plt.plot(rx_vol.real, rx_vol.imag, '.')
BER_count(test_data, rx_vol, 2);
# SER=length(find(sb2~=0))/length(sb2)   # calculate SER
# %%-----------------original CMA SER------------------
from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt

T = 25000;
dB = 15  # 35;
N = 20;  # smoothing length N+1
Lh = 5;  # channel length = Lh+1
P = round((N + Lh) / 2) + 1;  # equalization delay
h = np.array(
    [0.0545 + 1j * 0.05, 0.2832 - 0.1197 * 1j, -0.7676 + 0.2788 * 1j, -0.0641 - 0.0576 * 1j, 0.0566 - 0.2275 * 1j,
     0.4063 - 0.0739 * 1j]);
# h=randn(1,Lh+1)+sqrt(-1)*randn(1,Lh+1);   % channel (complex)
h = h / LA.norm(h)
tx = np.around(np.random.rand(T)) * 2 - 1  # QAM symbol
tx = tx + 1j * (np.around(np.random.rand(T)) * 2 - 1)  # QAM symbol
sig = signal.lfilter(h, 1, tx)
vn = np.random.rand(T) + 1j * np.random.rand(T)
vn = vn / LA.norm(vn) * 10 ** (-dB / 20) * LA.norm(sig)
SNR = 20 * np.log10(LA.norm(sig) / LA.norm(vn))
sig = sig + vn
sig = datanromalize(sig, 2)
sig0 = np.hstack((np.zeros(N), sig))
# tx_xi=np.genfromtxt(r'D:\新增資料夾\sr.txt')
# tx_xq=np.genfromtxt(r'D:\新增資料夾\si.txt')
# rx_xi=np.genfromtxt(r'D:\新增資料夾\xr.txt')
# rx_xq=np.genfromtxt(r'D:\新增資料夾\xi.txt')
# tx=tx_xi+1j*tx_xq
# sig=rx_xi+1j*rx_xq
Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
for i in range(Lp):
    a = (np.flipud(sig[i:i + N + 1]).T)
    # print(i,np.size(a))
    X[:, i] = a
e = np.zeros(Lp, dtype=complex);  # used to save instant error
f = np.zeros(N + 1, dtype=complex);
f[P] = 1;  # initial condition
R2 = 2  # np.sqrt(2);                  # constant modulas of QPSK symbols
mu = 0.001
for k in range(1):
    # print(f[0])
    for i in range(int(0.07 * T)):
        y = np.dot(np.conj(f), X[:, i])  # update y
        e[i] = (abs(y) ** 2 - R2)  # original instant error, R=2
        f = f - mu * 2 * e[i] * X[:, i] * np.dot(np.conj(X[:, i]), f)
        # f[P]=1
sb = 1 * np.dot(np.conj(f), X)
plt.plot(sb[0:Lp - 1].real, sb[0:Lp - 1].imag, '.', tx.real, tx.imag, 'o');
plt.show()
phasecorrection, rx_pll, p = PLL_(sb, 0.001,
                                  0.707)  # phase correct after blind EQ maybe correlation is -rx & soft decision used rx_pll
H = np.zeros([N + 1, N + Lh + 1], dtype=complex);
for i in range(N + 1):
    H[i, i:i + Lh + 1] = h  # channel matrix
fh = np.dot(np.conj(f), H)  # composite channel+equalizer response should be delta-like
temp = np.argmax(abs(fh));  # find the max of the composite response
sb1 = sb / fh[temp];  # scale the output
sbm = sb1
plt.plot(sbm.real, sbm.imag, '.');
plt.show()
sb1 = np.sign(sb1.real) + 1j * np.sign(sb1.imag);  # soft decision symbol detection
start = N - temp;  # carefully find the corresponding begining point
sb2 = sb1 - tx[start:start + len(sb1)];  # find error symbols
BERcount = 0
index = []
for i in range(len(sb2)):
    if sb2[i] != 0:
        BERcount = BERcount + 1
        index.append(i)
print(BERcount, '\n', len(index) / len(sb2))
# SER=length(find(sb2~=0))/length(sb2)   # calculate SER
BER_count(tx[start:start + len(rx_pll)], -rx_pll, 2);
# [mm,ss,aa,bb,yy]=data_synch(tx[5:5+len(rx_pll)],rx_pll,11)
pos_y = []
neg_y = []
for i in range(len(rx_pll) - 1500):
    pos_y.append(np.corrcoef(tx[i:i + 1500], rx_pll[0:1500])[0, 1])
    neg_y.append(np.corrcoef(tx[0:1500], rx_pll[i:i + 1500])[0, 1])
if max(pos_y) > max(neg_y):
    shift1 = np.argmax(pos_y);
    value1 = max(pos_y)
else:
    shift1 = np.argmax(neg_y);
    value1 = max(neg_y)
rpos_y = []
rneg_y = []
for i in range(len(rx_pll) - 1500):
    rpos_y.append(np.corrcoef(tx[i:i + 1500], -rx_pll[0:1500])[0, 1])
    rneg_y.append(np.corrcoef(tx[0:1500], -rx_pll[i:i + 1500])[0, 1])
if max(rpos_y) > max(rneg_y):
    shift2 = np.argmax(rpos_y);
    value2 = max(rpos_y)
else:
    shift2 = np.argmax(rneg_y);
    value2 = max(rneg_y)

if value1 > value2:
    shift = shift1;
    value = value1
else:
    shift = shift2;
    value = value2
print('max corr is', value.real, ',', 'corresponding index is', shift)
train_data, F, w, R_test, rx_vol, test_data = vol_c(1500, 39, tx[shift:shift + len(rx_pll)], rx_pll, 1)
plt.plot(rx_vol.real, rx_vol.imag, '.')
BER_count(test_data, rx_vol, 2);

# [mm,ss,aa,bb,yy]=data_synch(rx_pll,tx,11)
# train_data,F,w,R_test,rx_vol,test_data=vol_c(1500,39,bb,aa,1)
# plt.plot(rx_vol.real,rx_vol.imag,'.')

# [mm,ss,aa,bb,yy]=data_synch(tx,sb,11)
# train_data,F,w,R_test,rx_vol,test_data=vol_c(1500,39,aa,bb,1)
# plt.plot(rx_vol.real,rx_vol.imag,'.')
# %%-----------------original CMA SER 2------------------
from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt


def datanromalize(origin_data, PAM_order):
    mean_factor = np.mean(origin_data)
    shift_data = origin_data - mean_factor
    amplitude_factor = np.mean(np.abs(shift_data)) * 2 / PAM_order;
    normalize_data = shift_data / amplitude_factor
    return normalize_data  # ,mean_factor,amplitude_factor


T = 25000  # 15000;
dB = 15  # 35;
N = 27;  # smoothing length N+1
Lh = 5;  # channel length = Lh+1
P = round((N + Lh) / 2);  # equalization delay
h = np.array(
    [0.0545 + 1j * 0.05, 0.2832 - 0.1197 * 1j, -0.7676 + 0.2788 * 1j, -0.0641 - 0.0576 * 1j, 0.0566 - 0.2275 * 1j,
     0.4063 - 0.0739 * 1j]);
Lh = 5;
# h=np.array([-0.005-0.004*1j,-0.009-0.3*1j,-0.0024-0.104*1j,0.864+0.52*1j,-0.218+0.273*1j,0.049-0.074*1j,0.016+0.2*1j]); Lh=6;
# h=np.array([1,0.1294+0.483*1j]); Lh=1;
# h=np.array([0.9063+0.4226*1j,0.3214+0.3830*1j]); Lh=1;
# h=np.random.randn(Lh+1)+1j*np.random.randn(Lh+1)
h = h / LA.norm(h)
tx = np.around(np.random.rand(T)) * 2 - 1  # QAM symbol
tx = tx + 1j * (np.around(np.random.rand(T)) * 2 - 1)  # QAM symbol
# tx=((np.around(np.random.rand(T))+2*np.around(np.random.rand(T)))-1.5)*2             #16QAM symbol
# tx=tx+1j*(((np.around(np.random.rand(T))+2*np.around(np.random.rand(T)))-1.5)*2)     #16QAM symbol
QAMorder = 4
sig = signal.lfilter(h, 1, tx)
vn = np.random.rand(T) + 1j * np.random.rand(T)
vn = vn / LA.norm(vn) * 10 ** (-dB / 20) * LA.norm(sig)
SNR = 20 * np.log10(LA.norm(sig) / LA.norm(vn))
sig = sig + vn
sig = datanromalize(sig, 2)
# sig0=np.hstack((np.zeros(N),sig))
# tx_xi=np.genfromtxt(r'D:\新增資料夾\sr.txt')
# tx_xq=np.genfromtxt(r'D:\新增資料夾\si.txt')
# rx_xi=np.genfromtxt(r'D:\新增資料夾\xr.txt')
# rx_xq=np.genfromtxt(r'D:\新增資料夾\xi.txt')
# tx=tx_xi+1j*tx_xq
# sig=rx_xi+1j*rx_xq

Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
for i in range(Lp):
    X[:, i] = (np.flipud(sig[i:i + N + 1]).T)
e = np.zeros(Lp, dtype=complex);  # used to save instant error
f = np.zeros(N + 1, dtype=complex);
f[P] = 1;  # initial condition
if QAMorder == 4:
    R2 = 2  # np.sqrt(2);                  # constant modulas of QPSK symbols
    mu = 0.000237  # 0.001
    for k in range(1):
        for i in range(Lp):
            y = np.dot(np.conj(f), X[:, i])  # update y
            e[i] = (abs(y) ** 2 - R2)  # *np.dot(np.conj(X[:,i]),f)     # original instant error, R=2
            # e[i]=(abs(y.real)**2-R2)*(np.dot(np.conj(X[:,i]),f)).real+1j*((abs(y.imag)**2-R2)*(np.dot(np.conj(X[:,i]),f)).imag)     # MCMA
            f = f - mu * 2 * e[i] * X[:, i] * np.dot(np.conj(X[:, i]), f)  # original weight update
            # f[P]=1
elif QAMorder == 16:
    R2 = 10
    mu = 0.0005363  # 0.000000323;
    for k in range(30):
        for i in range(Lp - 1):
            y = np.dot(np.conj(f.T), X[:, i])  # update y
            # index=np.argmin([abs(abs(y)**2-R[0]),abs(abs(y)**2-R[1]),abs(abs(y)**2-R[2])])
            e[i] = (abs((y.real - 2 * np.sign(y.real)) + 1j * (y.imag - 2 * np.sign(y.imag))) ** 2 - R2) * (
                        y - 2 * np.sign(y.real) - 1j * 2 * np.sign(y.imag))  # MCMA cost function
            f = f - mu * np.conj(e[i].T) * X[:, i]  # original update equalizer coefficiency
sb = 1 * np.dot(np.conj(f), X)
plt.plot(sb[0:Lp - 1].real, sb[0:Lp - 1].imag, '.', tx.real, tx.imag, 'o');
plt.show()
phasecorrection, rx_pll, p = PLL_(sb, 0.00351,
                                  0.707)  # phase correct after blind EQ maybe correlation is -rx & soft decision used rx_pll

H = np.zeros([N + 1, N + Lh + 1], dtype=complex);
for i in range(N + 1):
    H[i, i:i + Lh + 1] = h  # channel matrix
fh = np.dot(np.conj(f), H)  # composite channel+equalizer response should be delta-like
temp = np.argmax(abs(fh));  # find the max of the composite response
sb1 = sb / fh[temp];  # scale the output
sbm = sb1
# plt.plot(sbm.real,sbm.imag,'.');plt.show()
sb1 = np.sign(sb1.real) + 1j * np.sign(sb1.imag);  # soft decision symbol detection
start = N - temp;  # carefully find the corresponding begining point
sb2 = sb1 - tx[start:start + len(sb1)];  # find error symbols
BERcount = 0
index = []
for i in range(len(sb2)):
    if sb2[i] != 0:
        BERcount = BERcount + 1
        index.append(i)
print(BERcount, '\n', len(index) / len(sb2))
# BER_count(tx[start:start+len(sb1)],sb1,PAM_order=2)
# SER=length(find(sb2~=0))/length(sb2)   # calculate SER
for i in range(len(tx) - len(rx_pll)):
    BER_count(tx[i:i + len(rx_pll)], -rx_pll, PAM_order=2)
    print(np.corrcoef(tx[i:i + len(rx_pll)], -rx_pll)[0, 1])

# [mm,ss,aa,bb,yy]=data_synch(sb,tx)
[mm, ss, aa, bb, yy] = data_synch(rx_pll, tx, 13)
train_data, F, w, R_test, rx_vol, test_data = vol_c(1750, 27, bb, aa, 1)
# plt.plot(rx_vol.real,rx_vol.imag,'.')
BER_count(test_data, rx_vol, 2);
# %%-------------time domain fiber dispersion FIR------------------
D = 16  # ps/nm/km
L = 80  # km
lemda = 1.55e-6
Baudrate = 53e9
Ts = 1 / Baudrate
T = Ts / 2
c = 2.997e8
N = 2 * (math.ceil((D * L * 1e-3 * lemda * lemda / (2 * c * T * T))) - 1)  # tap number
W = np.zeros(N + 1, dtype=complex)
for i in np.linspace(-N / 2, N / 2, N + 1):
    k = int(i)
    W[k] = np.sqrt(1j * c * Ts * Ts / (D * lemda * lemda * L)) * np.exp(
        -1j * k * k * (np.pi * c * Ts * Ts / (D * lemda * lemda * L)))
    # print(k,W[k])
plt.plot(np.linspace(-N / 2, N / 2, N + 1), W.real, '+', np.linspace(-N / 2, N / 2, N + 1), W.imag, '+')
rx_signal_x = rx_xi + 1j * rx_xq
zp_sig = np.hstack((np.zeros(int(N / 2)), rx_signal_x, np.zeros(int(N / 2))))
rxcdc = np.convolve(W, zp_sig)
X = np.zeros([len(W), len(rx_signal_x)], dtype=complex)
X_i = X.real;
X_q = X.imag;
for i in range(len(rx_signal_x)):
    # X_i[:,i]=zp_sig.real[i:i+len(W)].T
    # X_q[:,i]=zp_sig.imag[i:i+len(W)].T
    X[:, i] = zp_sig[i:i + len(W)].T
rx_cdc = np.dot(W, X)
# for i in range(N+1):
#   k=int(i)
#  W[k]=np.sqrt(1j*c*Ts*Ts/(D*lemda*lemda*L))*np.exp(-1j*k*k*(np.pi*c*Ts*Ts/(D*lemda*lemda*L)))
# print(k,W[k])
# plt.plot(range(N+1),W.real,'+',range(N+1),W.imag,'+');plt.show()
# %%------------Coarse Frequency test--------------
rx_xi = np.genfromtxt(r'C:\Users\user\Desktop\rx_xi.txt')
rx_xq = np.genfromtxt(r'C:\Users\user\Desktop\rx_xq.txt')
rx = rx_xi + 1j * rx_xq
plt.plot(rx_xi, rx_xq, '+');
plt.show();
fr = 1e3;
fs = 1e9;
m_order = 4;  # if formate is QAM, morder is 4
Nfft = 2 ** int(np.floor(np.log2(fs / fr)));
# if Nfft>1e7:
#    Nfft=2**23   # avoid memory error
# elif Nfft<1e3:
#    Nfft=2**10
N = len(rx);
# raiseSig=rx**m_order
absFFTSig = abs(np.fft.fft(rx ** m_order, Nfft))
plt.plot(absFFTSig / max(absFFTSig));
plt.xlabel('index');
plt.ylabel('magnitude');
plt.show();
maxIndex = np.argmax(absFFTSig)
if maxIndex > Nfft / 2:
    maxIndex = maxIndex - Nfft;
    print('maxIndex>Nfft/2');
estFreqOffset = np.round(fs / Nfft * (maxIndex - 1) / m_order);
print('Frequency offset=', estFreqOffset)
rx = rx * np.exp(-estFreqOffset * np.linspace(1, N, N) * 2 * np.pi * 1j / fs)
plt.plot(rx.real, rx.imag, '+');
plt.show()
# %%-----------------

absFFTSig = abs(np.fft.fft(rx ** m_order, Nfft))
plt.plot(absFFTSig);
plt.xlabel('index');
plt.ylabel('magnitude');
plt.show();
psd = np.fft.fftshift(absFFTSig)
maxIndex = np.argmax(psd)
offsetIndex = maxIndex - Nfft / 2
estFreqOffset = fs / Nfft * (offsetIndex - 1) / m_order;
print(estFreqOffset)
rx = rx * np.exp(-estFreqOffset * np.linspace(1, N, N) * 2 * np.pi * 1j / fs)
plt.plot(rx.real, rx.imag, '.');
plt.show()
# %%---------LMS equalizer---------------
from scipy import signal
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt


def datanromalize(origin_data, PAM_order):
    mean_factor = np.mean(origin_data)
    shift_data = origin_data - mean_factor
    amplitude_factor = np.mean(np.abs(shift_data)) * 2 / PAM_order;
    normalize_data = shift_data / amplitude_factor
    return normalize_data  # ,mean_factor,amplitude_factor


T = 25000  # 15000;
dB = 25  # 35;
N = 20;  # smoothing length N+1
Lh = 5;  # channel length = Lh+1
P = round((N + Lh) / 2);
h = np.array(
    [0.0545 + 1j * 0.05, 0.2832 - 0.1197 * 1j, -0.7676 + 0.2788 * 1j, -0.0641 - 0.0576 * 1j, 0.0566 - 0.2275 * 1j,
     0.4063 - 0.0739 * 1j]);
Lh = 5;
# h=np.array([-0.005-0.004*1j,-0.009-0.3*1j,-0.0024-0.104*1j,0.864+0.52*1j,-0.218+0.273*1j,0.049-0.074*1j,0.016+0.2*1j]); Lh=6;
h = h / LA.norm(h)
tx = np.around(np.random.rand(T)) * 2 - 1  # QAM symbol
tx = tx + 1j * (np.around(np.random.rand(T)) * 2 - 1)  # QAM symbol
QAMorder = 4
sig = signal.lfilter(h, 1, tx)
vn = np.random.rand(T) + 1j * np.random.rand(T)
vn = vn / LA.norm(vn) * 10 ** (-dB / 20) * LA.norm(sig)
SNR = 20 * np.log10(LA.norm(sig) / LA.norm(vn))
sig = sig + vn
sig = datanromalize(sig, 2)
Lp = T - N;  # remove several first samples to avoid 0 or negative subscript
X = np.zeros([N + 1, Lp], dtype=complex);  # sample vectors (each column is a sample vector)
for i in range(Lp):
    X[:, i] = (np.flipud(sig[i:i + N + 1]).T)
e = np.zeros(Lp, dtype=complex);  # used to save instant error
f = np.zeros(N + 1, dtype=complex);
f[P] = 1;
mu = 1.37e-3;
f = np.zeros(N + 1, dtype=complex);  # 0.001
for k in range(50):
    for i in range(Lp):
        # xi=X[:,i].real;xq=X[:,i].imag;
        # yi=np.dot(f.real,xi)-np.dot(f.imag,xq);yq=np.dot(f.real,xq)-np.dot(f.imag,xi);
        # di=tx[i].real;dq=tx[i].imag;
        # ei=di-1j*yi;eq=dq-1j*yq;
        # fi=f.real+mu*(ei*xi-eq*xq);fq=f.imag+mu*(ei*xq-eq*xi);
        # f=fi+1j*fq
        y = np.dot(np.conj(f.T), X[:, i])  # update y
        e[i] = tx[i] - y  # *np.dot(np.conj(X[:,i]),f)     # original instant error, R=2
        f = f + 9.52e-4 * np.conj(e[i]) * X[:, i] / (LA.norm(X[:, i]) ** 2)  # NLMS
        # f[P]=1
sb = 1 * np.dot(np.conj(f), X)
plt.plot(sb[0:Lp - 1].real, sb[0:Lp - 1].imag, '.', tx.real, tx.imag, 'o');
plt.show()
[mm, ss, aa, bb, yy] = data_synch(sb, tx, 13)
LMS_BERcount = BER_count(bb, aa, 2);
error_mark(sb, LMS_BERcount[1]);
train_data, F, w, R_test, rx_vol, test_data = vol_c(1750, 9, bb, aa, 1)
BER_count(test_data, rx_vol, 2);