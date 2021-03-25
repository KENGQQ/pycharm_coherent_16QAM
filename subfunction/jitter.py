import numpy as np
import cmath
import math
from numba import jit
import datetime
import numba as nb

@jit
def jitter(rx_x, rx_y, taps, iter, mean=True):
    starttime = datetime.datetime.now()
    mean = mean
    rx_x_single = np.array(rx_x)
    rx_y_single = np.array(rx_y)
    datalength = len(rx_x)
    stepsizelist = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 6.409e-6, 1e-6, 1e-7]
    overhead = 1
    cmataps = taps
    center = int((cmataps - 1) / 2)
    iterator = iter
    earlystop = 0.0001
    stepsizeadjust = 0.95
    stepsize = stepsizelist[6]
    stepsize_x = stepsize
    stepsize_y = stepsize

    costfunx = np.zeros((1, iterator), dtype="complex_")
    costfuny = np.zeros((1, iterator), dtype="complex_")
    inputrx = rx_x_single
    inputry = rx_y_single
    hxx = np.zeros(cmataps, dtype="complex_")
    hyy = np.zeros(cmataps, dtype="complex_")
    hxx[center] = 1
    hyy[center] = 1
    exout = np.zeros(datalength, dtype="complex_")
    eyout = np.zeros(datalength, dtype="complex_")
    errx = np.zeros(datalength, dtype="complex_")
    erry = np.zeros(datalength, dtype="complex_")
    cost_x = np.zeros(datalength, dtype="complex_")
    cost_y = np.zeros(datalength, dtype="complex_")
    squ_Rx = np.zeros(datalength, dtype="complex_")
    squ_Ry = np.zeros(datalength, dtype="complex_")
    squ_Rx, squ_Ry = squ_Rx + 10, squ_Ry + 10

    if mean == True:
        squ_Rx = np.zeros(datalength, dtype="complex_")
        squ_Ry = np.zeros(datalength, dtype="complex_")
        for indx in range(center, datalength - center):
            inx = inputrx[indx - center:indx + center + 1]
            iny = inputry[indx - center:indx + center + 1]
            squ_Rx[indx] = np.mean(abs(inx) ** 4) / np.mean(abs(inx) ** 2)
            squ_Ry[indx] = np.mean(abs(iny) ** 4) / np.mean(abs(iny) ** 2)

    for it in range(iterator):
        for indx in range(center, datalength - center):
            exout[indx] = np.matmul(hxx, inputrx[indx - center:indx + center + 1])
            eyout[indx] = np.matmul(hyy, inputry[indx - center:indx + center + 1])
            if np.isnan(exout[indx]) or np.isnan(eyout[indx]):
                raise Exception("CMA Equaliser didn't converge at iterator {}".format(it))

            errx[indx] = exout[indx] * (squ_Rx[indx] - np.abs(exout[indx]) ** 2)
            erry[indx] = eyout[indx] * (squ_Ry[indx] - np.abs(eyout[indx]) ** 2)

            hxx = hxx + stepsize_x * errx[indx] * np.conj(
                inputrx[indx - center:indx + center + 1])
            hyy = hyy + stepsize_y * erry[indx] * np.conj(
                inputry[indx - center:indx + center + 1])

            cost_x[indx] = (abs(exout[indx])) ** 2 - squ_Rx[indx]
            cost_y[indx] = (abs(eyout[indx])) ** 2 - squ_Ry[indx]
        costfunx[0][it] = -1 * (np.mean(cost_x))
        costfuny[0][it] = -1 * (np.mean(cost_y))
        print('iteration = {}'.format(it))
        print(costfunx[0][it])
        print(costfuny[0][it])
        print('-------')

        if it >= 1:
            # if np.abs(self.costfunx[0][it] - self.costfunx[0][it - 1]) < self.earlystop * \
            #         self.costfunx[0][it]:
            #     print("Earlybreak at iterator {}".format(it))
            #     break
            if np.abs(costfunx[0][it] - costfunx[0][it - 1]) < earlystop:
                stepsize_x *= stepsizeadjust
                print('Stepsize_x adjust to {}'.format(stepsize_x))
            if np.abs(costfuny[0][it] - costfuny[0][it - 1]) < earlystop:
                stepsize_y *= stepsizeadjust
                print('Stepsize_y adjust to {}'.format(stepsize_y))

        # rx_x_cma = exout
        # rx_y_cma = eyout
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    return exout, eyout
