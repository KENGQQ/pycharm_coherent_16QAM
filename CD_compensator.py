import numpy as np
import cmath
import math


class CD_compensator:
    def __init__(self, RxX, RxY, Gbaud, KM):
        self.KM = KM
        self.Gbaud = Gbaud
        print("CD_tap_needed : {}".format(np.floor(16e-12 * self.KM * (1553 ** 2) / 2 / 3e17 / (1 / self.Gbaud) ** 2)))
        self.N = np.floor(16e-12 * self.KM * (1553 ** 2) / 2 / 3e17 / (1 / self.Gbaud) ** 2)
        self.tap = int(np.floor(16e-12 * self.KM * (1553 ** 2) / 2 / 3e17 / (1 / self.Gbaud) ** 2) * 2 + 1)
        self.center = int((self.tap - 1) / 2)
        self.rx_x_single = np.array(RxX)
        self.rx_y_single = np.array(RxY)
        self.datalength = len(RxX)


    def FIR_CD(self):
        self.ak = np.zeros(int(self.tap), dtype="complex_")
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")

        inputrx = self.rx_x_single
        inputry = self.rx_y_single

        tt = -self.N

        for k in range(0, self.tap):
            self.ak[k] = np.sqrt(1j * 3e17 * (1 / self.Gbaud) ** 2 / 16e-12 / (1553 ** 2) / self.KM) * \
                    np.exp(-1j * math.pi * 3e17 * (1 / self.Gbaud) ** 2 * tt ** 2 / 16e-12 / self.KM / (1553 ** 2))
            # print(tt)
            tt+=1

        #
        for indx in range(self.center, self.datalength - self.center):
            exout[indx] = np.matmul(self.ak, inputrx[indx - self.center : indx + self.center + 1])
            eyout[indx] = np.matmul(self.ak, inputry[indx - self.center : indx + self.center + 1])
        return exout, eyout