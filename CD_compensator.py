import numpy as np
import cmath
import math


class CD_compensator:
    def __init__(self, RxX, RxY, Gbaud, KM):
        self.KM = KM
        self.Gbaud = Gbaud

        self.rx_x = np.array(RxX)
        self.rx_y = np.array(RxY)
        self.datalength = len(RxX)


    def FIR_CD(self):
        c = 3e17     #nm/s
        T = 1 / self.Gbaud
        wavelength = 1553     #nm
        D = 17e-12   # s / nm /km

        N = np.floor(D * self.KM * (wavelength ** 2) / 2 / c / T ** 2)
        tap = int(N * 2 + 1)
        print("FIR_CD_tap_needed : {}".format(tap))
        center = int((tap - 1) / 2)

        self.ak = np.zeros(int(tap), dtype="complex_")
        exout = np.zeros(self.datalength, dtype="complex_")
        eyout = np.zeros(self.datalength, dtype="complex_")

        inputrx = self.rx_x
        inputry = self.rx_y

        tt = -N

        for k in range(0, tap):
            self.ak[k] = np.sqrt(1j * c * (1 / self.Gbaud) ** 2 / D / (wavelength ** 2) / self.KM) * \
                    np.exp(-1j * math.pi * c * (1 / self.Gbaud) ** 2 * tt ** 2 / D / self.KM / (wavelength ** 2))
            tt+=1
        #
        for indx in range(center, self.datalength - center):
            exout[indx] = np.matmul(self.ak, inputrx[indx - center : indx + center + 1])
            eyout[indx] = np.matmul(self.ak, inputry[indx - center : indx + center + 1])
        return exout, eyout

    def FFT_CD(self):
        Nfft = 256

        c = 3e17     #nm/s
        T = 1 / (self.Gbaud)
        q = np.linspace(int(-Nfft / 2), int(Nfft / 2) - 1, Nfft)
        w = 2 * math.pi * (1 / T) * q / Nfft    # angular freq
        wavelength = 1553    #nm
        D = 16e-12   # nm / s /km
        N = 2 * np.ceil(np.sqrt((math.pi ** 2) * (c ** 2) * (T ** 4) + 4 * (wavelength ** 4) * (D ** 2) * (self.KM ** 2)) / math.pi / c / (T ** 2) ) + 2
        N = int(N)
        print("FFT_CD_tap_needed : {}".format(N))

        step = Nfft - N
        self.wk = np.zeros(int(Nfft), dtype="complex_")

        RxX_out = np.zeros(self.datalength, dtype = "complex_")
        for k in range(1, int(self.datalength / step)):
            RxX_temp = np.zeros(self.datalength, dtype="complex_")
            FFT_RxX = self.rx_x[(k - 1) * step: k * step]
            zeropadding = np.zeros(N, dtype = "complex_")
            RxX_temp[(k - 1) * step : k * Nfft - (k - 1) * N] = np.concatenate((FFT_RxX, zeropadding), axis=None)
            # RxX_temp[(k - 1) * step : k * Nfft - (k - 1) * N] = (np.fft.fft(RxX_temp[(k - 1) * step : k * Nfft - (k - 1) * N])) \
            #                                      * np.exp(-1j * D * (wavelength ** 2) * (w ** 2) * self.KM / 4 / math.pi / c)
            RxX_temp[(k - 1) * step : k * Nfft - (k - 1) * N] = np.matmul(np.fft.fft(RxX_temp[(k - 1) * step : k * Nfft - (k - 1) * N]), \
                                                 (np.exp(-1j * D * (wavelength ** 2) * (w ** 2) * self.KM / 4 / math.pi / c)))


            RxX_temp[(k - 1) * step : k * Nfft - (k - 1) * N] = np.fft.ifft(RxX_temp[(k - 1) * step : k * Nfft - (k - 1) * N])
            RxX_out += RxX_temp


        RxY_out = np.zeros(self.datalength, dtype = "complex_")
        for k in range(1, int(self.datalength / step)):
            RxY_temp = np.zeros(self.datalength, dtype="complex_")
            FFT_RxY = self.rx_y[(k - 1) * step : k * step]
            zeropadding = np.zeros(N, dtype = "complex_")
            RxY_temp[(k - 1) * step : k * Nfft - (k - 1) * N] = np.concatenate((FFT_RxY, zeropadding), axis=None)
            RxY_temp[(k - 1) * step : k * Nfft - (k - 1) * N] = (np.fft.fft(RxY_temp[(k - 1) * step : k * Nfft - (k - 1) * N])) \
                                                 * np.exp(-1j * D * (wavelength ** 2) * (w ** 2) * self.KM / 4 / math.pi / c)
            RxY_temp[(k - 1) * step : k * Nfft - (k - 1) * N] = np.fft.ifft(RxY_temp[(k - 1) * step : k * Nfft - (k - 1) * N])
            RxY_out += RxY_temp


        return RxX_out, RxY_out


    def FFT_CD_2(self):
        Nfft = 64
        G = int(Nfft / 4)
        L = int(Nfft / 2)

        c = 3e17     #nm/s
        T = 1 / (self.Gbaud)
        q = np.linspace(int(-Nfft / 2), int(Nfft / 2) - 1, Nfft)
        w = 2 * math.pi * (1 / T) * q / Nfft    # angular freq
        wavelength = 1553    #nm
        D = 16e-12   # s / nm /km
        N = 2 * np.ceil(np.sqrt((math.pi ** 2) * (c ** 2) * (T ** 4) + 4 * (wavelength ** 4) * (D ** 2) * (self.KM ** 2)) / math.pi / c / (T ** 2) ) + 2
        N = int(N)
        print("FFT_CD_tap_needed : {}".format(N))

        filter = np.exp(-1j * D * (wavelength ** 2) * (w ** 2) * self.KM / 4 / math.pi / c)
        # filter = np.exp(-1j * D * (wavelength ** 2) * (w ** 2)  / 4 / math.pi / c)

        # FDE = np.fft.fftshift(filter)
        FDE = filter

        RxX_matrix_padding = np.zeros([int(self.datalength / L) - 1, Nfft], dtype='complex_')
        RxX_out = np.zeros(L * int(self.datalength / L), dtype="complex_")

        for k in range(0, int(self.datalength / L) - 1):
            RxX_matrix_padding[k, G : Nfft - G] = self.rx_x[k * L : (k + 1) * L]

        for k in range(0, int(self.datalength / L) - 1):
            RxX_out_tmp = np.zeros(L * int(self.datalength / L), dtype="complex_")
            RxX_out_tmp[k * L : (k + 1) * Nfft - k * 2 * G] = np.fft.ifft(np.fft.fft(RxX_matrix_padding[k, :]) * FDE)
            RxX_out += RxX_out_tmp


        RxY_matrix_padding = np.zeros([int(self.datalength / L) - 1, Nfft], dtype='complex_')
        RxY_out = np.zeros(L * int(self.datalength / L), dtype="complex_")

        for k in range(0, int(self.datalength / L) - 1):
            RxY_matrix_padding[k, G : Nfft - G] = self.rx_y[k * L : (k + 1) * L]

        for k in range(0, int(self.datalength / L) - 1):
            RxY_out_tmp = np.zeros(L * int(self.datalength / L), dtype="complex_")
            RxY_out_tmp[k * L : (k + 1) * Nfft - k * 2 * G] = np.fft.ifft(np.fft.fft(RxY_matrix_padding[k, :]) * FDE)
            RxY_out += RxY_out_tmp


        return RxX_out, RxY_out


    def FFT_CD_3(self):
        Nfft =1024
        # G = int(Nfft / 4)
        # L = int(Nfft / 2)

        c = 3e17     #nm/s
        T = 1 / (self.Gbaud)
        q = np.linspace(int(-Nfft / 2), int(Nfft / 2) - 1, Nfft)
        w = 2 * math.pi * (1 / T)    # angular freq
        wn = 2 * math.pi * (1 / T / 2)
        wavelength = 1553    #nm
        D = 16e-12   # s / nm /km
        N = 2 * np.ceil(np.sqrt((math.pi ** 2) * (c ** 2) * (T ** 4) + 4 * (wavelength ** 4) * (D ** 2) * (self.KM ** 2)) / math.pi / c / (T ** 2) ) + 2
        N = int(N)
        print("FFT_CD_tap_needed : {}".format(N))

        G = int(N)
        L = Nfft - G

        filter = np.exp(-1j * D * (wavelength ** 2) * ((q * w / Nfft) ** 2) * self.KM / 4 / math.pi / c)
        FDE = filter

        RxX_matrix_padding = np.zeros([int(self.datalength / L), Nfft], dtype='complex_')
        RxX_out = np.zeros(L * int(self.datalength / L) + 2 * G, dtype="complex_")

        RxX_matrix_padding[0, G: Nfft] = self.rx_x[0 * L: (0 + 1) * L]
        for k in range(1, int(self.datalength / L)):
            RxX_matrix_padding[k, :] = np.concatenate((self.rx_x[-G:],self.rx_x[k * L : (k + 1) * L]), axis=None)

        for k in range(0, int(self.datalength / L)):

            RxX_out_tmp = np.zeros(L * int(self.datalength / L) + 2 * G, dtype="complex_")
            RxX_out_tmp[k * L : (k + 1) * Nfft - k * G] = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(RxX_matrix_padding[k, :])) * FDE))
            RxX_out += RxX_out_tmp


        RxY_matrix_padding = np.zeros([int(self.datalength / L), Nfft], dtype='complex_')
        RxY_out = np.zeros(L * int(self.datalength / L) + 2 * G, dtype="complex_")

        RxY_matrix_padding[0, G: Nfft] = self.rx_y[0 * L: (0 + 1) * L]
        for k in range(1, int(self.datalength / L)):
            RxY_matrix_padding[k, :] = np.concatenate((self.rx_y[-G:],self.rx_y[k * L : (k + 1) * L]), axis=None)

        for k in range(0, int(self.datalength / L)):
            RxY_out_tmp = np.zeros(L * int(self.datalength / L) + 2 * G, dtype="complex_")
            RxY_out_tmp[k * L : (k + 1) * Nfft - k * G] = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(RxY_matrix_padding[k, :])) * FDE))
            RxY_out += RxY_out_tmp

        return RxX_out[G :], RxY_out[G :]


    def overlap_save(self, Nfft, NOverlap):
        c = 3e17  # nm/s
        T = 1 / (self.Gbaud)
        q = np.linspace(int(-Nfft / 2), int(Nfft / 2) - 1, Nfft)
        w = 2 * math.pi * (1 / T)  # angular freq
        wn = 2 * math.pi * (1 / T / 2)
        wavelength = 1553  # nm
        D = 16e-12  # s / nm /km
        N = 2 * np.ceil(np.sqrt((math.pi ** 2) * (c ** 2) * (T ** 4) + 4 * (wavelength ** 4) * (D ** 2) * (self.KM ** 2)) / math.pi / c / (T ** 2)) + 2
        N = int(N + N % 2)
        print("FFT_CD_tap_needed : {}".format(N))

        filter = np.exp(-1j * D * (wavelength ** 2) * ((q * w / Nfft) ** 2) * self.KM / 4 / math.pi / c)
        filter = np.exp(-1j * math.pi)
        FDE = filter

        L = int(Nfft - NOverlap)
        a = np.ceil(6.67 * abs(wavelength ** 2 * D / 2 / math.pi / c) * self.KM * (self.Gbaud ** 2))
        print(a)

        AuxLen = self.datalength / L
        if(AuxLen != np.ceil(AuxLen)):
            NExtra = int(np.ceil(AuxLen) * L - self.datalength)
            self.rx_x = np.concatenate((self.rx_x[int(np.floor(-NExtra / 2)):], self.rx_x, self.rx_x[: int(NExtra / 2)]),axis=None)
            self.rx_y = np.concatenate((self.rx_y[int(np.floor(-NExtra / 2)):], self.rx_y, self.rx_y[: int(NExtra / 2)]),axis=None)
            self.datalength = len(self.rx_x)

        Blocks_X = np.reshape(self.rx_x, [int(self.datalength / L), L])
        Blocks_Y = np.reshape(self.rx_y, [int(self.datalength / L), L])

        outX = np.zeros(np.shape(Blocks_X), dtype='complex_') ; Overlap_X = np.zeros(NOverlap, dtype='complex_')
        outY = np.zeros(np.shape(Blocks_Y), dtype='complex_') ; Overlap_Y = np.zeros(NOverlap, dtype='complex_')

        for k in range(int(self.datalength / L)):
            InB_X = np.concatenate((Overlap_X, Blocks_X[k, :]), axis=None)
            InB_Y = np.concatenate((Overlap_Y, Blocks_Y[k, :]), axis=None)

            Overlap_X = InB_X[-NOverlap:]
            Overlap_Y = InB_Y[-NOverlap:]

            OutB_X = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(InB_X)) * FDE))
            OutB_Y = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(InB_Y)) * FDE))

            outX[k, :] = OutB_X[int(NOverlap/2): int(-NOverlap/2)]
            outY[k, :] = OutB_Y[int(NOverlap/2): int(-NOverlap/2)]

        outX = np.reshape(outX, -1)
        outY = np.reshape(outY, -1)

        return outX[int(NOverlap/2): int(np.floor(-NOverlap/2))], outY[int(NOverlap/2): int(np.floor(-NOverlap/2))]