import scipy.io
import numpy as np
import scipy.signal as signal
from subfunction.Constellation import *
from subfunction.DataNormalize import *
from subfunction.Histogram2D import *
from subfunction.IQimbaCompensator import *
# from subfunction.corr import *
from subfunction.BERcount import *
from subfunction.SNR import *
from subfunction.Downsample import *
from subfunction.excelrecord import *

from CMA import *
from CMA_tseng import *

from KENG_Tx2Bit import *
from KENG_downsample import *
from KENG_Parameter_16QAM import *
from KENG_phaserecovery import *
from KENG_correlation import *
from KENG_Volterra import *

from Equalizer import *
from Phaserecovery import *

address = r'data/20210108_FINISAR_16QAM/00KM_16QAM.mat'
Imageaddress = r'data/20210108_FINISAR_16QAM/' + 'image'
parameter = Parameter(address, simulation=False)
# open_excel(address)

print("symbolrate = {}Gbit/s\npamorder = {}\nresamplenumber = {}".format(parameter.symbolRate / 1e9, parameter.pamorder, parameter.resamplenumber))
Tx2Bit = KENG_Tx2Bit(PAM_order=parameter.pamorder)
downsample_Tx = KENG_downsample(down_coeff=parameter.resamplenumber)
downsample_Rx = KENG_downsample(down_coeff=parameter.resamplenumber)

# Tx_XI, Tx_XQ = DataNormalize(parameter.TxXI, parameter.TxXQ, parameter.pamorder)
# Tx_YI, Tx_YQ = DataNormalize(parameter.TxYI, parameter.TxYQ, parameter.pamorder)
# TxXI = Tx2Bit.return_Tx(Tx_XI)
# TxXQ = Tx2Bit.return_Tx(Tx_XQ)
# TxYI = Tx2Bit.return_Tx(Tx_YI)
# TxYQ = Tx2Bit.return_Tx(Tx_YQ)
#
# Tx_Signal_X = TxXI[:, 0] + 1j * TxXQ[:, 0]
# Tx_Signal_Y = TxYI[:, 0] + 1j * TxYQ[:, 0]
# Histogram2D('Tx', Tx_Signal_X[0:10000], Imageaddress)

# Rx Upsample
Rx_XI, Rx_XQ = DataNormalize(signal.resample_poly(parameter.RxXI, up=parameter.upsamplenum, down=1),
                           signal.resample_poly(parameter.RxXQ, up=parameter.upsamplenum, down=1),
                           parameter.pamorder)
Rx_YI, Rx_YQ = DataNormalize(signal.resample_poly(parameter.RxYI, up=parameter.upsamplenum, down=1),
                           signal.resample_poly(parameter.RxYQ, up=parameter.upsamplenum, down=1),
                           parameter.pamorder)

# print('Tx_Resample Length={}'.format(len(Tx_Signal_X)), 'Rx_Resample Length={}'.format(len(Rx_XI)))
prbs = np.ceil(DataNormalize(parameter.PRBS, [], parameter.pamorder))
XSNR, XEVM, YSNR, YEVM = np.zeros(parameter.resamplenumber), np.zeros(parameter.resamplenumber), np.zeros(
    parameter.resamplenumber), np.zeros(parameter.resamplenumber)

#Eye position scan2
for eyepos in range(3,4):
    down_num = eyepos
    print('eye position = {}'.format(down_num))
    n = 1
    RxXI = signal.resample_poly(Rx_XI[down_num:], up=1, down=parameter.resamplenumber / n)
    RxXQ = signal.resample_poly(Rx_XQ[down_num:], up=1, down=parameter.resamplenumber / n)
    RxYI = signal.resample_poly(Rx_YI[down_num:], up=1, down=parameter.resamplenumber / n)
    RxYQ = signal.resample_poly(Rx_YQ[down_num:], up=1, down=parameter.resamplenumber / n)
    Rx_Signal_X = RxXI + 1j * RxXQ
    Rx_Signal_Y = RxYI + 1j * RxYQ

    for taps in range(15, 17, 2):
        cma = CMA_single(Rx_Signal_X[30000:130000], Rx_Signal_Y[30000:130000], taps=taps, iter=20, mean=0)

        # cma.qam_4_side_real()
        cma.qam_4_butter_real()
        # cma.qam_4_butter_real_shift()
        # cma.qam_4_butter_conj_shift()
        # cma.qam_4_side_conj()

        # Rx_X_CMA, Rx_Y_CMA = Downsample(cma.rx_x_cma, n, cma.center), Downsample(cma.rx_y_cma, n, cma.center)
        Rx_X_CMA = cma.rx_x_cma[cma.rx_x_cma != 0]
        Rx_Y_CMA = cma.rx_y_cma[cma.rx_y_cma != 0]
        Histogram2D('CMA_X_{} taps={} {}'.format(eyepos, cma.cmataps, cma.type),
                    Rx_X_CMA, Imageaddress)
        Histogram2D('CMA_Y_{} taps={} {}'.format(eyepos, cma.cmataps, cma.type),
                    Rx_Y_CMA, Imageaddress)


    #--------------------------- X PART------------------------
    print('--------------------------------')
    print('X part')
    ph = KENG_phaserecovery()
    FOcompen_X = ph.FreqOffsetComp(Rx_X_CMA)
    # Histogram2D('KENG_FOcompensate_X', FOcompen_X, Imageaddress)

    DDPLL_RxX = ph.PLL(FOcompen_X)
    # Histogram2D('KENG_FreqOffset_X', DDPLL_RxX[0, :], Imageaddress)

    phasenoise_RxX = DDPLL_RxX
    PN_RxX = ph.QAM_6(phasenoise_RxX, c1_radius=1.55, c2_radius=3.2)
    PN_RxX = PN_RxX[PN_RxX != 0]
    Histogram2D('KENG_PhaseNoise_X', PN_RxX, Imageaddress)

    Normal_ph_RxX_real, Normal_ph_RxX_imag = DataNormalize(np.real(PN_RxX), np.imag(PN_RxX), parameter.pamorder)
    Normal_ph_RxX = Normal_ph_RxX_real + 1j * Normal_ph_RxX_imag
    # Histogram2D('KENG_PLL_Normalized_Y', Normal_ph_RxX, Imageaddress)

    Correlation = KENG_corr(window_length=7000)
    TxX_real, RxX_real, p = Correlation.corr(TxXI, np.real(Normal_ph_RxX[0:50000]), 13)
    TxX_imag, RxX_imag, p = Correlation.corr(TxXQ, np.imag(Normal_ph_RxX[0:50000]), 13)
    RxX_corr = RxX_real[0:40000] + 1j * RxX_imag[0:40000]
    TxX_corr = TxX_real[0:40000] + 1j * TxX_imag[0:40000]
    # Histogram2D('KENG_Corr_X', RxX_corr, Imageaddress)

    SNR_X, EVM_X = SNR(RxX_corr, TxX_corr)
    bercount_X = BERcount(np.array(TxX_corr), np.array(RxX_corr), parameter.pamorder)
    print('BER_X = {} \nSNR_X = {} \nEVM_X = {}'.format(bercount_X, SNR_X, EVM_X))
    XSNR[eyepos] ,XEVM[eyepos] = SNR_X, EVM_X
    #--------------------------- Y PART------------------------
    print('--------------------------------')
    print('Y part')
    ph = KENG_phaserecovery()
    FOcompen_Y = ph.FreqOffsetComp(Rx_Y_CMA)
    # Histogram2D('KENG_FOcompensate_Y', FOcompen_Y, Imageaddress)

    DDPLL_RxY = ph.PLL(FOcompen_Y)
    # Histogram2D('KENG_FreqOffset_Y', DDPLL_RxY[0, :], Imageaddress)

    phasenoise_RxY = DDPLL_RxY
    PN_RxY = ph.QAM_6(phasenoise_RxY, c1_radius=1.55, c2_radius=3.2)
    PN_RxY = PN_RxY[PN_RxY != 0]
    Histogram2D('KENG_PhaseNoise_Y', PN_RxY, Imageaddress)

    Normal_ph_RxY_real, Normal_ph_RxY_imag = DataNormalize(np.real(PN_RxY), np.imag(PN_RxY), parameter.pamorder)
    Normal_ph_RxY = Normal_ph_RxY_real + 1j * Normal_ph_RxY_imag
    # Histogram2D('KENG_PLL_Normalized_Y', Normal_ph_RxY, Imageaddress)

    Correlation = KENG_corr(window_length=7000)
    TxY_real, RxY_real, p = Correlation.corr(TxYI, np.real(Normal_ph_RxY[0:50000]), 13)
    TxY_imag, RxY_imag, p = Correlation.corr(TxYQ, np.imag(Normal_ph_RxY[0:50000]), 13)
    RxY_corr = RxY_real[0:40000] + 1j * RxY_imag[0:40000]
    TxY_corr = TxY_real[0:40000] + 1j * TxY_imag[0:40000]
    # Histogram2D('KENG_Corr_Y', RxY_corr, Imageaddress)

    SNR_Y, EVM_Y = SNR(RxY_corr, TxY_corr)
    bercount_Y = BERcount(np.array(TxY_corr), np.array(RxY_corr), parameter.pamorder)
    print('BER_Y = {} \nSNR_Y = {} \nEVM_Y = {}'.format(bercount_Y, SNR_Y, EVM_Y))
    YSNR[eyepos], YEVM[eyepos] = SNR_Y, EVM_Y
    print('--------------------------------')



#---
equalizer_real = Equalizer(np.real(np.array(Tx_corr.T)[0,:]), np.real(np.array(Rx_corr.T)[0,:]), 3, [11, 31, 31], 0.5)
equalizer_imag = Equalizer(np.imag(np.array(Tx_corr.T)[0,:]), np.imag(np.array(Rx_corr.T)[0,:]), 3, [11, 31, 31], 0.5)
Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
#---
# equalizer_complex = Equalizer(Tx_corr, Rx_corr, 3, [11, 3, 3], 0.5)
# equalizer_complex = Equalizer( np.array(Tx_corr.T)[0,:], np.array(Rx_corr.T)[0,:], 3, [21, 3, 1], 0.1)

# equalizer_real = Equalizer(np.real(Tx_corr), np.real(Rx_corr), 3, [11, 11, 11])
# equalizer_imag = Equalizer(np.imag(Tx_corr), np.imag(Rx_corr), 3, [11, 11, 11])
# Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
# Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
# Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
# Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag
# Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()
snr_volterra, evm_volterra = SNR(Tx_real_volterra, Rx_real_volterra)
# snr_volterra, evm_volterra = SNR(Rx_complex_volterra, Tx_complex_volterra)
# bercount = BERcount(Rx_complex_volterra, Tx_complex_volterra, parameter.pamorder)
# bercount = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
# print(bercount)
print(snr_volterra, evm_volterra)
Histogram2D("ComplexVolterra", Rx_real_volterra, snr_volterra, evm_volterra)
# if __name__ == '__main__':
#     main()