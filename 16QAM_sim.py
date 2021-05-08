import scipy.io
import numpy as np
import scipy.signal as signal
from subfunction.Constellation import *
from subfunction.DataNormalize import *
from subfunction.Histogram2D import *
from subfunction.Histogram2D_tseng import *
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
from KENG_16QAM_LogicTx import *

from Equalizer import *
from Phaserecovery import *

address = r'G:\KENG\GoogleCloud\OptsimData_coherent\QAM16_data/'
# address = r'C:\Users\kengw\Google 雲端硬碟 (keng.eo08g@nctu.edu.tw)\OptsimData_coherent\QAM16_data/'
folder = '20210504_DATA_new/500KLW_1GFO_50GBW_0dBLO_sample32_1000ns_CD-1280_EDC0_TxO-2dBm_RxO-08dBm_OSNR26dB_LO00dBm/'
address += folder

Imageaddress = address + 'image'
parameter = Parameter(address, simulation=True)
open_excel(address)
##################### control
isplot = 0
iswrite = 1
xpart, ypart = 1, 1
eyestart, eyeend = 0, 32
tap_start, tap_end = 57, 91
cma_stage= [1, 2, 3]; cma_iter = [50, 20, 5]
isrealvolterra = 0
iscomplexvolterra = 0

window_length = 7000
correlation_length = 55000
final_length = correlation_length - 9000
##################### control

print("symbolrate = {}Gbit/s\npamorder = {}\nresamplenumber = {}".format(parameter.symbolRate / 1e9, parameter.pamorder, parameter.resamplenumber))
Tx2Bit = KENG_Tx2Bit(PAM_order=parameter.pamorder)
downsample_Tx = KENG_downsample(down_coeff=parameter.resamplenumber)
downsample_Rx = KENG_downsample(down_coeff=parameter.resamplenumber)

# Tx_XI, Tx_XQ = DataNormalize(parameter.TxXI ,parameter.TxXQ ,parameter.pamorder)
# Tx_YI, Tx_YQ = DataNormalize(parameter.TxYI ,parameter.TxYQ ,parameter.pamorder)
# parameter.TxXI = np.array(parameter.TxXI)
# parameter.TxXQ = np.array(parameter.TxXQ)
# parameter.TxYI = np.array(parameter.TxYI)
# parameter.TxYQ = np.array(parameter.TxYQ)
#
# Tx_XI = Tx2Bit.return_Tx(parameter.TxXI)
# Tx_XQ = Tx2Bit.return_Tx(parameter.TxXQ)
# Tx_YI = Tx2Bit.return_Tx(parameter.TxYI)
# Tx_YQ = Tx2Bit.return_Tx(parameter.TxYQ)

Rx_XI, Rx_XQ = DataNormalize(parameter.RxXI, parameter.RxXQ, parameter.pamorder)
Rx_YI, Rx_YQ = DataNormalize(parameter.RxYI, parameter.RxYQ, parameter.pamorder)

XSNR, XEVM, YSNR, YEVM = np.zeros(parameter.resamplenumber), np.zeros(parameter.resamplenumber), np.zeros(
    parameter.resamplenumber), np.zeros(parameter.resamplenumber)

# sio.savemat('RxX_py.mat', {'RxX': Rx_XI + 1j * Rx_XQ})
# sio.savemat('RxY_py.mat', {'RxY': Rx_YI + 1j * Rx_YQ})

LogTxXI1 = downsample_Tx.return_value(parameter.LogTxXI1)
LogTxXI2 = downsample_Tx.return_value(parameter.LogTxXI2)
LogTxXQ1 = downsample_Tx.return_value(parameter.LogTxXQ1)
LogTxXQ2 = downsample_Tx.return_value(parameter.LogTxXQ2)
LogTxYI1 = downsample_Tx.return_value(parameter.LogTxYI1)
LogTxYI2 = downsample_Tx.return_value(parameter.LogTxYI2)
LogTxYQ1 = downsample_Tx.return_value(parameter.LogTxYQ1)
LogTxYQ2 = downsample_Tx.return_value(parameter.LogTxYQ2)

TxXI, TxXQ = QAM16_LogicTx(LogTxXI1, LogTxXI2, LogTxXQ1, LogTxXQ2)
TxYI, TxYQ = QAM16_LogicTx(LogTxYI1, LogTxYI2, LogTxYQ1, LogTxYQ2)
Tx_Signal_X = TxXI + 1j * TxXQ
Tx_Signal_Y = TxYI + 1j * TxYQ

# TxXI = downsample_Tx.return_value(Tx_XI[down_num:])
# TxXQ = downsample_Tx.return_value(Tx_XQ[down_num:])
# TxYI = downsample_Tx.return_value(Tx_YI[down_num:])
# TxYQ = downsample_Tx.return_value(Tx_YQ[down_num:])
# Tx_Signal_X = TxXI + 1j * TxXQ
# Tx_Signal_Y = TxYI + 1j * TxYQ
# Histogram2D('Tx_X_normalized', Tx_Signal_X, Imageaddress)

for eyepos in range(eyestart, eyeend, 1):
    down_num = eyepos

    n = 1
    RxXI = signal.resample_poly(Rx_XI[down_num:], up=1, down=parameter.resamplenumber / n)
    RxXQ = signal.resample_poly(Rx_XQ[down_num:], up=1, down=parameter.resamplenumber / n)
    RxYI = signal.resample_poly(Rx_YI[down_num:], up=1, down=parameter.resamplenumber / n)
    RxYQ = signal.resample_poly(Rx_YQ[down_num:], up=1, down=parameter.resamplenumber / n)
    Rx_Signal_X = RxXI[:, 0] + 1j * RxXQ[:, 0]
    Rx_Signal_Y = RxYI[:, 0] + 1j * RxYQ[:, 0]

    # RxXI=downsample_Rx.return_value(Rx_XI[down_num:])
    # RxXQ=downsample_Rx.return_value(Rx_XQ[down_num:])
    # RxYI=downsample_Rx.return_value(Rx_YI[down_num:])
    # RxYQ=downsample_Rx.return_value(Rx_YQ[down_num:])
    # Rx_Signal_X=RxXI+1j*RxXQ
    # Rx_Signal_Y=RxYI+1j*RxYQ
    # RxXI ,RxXQ=DataNormalize(RxXI ,RxXQ, parameter.pamorder)
    # RxYI ,RxYQ=DataNormalize(RxYI ,RxYQ, parameter.pamorder)

    Histogram2D('Rx_X_origin_{}'.format(eyepos), Rx_Signal_X, Imageaddress)
    Histogram2D('Rx_Y_origin_{}'.format(eyepos), Rx_Signal_Y, Imageaddress)
    #########IQimba################
    # Rx_Signal_X = np.reshape(Rx_Signal_X,(1,-1))
    # Rx_Signal_Y = np.reshape(Rx_Signal_Y,(1,-1))
    # Rx_X_iqimba = IQimbaCompensator(Rx_Signal_X, 1e-6)
    # Rx_Y_iqimba = IQimbaCompensator(Rx_Signal_Y, 1e-6)
    # Rx_X_iqimba = Rx_X_iqimba[0]
    # Rx_Y_iqimba = Rx_Y_iqimba[0]
    # Histogram2D("IQimba", Rx_X_iqimba, Imageaddress)
    ##########IQimba################
    for taps in range(tap_start, tap_end, 2):
        print("eye : {} ,tap : {}".format(eyepos,taps))
        # Rx_Signal_X_mat = sio.loadmat('RxX_mat.mat')
        # Rx_Signal_X_mat = Rx_Signal_X_mat['rxSym']
        # Rx_Signal_X = np.reshape(Rx_Signal_X_mat, -1)
        # Rx_Signal_Y_mat = sio.loadmat('RxY_mat.mat')
        # Rx_Signal_Y_mat = Rx_Signal_Y_mat['rxSym']
        # Rx_Signal_Y = np.reshape(Rx_Signal_Y_mat, -1)

        cma = CMA_single(Rx_Signal_X, Rx_Signal_Y, taps=taps, iter=cma_iter[0], mean=0)

        # cma.qam_4_butter_real()
        # cma.qam_4_butter_real_shift()
        # cma.qam_4_side_real_m()
        # cma.qam_4_side_real()
        # cma.qam_4_side_conj()
        # cma.qam_4_side_real_shift()
        # cma.qam_4_butter_conj()
        # cma.MCMA_MDD_()
        # cma.qam_4_butter_conj_shift()
        # cma.qam_4_side_conj_SBD_polarization()

        # cma.qam_4_butter_RD(stage=cma_stage[0])
        cma.qam_4_side_RD(stage=cma_stage[0])
        Rx_X_CMA_stage1 = cma.rx_x_cma[cma.rx_x_cma != 0]
        Rx_Y_CMA_stage1 = cma.rx_y_cma[cma.rx_y_cma != 0]
        if isplot == True:
            Histogram2D('CMA_X_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type),
                        Rx_X_CMA_stage1, Imageaddress)
            Histogram2D('CMA_Y_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type),
                        Rx_Y_CMA_stage1, Imageaddress)
        Rx_X_CMA = Rx_X_CMA_stage1
        Rx_Y_CMA = Rx_Y_CMA_stage1

        # Rx_X_CMA, Rx_Y_CMA = Downsample(cma.rx_x_cma, n, cma.center), Downsample(cma.rx_y_cma, n, cma.center)

        # cma = CMA_single(Rx_Signal_X, Rx_Signal_Y, taps=taps, iter=20, mean=0)
        # cma.stepsize_x, cma.stepsize_y = cma.stepsizelist[7], cma.stepsizelist[7]
        # cma.qam_4_butter_RD(stage=cma_stage[1])
        # # cma.qam_4_side_RD(stage=cma_stage[1])
        # Rx_X_CMA_stage2 = cma.rx_x_cma[cma.rx_x_cma != 0]
        # Rx_Y_CMA_stage2 = cma.rx_y_cma[cma.rx_y_cma != 0]
        # if isplot == True:
        #     Histogram2D('CMA_X_{}_stage2 taps={} {}'.format(eyepos, cma.cmataps, cma.type),
        #                 Rx_X_CMA_stage2, Imageaddress)
        #     Histogram2D('CMA_Y_{}_stage1 taps={} {}'.format(eyepos, cma.cmataps, cma.type),
        #                 Rx_Y_CMA_stage2, Imageaddress)
        # Rx_X_CMA = Rx_X_CMA_stage2
        # Rx_Y_CMA = Rx_Y_CMA_stage2

        CMA_cost_X = np.round(cma.costfunx[0][-1], 5)
        CMA_cost_Y = np.round(cma.costfuny[0][-1], 5)
        # print("cost :", CMA_cost_X, CMA_cost_Y)
        # --------------

    #--------------------------- X PART------------------------
        if xpart == True :
            print('================================')
            print('X part')
            ph = KENG_phaserecovery()
            FOcompen_X = ph.FreqOffsetComp(Rx_X_CMA)
            if isplot == True: Histogram2D('KENG_FOcompensate_X', FOcompen_X, Imageaddress)

            DDPLL_RxX = ph.PLL(FOcompen_X)
            PLL_BW = ph.bandwidth
            if isplot == True: Histogram2D('KENG_FreqOffset_X', DDPLL_RxX[0, :], Imageaddress)

            phasenoise_RxX = np.reshape(DDPLL_RxX, -1)
            PN_RxX = ph.QAM_6(phasenoise_RxX, c1_radius=1.55, c2_radius=3.2)
            PN_RxX = PN_RxX[PN_RxX != 0]
            if isplot == True: Histogram2D('KENG_PhaseNoise_X', PN_RxX, Imageaddress)

            Normal_ph_RxX_real, Normal_ph_RxX_imag = DataNormalize(np.real(PN_RxX), np.imag(PN_RxX), parameter.pamorder)
            Normal_ph_RxX = Normal_ph_RxX_real + 1j * Normal_ph_RxX_imag
            # if isplot == True: Histogram2D('KENG_PLL_Normalized_X', Normal_ph_RxX, Imageaddress)

            Correlation = KENG_corr(window_length=window_length)
            TxX_real, RxX_real, p = Correlation.corr(TxXI, np.real(Normal_ph_RxX[0:correlation_length]), 13); XIshift = Correlation.shift; XI_corr = Correlation.corr;
            Correlation = KENG_corr(window_length=window_length)
            TxX_imag, RxX_imag, p = Correlation.corr(TxXQ, np.imag(Normal_ph_RxX[0:correlation_length]), 13); XQshift = Correlation.shift; XQ_corr = Correlation.corr;
            RxX_corr = RxX_real[0:final_length] + 1j * RxX_imag[0:final_length]
            TxX_corr = TxX_real[0:final_length] + 1j * TxX_imag[0:final_length]
            # if isplot == True: Histogram2D('KENG_Corr_X', RxX_corr, Imageaddress)

            SNR_X, EVM_X = SNR(RxX_corr, TxX_corr)
            bercount_X = BERcount(np.array(TxX_corr), np.array(RxX_corr), parameter.pamorder)
            print('BER_X = {} \nSNR_X = {} \nEVM_X = {}'.format(bercount_X, SNR_X, EVM_X))
            XSNR[eyepos] ,XEVM[eyepos] = SNR_X, EVM_X
            if isplot == True: Histogram2D('KENG_X_beforeVol', RxX_corr, Imageaddress, SNR_X, EVM_X)

        # --------------------------- Y PART------------------------
        if ypart == True:
            print('================================')
            print('Y part')
            ph = KENG_phaserecovery()
            FOcompen_Y = ph.FreqOffsetComp(Rx_Y_CMA)
            if isplot == True: Histogram2D('KENG_FOcompensate_Y', FOcompen_Y, Imageaddress)

            DDPLL_RxY = ph.PLL(FOcompen_Y)
            PLL_BW = ph.bandwidth
            if isplot == True: Histogram2D('KENG_FreqOffset_Y', DDPLL_RxY[0, :], Imageaddress)

            phasenoise_RxY = np.reshape(DDPLL_RxY, -1)
            PN_RxY = ph.QAM_6(phasenoise_RxY, c1_radius=1.55, c2_radius=3.2)
            # PN_RxY = ph.QAM_6(phasenoise_RxY, c1_radius=1.85, c2_radius=3.8)
            PN_RxY = PN_RxY[PN_RxY != 0]
            if isplot == True: Histogram2D('KENG_PhaseNoise_Y', PN_RxY, Imageaddress)

            Normal_ph_RxY_real, Normal_ph_RxY_imag = DataNormalize(np.real(PN_RxY), np.imag(PN_RxY), parameter.pamorder)
            Normal_ph_RxY = Normal_ph_RxY_real + 1j * Normal_ph_RxY_imag
            # if isplot == True: Histogram2D('KENG_PLL_Normalized_Y', Normal_ph_RxY, Imageaddress)

            Correlation = KENG_corr(window_length=window_length)
            TxY_real, RxY_real, p = Correlation.corr(TxYI, np.real(Normal_ph_RxY[0:correlation_length]), 13); YIshift = Correlation.shift; YI_corr = Correlation.corr;
            Correlation = KENG_corr(window_length=window_length)
            TxY_imag, RxY_imag, p = Correlation.corr(TxYQ, np.imag(Normal_ph_RxY[0:correlation_length]), 13); YQshift = Correlation.shift; YQ_corr = Correlation.corr;
            RxY_corr = RxY_real[0:final_length] + 1j * RxY_imag[0:final_length]
            TxY_corr = TxY_real[0:final_length] + 1j * TxY_imag[0:final_length]
            # if isplot == True: Histogram2D('KENG_Corr_Y', RxY_corr, Imageaddress)

            SNR_Y, EVM_Y = SNR(RxY_corr, TxY_corr)
            bercount_Y = BERcount(np.array(TxY_corr), np.array(RxY_corr), parameter.pamorder)
            print('BER_Y = {} \nSNR_Y = {} \nEVM_Y = {}'.format(bercount_Y, SNR_Y, EVM_Y))
            YSNR[eyepos], YEVM[eyepos] = SNR_Y, EVM_Y
            if isplot == True: Histogram2D('KENG_Y_beforeVol', RxY_corr, Imageaddress, SNR_Y, EVM_Y)

        if iswrite == True:
            print('----------------write excel----------------')
            parameter_record = [eyepos, str([cma.mean, cma.type, cma.overhead * 100, cma.cmataps, cma.stepsize, cma.iterator, cma.earlystop, cma.stepsizeadjust]), \
                    str([CMA_cost_X, CMA_cost_Y]), PLL_BW, str([1.55, 3.2]), \
                    str([(XIshift, XQshift), (XI_corr, XQ_corr)]), str([SNR_X, EVM_X, bercount_X]), \
                    str([(YIshift, YQshift), (YI_corr, YQ_corr)]), str([SNR_Y, EVM_Y, bercount_Y])]

            write_excel(address, parameter_record)


    # PN_Rx=ph.QAM_4(phasenoise_Rx,c1_radius=1.8,c2_radius=3.2)
    # PN_Rx = ph.QAM_5(phasenoise_Rx, c1_radius=1.8, c2_radius=3.2)
    # PN_Rx = ph.QAM_6(Rx_X_CMA, c1_radius=1.55, c2_radius=3.2)
    # PN_Rx = ph.QAM_6(phasenoise_Rx, c1_radius=2, c2_radius=5.22)
    # #--------------

    # Correlation = KENG_corr(window_length=7000)
    # Rx_real, Tx_real = Correlation.calculate_Rx(np.real(Normal_ph_RxX), -TxXI[0:Normal_ph_RxX.size])
    # Rx_imag, Tx_imag = Correlation.calculate_Rx(-np.imag(Normal_ph_Rx[10000:50000]), TxXQ[10000:50000])
    # Rx_corr = np.array(Rx_real[0:40000].T) + 1j * np.array(Rx_imag[0:40000].T)
    # Tx_corr = np.array(Tx_real[0:40000].T) + 1j * np.array(Tx_imag[0:40000].T)
    # Rx_corr = np.reshape(Rx_corr, (-1))
    # Tx_corr = np.reshape(Tx_corr, (-1))
    # Histogram2D('KENG_Corr', Rx_corr)

    # parameter_record = [eyepos, str(
    #     [cma.mean, cma.type, cma.overhead*100, cma.cmataps, cma.stepsize, cma.iterator, cma.earlystop, cma.stepsizeadjust]),
    #                     1e-3, str([0, 0, 0])]
    #
    # write_excel(address, parameter_record)
    #
    # np.save(address + 'XSNR', XSNR)
    # np.save(address + 'YSNR', YSNR)
# ===========================================volterra=========================================================
if isrealvolterra == 1:
    equalizer_real = Equalizer(np.real(np.array(TxY_corr)), np.real(np.array(RxY_corr)), 3, [31, 31, 31], 0.2)
    equalizer_imag = Equalizer(np.imag(np.array(TxY_corr)), np.imag(np.array(RxY_corr)), 3, [31, 31, 31], 0.2)
    Tx_volterra_real, Rx_volterra_real = equalizer_real.realvolterra()
    Tx_volterra_imag, Rx_volterra_imag = equalizer_imag.realvolterra()
    Tx_real_volterra = Tx_volterra_real + 1j * Tx_volterra_imag
    Rx_real_volterra = Rx_volterra_real + 1j * Rx_volterra_imag

    snr_realvolterra, evm_realvolterra = SNR(Tx_real_volterra, Rx_real_volterra)
    print("SNR_realvol : {}, EVM_realvol : {}".format(snr_realvolterra, evm_realvolterra))
    Histogram2D("RealVolterra", Rx_real_volterra, Imageaddress, snr_realvolterra, evm_realvolterra)
    #
    realvol_bercount = BERcount(Tx_real_volterra, Rx_real_volterra, parameter.pamorder)
    print("BERcount_realvol : {}".format(realvol_bercount))

# ------
if iscomplexvolterra == 1:
    equalizer_complex = Equalizer(np.array(TxY_corr), np.array(RxY_corr), 3, [31, 25, 25], 0.1)
    Tx_complex_volterra, Rx_complex_volterra = equalizer_complex.complexvolterra()

    snr_complexvolterra, evm_complexvolterra = SNR(Tx_complex_volterra, Rx_complex_volterra)
    print("SNR_cmplvol : {}, EVM_cmplvol : {}".format(snr_complexvolterra, evm_complexvolterra))
    # Histogram2D("complexVolterra", Rx_complex_volterra, Imageaddress, snr_complexvolterra, evm_complexvolterra)

    compvol_bercount = BERcount(Tx_complex_volterra, Rx_complex_volterra, parameter.pamorder)
    print("BERcount_cmplvol : {}".format(compvol_bercount))



# # # print(bercount)
# # #-----
# # vol=KENG_volterra(Rx_corr,Tx_corr,[21,1,3],25000)
# # vol_Rx=vol.first_third_order()

# kengvol_snr,kengvol_evm=SNR(vol.Rx_vol_test[0:5000,0], vol.Tx_vol_test[0:5000,0])
# print(kengvol_snr,kengvol_evm)
# Histogram2D('KENGvol',vol.Rx_vol_test[0:25000,0],kengvol_snr,kengvol_evm)


# Histogram2D_tseng("complexVolterra_tseng", Rx_complex_volterra)
# Histogram2D_tseng("CMA_tseng", Rx_X_CMA)
# Histogram2D_tseng("Focomoen_tseng", FOcompen_Y)
# Histogram2D_tseng("V-V_tseng", PN_RxX)
# Histogram2D_tseng("Rx_tseng", Rx_Signal_Y)
