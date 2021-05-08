import scipy.io as sio
import numpy as np
import pandas as pd


class Parameter:
    def __init__(self, datafolder, simulation=False):
        if simulation == True:
            self.symbolRate = 56e9
            self.sampleRate = 32 * self.symbolRate
            self.upsamplenum = 1
            self.pamorder = 4
            self.Prbsnum = 13
            self.samplepersymbol = self.sampleRate / self.symbolRate
            self.resamplenumber = int(self.samplepersymbol * self.upsamplenum)
            self.datafolder = datafolder
            time = 1000
            # self.RxXI = pd.read_table(self.datafolder + 'RxXI.txt', names=['RxXI'])['RxXI'].tolist()
            # self.RxXQ = pd.read_table(self.datafolder + 'RxXQ.txt', names=['RxXQ'])['RxXQ'].tolist()
            # self.RxYI = pd.read_table(self.datafolder + 'RxYI.txt', names=['RxYI'])['RxYI'].tolist()
            # self.RxYQ = pd.read_table(self.datafolder + 'RxYQ.txt', names=['RxYQ'])['RxYQ'].tolist()
            # self.TxXI = pd.read_table(self.datafolder + 'TxXI.txt', names=['TxXI'])['TxXI'].tolist()
            # self.TxXQ = pd.read_table(self.datafolder + 'TxXQ.txt', names=['TxXQ'])['TxXQ'].tolist()
            # self.TxYI = pd.read_table(self.datafolder + 'TxYI.txt', names=['TxYI'])['TxYI'].tolist()
            # self.TxYQ = pd.read_table(self.datafolder + 'TxYQ.txt', names=['TxYQ'])['TxYQ'].tolist()
            self.RxXI = np.mat(pd.read_table(self.datafolder + 'RxXI.txt', names=['RxXI'])['RxXI'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]
            self.RxXQ = np.mat(pd.read_table(self.datafolder + 'RxXQ.txt', names=['RxXQ'])['RxXQ'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]
            self.RxYI = np.mat(pd.read_table(self.datafolder + 'RxYI.txt', names=['RxYI'])['RxYI'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]
            self.RxYQ = np.mat(pd.read_table(self.datafolder + 'RxYQ.txt', names=['RxYQ'])['RxYQ'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]
            # self.TxXI = np.mat(pd.read_table(self.datafolder + 'TxXI.txt', names=['TxXI'])['TxXI'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]
            # self.TxXQ = np.mat(pd.read_table(self.datafolder + 'TxXQ.txt', names=['TxXQ'])['TxXQ'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]
            # self.TxYI = np.mat(pd.read_table(self.datafolder + 'TxYI.txt', names=['TxYI'])['TxYI'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]
            # self.TxYQ = np.mat(pd.read_table(self.datafolder + 'TxYQ.txt', names=['TxYQ'])['TxYQ'],dtype='complex_').T[int(-time*self.sampleRate/1e9):,0]

            self.datafolder = r'G:\KENG\GoogleCloud\OptsimData_coherent\QAM16_data/'
            # self.datafolder = r'C:\Users\kengw\Google 雲端硬碟 (keng.eo08g@nctu.edu.tw)\OptsimData_coherent\QAM16_data/'
            self.datafolder += '20210504_DATA_new/'
            self.LogTxXI1=pd.read_table(self.datafolder+'LogTxXI1.txt',names=['L1'])['L1'].tolist()
            self.LogTxXI2=pd.read_table(self.datafolder+'LogTxXI2.txt',names=['L1'])['L1'].tolist()
            self.LogTxXQ1=pd.read_table(self.datafolder+'LogTxXQ1.txt',names=['L1'])['L1'].tolist()
            self.LogTxXQ2=pd.read_table(self.datafolder+'LogTxXQ2.txt',names=['L1'])['L1'].tolist()
            self.LogTxYI1=pd.read_table(self.datafolder+'LogTxYI1.txt',names=['L1'])['L1'].tolist()
            self.LogTxYI2=pd.read_table(self.datafolder+'LogTxYI2.txt',names=['L1'])['L1'].tolist()
            self.LogTxYQ1=pd.read_table(self.datafolder+'LogTxYQ1.txt',names=['L1'])['L1'].tolist()
            self.LogTxYQ2=pd.read_table(self.datafolder+'LogTxYQ2.txt',names=['L1'])['L1'].tolist()

        else:
            self.symbolRate = 28.125e9
            self.sampleRate = 50e9
            self.pamorder = 4
            self.upsamplenum = 9
            self.loadfile = sio.loadmat(datafolder)
            self.Prbsnum = 15
            self.PRBS = pd.read_table(r'PRBS_TX15.txt', names=['PRBS'])['PRBS'].tolist()
            self.samplepersymbol = self.sampleRate / self.symbolRate
            self.resamplenumber = int(self.samplepersymbol * self.upsamplenum)
            self.RxXI = self.loadfile["Vblock"]["Values"][0][0][0].tolist()
            self.RxXQ = self.loadfile["Vblock"]["Values"][0][1][0].tolist()
            self.RxYI = self.loadfile["Vblock"]["Values"][0][2][0].tolist()
            self.RxYQ = self.loadfile["Vblock"]["Values"][0][3][0].tolist()
            self.Xsoft = self.loadfile["zXSym"]["Values"][0][0][0].tolist()  #tek DSP
            self.Ysoft = self.loadfile["zYSym"]["Values"][0][0][0].tolist()  #tek DSP

            if self.pamorder == 4:
                self.TxXI = self.loadfile["zXSym"]["SeqRe"][0][0][:].tolist()
                self.TxXI = np.reshape(np.reshape(self.TxXI, -1), (-1, 2), order='F')
                self.TxXQ = self.loadfile["zXSym"]["SeqIm"][0][0][:].tolist()
                self.TxXQ = np.reshape(np.reshape(self.TxXQ, -1), (-1, 2), order='F')
                self.TxYI = self.loadfile["zYSym"]["SeqRe"][0][0][:].tolist()
                self.TxYI = np.reshape(np.reshape(self.TxYI, -1), (-1, 2), order='F')
                self.TxYQ = self.loadfile["zYSym"]["SeqIm"][0][0][:].tolist()
                self.TxYQ = np.reshape(np.reshape(self.TxYQ, -1), (-1, 2), order='F')
            else:
                self.TxXI = self.loadfile["zXSym"]["SeqRe"][0][0][0].tolist()
                self.TxXQ = self.loadfile["zXSym"]["SeqIm"][0][0][0].tolist()
                self.TxYI = self.loadfile["zYSym"]["SeqRe"][0][0][0].tolist()
                self.TxYQ = self.loadfile["zYSym"]["SeqIm"][0][0][0].tolist()

