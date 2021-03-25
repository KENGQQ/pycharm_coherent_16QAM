import numpy as np


def QAM16_LogicTx(Logic_Ichannel1, Logic_Ichannel2 ,Logic_Qchannel1 ,Logic_Qchannel2):

    assert(len(Logic_Ichannel1) == len(Logic_Ichannel2))
    assert(len(Logic_Qchannel1) == len(Logic_Qchannel2))

    TxI = np.zeros(len(Logic_Ichannel1))
    TxQ = np.zeros(len(Logic_Qchannel1))


    for i in range(len(Logic_Ichannel1)):
        if Logic_Ichannel1[i] == 1 and Logic_Ichannel2[i] == 1:
            TxI[i] = 3
        if Logic_Ichannel1[i] == 0 and Logic_Ichannel2[i] == 1:
            TxI[i] = 1
        if Logic_Ichannel1[i] == 1 and Logic_Ichannel2[i] == 0:
            TxI[i] = -1
        if Logic_Ichannel1[i] == 0 and Logic_Ichannel2[i] == 0:
            TxI[i] = -3

    for i in range(len(Logic_Qchannel1)):
        if Logic_Qchannel1[i] == 1 and Logic_Qchannel2[i] == 1:
            TxQ[i] = 3
        if Logic_Qchannel1[i] == 0 and Logic_Qchannel2[i] == 1:
            TxQ[i] = 1
        if Logic_Qchannel1[i] == 1 and Logic_Qchannel2[i] == 0:
            TxQ[i] = -1
        if Logic_Qchannel1[i] == 0 and Logic_Qchannel2[i] == 0:
            TxQ[i] = -3

    return TxI, TxQ