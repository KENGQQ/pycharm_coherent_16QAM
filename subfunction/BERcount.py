import numpy as np

def BERcount(tg, pred,PAM_order):
    # tg=tg[:,0]
    tg = tg.tolist()
    # pred=pred[:,0]
    pred = pred.tolist()
    error = 0
    
    if PAM_order == 2:
        for indx in range(len(tg)):
            if np.real(tg[indx]) * np.real(pred[indx]) > 0 and np.imag(tg[indx]) * np.imag(pred[indx]) > 0:
                pass
            else:
                error += 1
    if PAM_order == 4:

        for indx in range(len(tg)):      
            
            if np.real(pred[indx]) > 2:
                if np.imag(pred[indx]) > 2 and tg[indx] != 3 + 3j:
                    error += 1
                    # print(indx)
                if 2 > np.imag(pred[indx]) > 0 and tg[indx] != 3 + 1j:
                    error += 1
                    # print(indx)
                if 0 > np.imag(pred[indx]) > -2 and tg[indx] != 3 - 1j:
                    error += 1
                    # print(indx)
                if -2 > np.imag(pred[indx]) > -4 and tg[indx] != 3 - 3j:
                    error += 1
                    # print(indx)
            if 2 > np.real(pred[indx]) > 0:
                if np.imag(pred[indx]) > 2 and tg[indx] != 1 + 3j:
                    error += 1
                    # print(indx)
                if 2 > np.imag(pred[indx]) > 0 and tg[indx] != 1 + 1j:
                    error += 1
                    # print(indx)
                if 0 > np.imag(pred[indx]) > -2 and tg[indx] != 1 - 1j:
                    error += 1
                    # print(indx)
                if -2 > np.imag(pred[indx]) > -4 and tg[indx] != 1 - 3j:
                    error += 1
                    # print(indx)
            if 0 > np.real(pred[indx]) > -2:
                if np.imag(pred[indx]) > 2 and tg[indx] != -1 + 3j:
                    error += 1
                    # print(indx)
                if 2 > np.imag(pred[indx]) > 0 and tg[indx] != -1 + 1j:
                    error += 1
                    # print(indx)
                if 0 > np.imag(pred[indx]) > -2 and tg[indx] != -1 - 1j:
                    error += 1
                    # print(indx)
                if -2 > np.imag(pred[indx]) > -4 and tg[indx] != -1 - 3j:
                    error += 1
                    # print(indx)
            if -2 > np.real(pred[indx]) > -4:
                if np.imag(pred[indx]) > 2 and tg[indx] != -3 + 3j:
                    error += 1
                    # print(indx)
                if 2 > np.imag(pred[indx]) > 0 and tg[indx] != -3 + 1j:
                    error += 1
                    # print(indx)
                if 0 > np.imag(pred[indx]) > -2 and tg[indx] != -3 - 1j:
                    error += 1
                    # print(indx)
                if -2 > np.imag(pred[indx]) > -4 and tg[indx] != -3 - 3j:
                    error += 1
                    # print(indx)
    BER = error / len(tg)
    print(len(tg))
    print(error)
    return BER
