import math  
import numpy as np

############### setting
#multi-anttena
Nt = 4   # transmit antenna
K = 1    # users
Nr = 2   # receive antenna
d = 2    # data streams  ** d <= min(Nt/K, Nr)  **
P = 1    # power
bits_per_symbol = 2
bits_per_mimo = bits_per_symbol*d 

Es_16qam = 10
mapping_table_16qam = {
    (1,0,1,0) : -3-3j,
    (1,0,1,1) : -3-1j,
    (1,0,0,1) : -3+1j,
    (1,0,0,0) : -3+3j,
    (1,1,1,0) : -1-3j,
    (1,1,1,1) : -1-1j,
    (1,1,0,1) : -1+1j,
    (1,1,0,0) : -1+3j,
    (0,1,1,0) : 1-3j,
    (0,1,1,1) : 1-1j,
    (0,1,0,1) : 1+1j,
    (0,1,0,0) : 1+3j,
    (0,0,1,0) : 3-3j,
    (0,0,1,1) : 3-1j,
    (0,0,0,1) : 3+1j,
    (0,0,0,0) : 3+3j,
}

Es_qpsk = 2
mapping_table_qpsk = {
    (0,0) :  1+1j,
    (0,1) :  1-1j,
    (1,0) : -1+1j,
    (1,1) : -1-1j,
}

bits_per_digit = 9


demapping_table_16qam = {v : k for k, v in mapping_table_16qam.items()}
demapping_table_qpsk = {v : k for k, v in mapping_table_qpsk.items()}

#####################
def group_bits(bitc, mod_type='QPSK'):
    if mod_type=='QPSK':
        bits_per_symbol = 2
    elif mod_type=='16QAM':
        bits_per_symbol = 4
    bity = []
    x = 0
    for i in range((len(bitc)//bits_per_symbol)):
        bity.append(bitc[x:x+bits_per_symbol])
        x = x+bits_per_symbol
    return bity

def group_symbol(bitc, d, mod_type):
    if mod_type=='QPSK':
        bits_per_symbol = 2
        bits_per_mimo = bits_per_symbol*d 
    elif mod_type=='16QAM':
        bits_per_symbol = 4
        bits_per_mimo = bits_per_symbol*d 
    bitm = []
    x = 0
    for i in range((len(bitc)//bits_per_mimo)):
        bitm.append(bitc[x:x+bits_per_mimo])
        x = x+bits_per_mimo
    return bitm

def modulate(signal_bits, d, mod_type='QPSK'):
    symbol_signal = np.zeros((d, 1), dtype=complex)
    bits_k = group_bits(signal_bits, mod_type) 
    modulated_symbols = []
    for bits in bits_k:
        if mod_type=='QPSK':
            symbol = mapping_table_qpsk[tuple(bits)]
            modulated_symbols.append(symbol)
        elif mod_type=='16QAM':
            symbol = mapping_table_16qam[tuple(bits)]
            modulated_symbols.append(symbol)
        else:
            print(mod_type, 'is not supported! The supported modulation scheme includes QPSK and 16QAM.')
            exit(0)
    modulated_symbols = np.array(modulated_symbols) 
    symbol_signal[:,0] = modulated_symbols
    return symbol_signal


def SVD_Precoding(H, P):
    U,D,V = np.linalg.svd(H, full_matrices=True)
    W_svd = V.conj().T[:,:d]
    M_svd = U

    W_svd_norm = np.sqrt(np.trace(W_svd.dot(W_svd.conj().T)))   #power norm
    W_svd = W_svd * np.sqrt(P) / W_svd_norm
    return W_svd, D, M_svd
    
def SignalNorm(signal, mod_type='QPSK'):
    signal_power = np.mean(abs(signal**2))
    if mod_type=='QPSK':
        return signal / np.sqrt(Es_qpsk) /np.sqrt(2) * 2 
    elif mod_type=='16QAM':
        return signal / np.sqrt(Es_16qam) /np.sqrt(2) * 2 
    else:
        print(mod_type, 'is not supported! The supported modulation schemes include QPSK and 16QAM.')
        exit(0)


def SignalDenorm(signal, mod_type='QPSK'):
    signal_power = np.mean(abs(signal**2))
    if mod_type=='QPSK':
        return signal * np.sqrt(Es_qpsk)
    elif mod_type=='16QAM':
        return signal * np.sqrt(Es_16qam)
    else:
        print(mod_type, 'is not supported! The supported modulation scheme includes QPSK and 16QAM.')
        exit(0)

def Demapping(modulated_signal, mod_type='QPSK'):
    if mod_type=='QPSK':
        constellation = np.array([x for x in demapping_table_qpsk.keys()])
    elif mod_type=='16QAM':
        constellation = np.array([x for x in demapping_table_16qam.keys()])
    else:
        print(mod_type, 'is not supported! The supported modulation scheme includes QPSK and 16QAM.')
        exit(0)
    # calculate distance of each RX point to each possible point
    dists = abs(modulated_signal.reshape((-1,1)) - constellation.reshape((1,-1)))
    # for each element in modulated symbols, choose the index in constellation that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    if mod_type=='QPSK':
        return np.vstack([demapping_table_qpsk[C] for C in hardDecision]), hardDecision
    elif mod_type=='16QAM':
        return np.vstack([demapping_table_16qam[C] for C in hardDecision]), hardDecision
    else:
        print(mod_type, 'is not supported! The supported modulation scheme includes QPSK and 16QAM.')
        exit(0)


##########################