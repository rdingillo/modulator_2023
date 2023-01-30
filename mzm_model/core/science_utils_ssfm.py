import math
import numpy as np
import random
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import pandas as pd
from mzm_model.core.math_utils import lin2dbm, dbm2lin

from mzm_model.core.modulator_ssfm_params import M, L, phi_laser, b, vcm_phase

'''Modulation format methods'''


# Function to create random binary string
def rand_key(p):
    # variable to store the string
    key1 = ''
    # loop to find the string of the desired length:
    for i in range(p):
        # randint function to generate 0,1 randomly and converting result into str
        temp = str(random.randint(0, 1))
        # concatenate random 0,1 to the final result
        key1 += temp

    return key1


'''EVM AND BER ESTIMATION'''


# inputs are the two lists of constellation points
def evm_rms_estimation(prbs_in, indexes, ideal_const, actual_const):
    prbs_no_duplicates = prbs_in['full'].tolist()

    evm_partial = []
    for i in prbs_no_duplicates:
        Pi = ideal_const[indexes[i]]    # ideal constellation points
        Ps = actual_const[indexes[i]]   # actual constellation points
        T = len(actual_const[indexes[i]])   # number of transmitted symbols

        # define normalization factors
        sum_p_measure = np.abs(np.sum(Ps.real**2 + Ps.imag**2))
        norm_measure = np.abs(np.sqrt(T/sum_p_measure))
        sum_p_ideal = np.abs(np.sum(Pi.real**2 + Pi.imag**2))
        norm_ideal = np.abs(np.sqrt(T/sum_p_ideal))
        norm_i_meas = Ps.real * norm_measure
        norm_i_ideal = Pi.real * norm_ideal
        norm_q_meas = Ps.imag * norm_measure
        norm_q_ideal = Pi.imag * norm_ideal
        num = np.sqrt(np.average((np.abs(norm_i_meas - norm_i_ideal)**2 + np.abs(norm_q_meas - norm_q_ideal)**2)))
        den = np.sqrt(np.average((np.abs(norm_i_ideal)**2 + np.abs(norm_q_ideal)**2)))
        # num = np.sqrt(np.mean((abs(Ps.real - Pi.real) ** 2 + abs(Ps.imag - Pi.imag) ** 2)))
        # den = np.mean(abs(Pi) ** 2)*evm_factor
        evm = (num/den)
        # numerator = (np.average((np.abs(actual_const[indexes[i]] - ideal_const[indexes[i]]))))
        # # FOR DENOMINATOR: FORMULA FROM PAPER, AVERAGE OF ALL SYMBOLS IN CONSTELLATION MULTIPLIED BY NORM FACTOR
        # denominator = (np.average((np.abs(ideal_const[indexes[i]]))))
        evm_partial.append(evm)
    evm_fin = np.mean((np.array(evm_partial)))
    return evm_fin


# SNR estimation based on EVM (in input)
def snr_estimation(evm):
    snr = 1/evm**2
    return snr


# BER estimation based on EVM (in input)
def ber_estimation(evm):
    # ber = (1 - (1/np.sqrt(M)))/(0.5*np.log2(M))*\
    #     scipy.special.erfc(np.sqrt(1.5/((M-1)*evm**2)))
    ber = (2*(1 - 1/L)/np.log2(L))*\
          math.erfc(np.sqrt((((3*np.log2(L))/(L**(2)- 1)))*2/(evm**2 * np.log2(M))))
    return ber

# AWGN channel definition
def awgn(s,SNRdB,L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal
    vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """

    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return r


def evaluate_field(power):
    out_field = np.sqrt(dbm2lin(power)) * np.exp(phi_laser * -2j)
    return out_field


def griffin_phase_electrode(vl, vr, vpi_l, vpi_r, b_ph):
    phase_l = ((2 * b_ph * vcm_phase * vpi_l - np.pi) / vpi_l) * vl - b_ph * vl ** 2
    phase_r = ((2 * b_ph * vcm_phase * vpi_r - np.pi) / vpi_r) * vr - b_ph * vr ** 2
    return [phase_l, phase_r]


def field_to_power_dbm(field):
    eo_field_conj = np.conjugate(field)
    eo_tf_power = lin2dbm(np.abs(field * eo_field_conj))
    return eo_tf_power

