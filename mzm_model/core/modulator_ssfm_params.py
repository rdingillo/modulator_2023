"modulator_ssfm_params.py"

import scipy.constants as con
import pandas as pd
import numpy as np
import os
import json

from pathlib import Path
from mzm_model.core.math_utils import db2lin

plot_flag = 1


# retrieve vpi
def eval_vpi(lambda_label, vcm, csv):
    vcm_list = [3, 5, 7, 9]
    vcm_index = str([i+1 for i in range(np.size(vcm_list)) if vcm_list[i] == vcm][0])

    df = pd.read_csv(csv, index_col=None)

    vpi_row_df = df.loc[df['VCM_LAMBDA']=='VCM'+vcm_index+'_'+lambda_label]
    vpi_row = [vpi_row_df['Vpi_MZ'+str(i+1)].values[0] for i in range(4)]
    return vpi_row

root = Path('~/PycharmProjects/modulator_2023/').expanduser()
json_source = root.parent/'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'

json_file = json_source/'ssi_cut0_pump0_2023.json'
json_params = json.load(open(json_file, 'r'))
folder_results = root/'mzm_model'/'results'
folder_resources = root/'mzm_model'/'resources'

json_out = folder_results/'json_params.json'
csv_vpi = folder_resources/'vpi_vcm_mzi_lambda.csv'

with open(json_out, 'w') as outfile:
    json.dump(json_params, outfile, indent=4)

json_spectral = json_params['spectral_info']
json_modulator = json_params['modulator_params']
json_soa = json_params['soa_params']
json_ssfm = json_params['ssfm_params']

'Physical constants'

h = con.h   # Plank constant [J*s]
q = 1.602e-19   # electron charge [C]
light_speed = con.c   # speed of light in vacuum [m/s]

'Ideal light source params'

frequency = json_spectral['frequency']    # THz
lambda_wave = round(light_speed/frequency, 10)
lambda_wave_label = str(lambda_wave*1e9)
# eta_s = 0.03   # quantum efficiency [%, photons/electrons]
# input_current = 350-3  # [mA] [taken in range between 200 mA and 500 mA]
phi_laser = 0
channel_power = json_spectral['channel_powers_dBm']

'Splitter and Combiner params'

# gammas values at 1/sqrt(2) is due to maximize InP model tf
gamma_1 = 1/np.sqrt(2)  # input field split ratio (equal to gamma_1**2 for power splitter ratio)
gamma_2 = 1/np.sqrt(2)    # output field split ratio of the combiner (consider a symmetrical split ratio, so 0.5*2=1)
k_splitter = 1  # arbitrary electric field conversion value

'Waveguides params'

vcm_bias = json_modulator['vcm_bias']
vcm_phase = json_modulator['vcm_phase']

v_diff_i = json_modulator['v_diff_i']
v_diff_q = json_modulator['v_diff_q']
v_diff_ph = json_modulator['v_diff_ph']
v_diffs = [v_diff_i, v_diff_q, v_diff_ph]

v_pi_values = eval_vpi(lambda_wave_label, vcm_bias, csv_vpi)
v_pi = np.mean(v_pi_values)
# v_L = json_modulator['v_L']   # [V]
# v_R = json_modulator['v_R']    # [V]
# define static v_pi only for classic LiNbO3 MZM, because v_pi in InP MZM varies with v_cm = v_bias
# v_pi = json_modulator['v_pi']    # [V]

'''MZM Params'''
# v_diff = (v_L-v_R)/2    # differential voltage

# v_in_step = json_modulator['v_in_step']    # v_in step

'''Non Ideal MZM Additional Params'''

# For ER values consider 40, 32 and 27 dB values (40 is similar to infinite)
# er_lin = json_modulator['er_lin']
# er = db2lin(er_lin)
# er = db2lin(1000)   # extinction ratio, in linear units, but usually written in dB (using lower values some issues,
# but recovered by MF presence)

# insertion_loss = json_modulator['insertion_loss']
v_off = json_modulator['v_off']     # [V]

'''InP MZM Params'''

# v_bias = - np.pi / v_pi
# v_pi_griffin = - np.pi / v_bias
b = json_modulator['b']    # Phase Non-linearity parameter, between 0.0 (linear) and 0.15 (non-linear)
c = json_modulator['c']    # Transmission Absorption parameter, between 20 (same LiNbO3 behavior)and 4.3

phase_offset = json_modulator['phase_offset']
'''DRIVER PARAMS'''
driver_gain_i = json_modulator['driver_gain_i']
driver_gain_q = json_modulator['driver_gain_q']

'''SOA PARAMS'''
current_in_pre_soa = json_soa['current_in_pre_soa']
current_in_post_soa = json_soa['current_in_post_soa']

# ER OF I AND Q MZMS
# er_i = db2lin(json_modulator['er_i_lin'])
# er_q = db2lin(json_modulator['er_q_lin'])
# BIAS OFFSET OF I AND Q GRIFFIN MZMS
bias_offset_i = json_modulator['bias_offset_i']
bias_offset_q = json_modulator['bias_offset_q']
bias_offset_phase = json_modulator['bias_offset_phase']
# Noise params
noise_flag = True
std = 0.01     # std is the standard deviation of normal distribution, for AWGN (default value is 0.01)
# sim_duration = json_modulator['sim_duration']     # simulation duration
# SNRdB_InP = json_modulator['SNRdB_InP']     # for AWGN channel of InP
# SNRdB_LiNb = json_modulator['SNRdB_LiNb'] # for AWGN channel of LiNb

# symbol generator params
'PRBS'
poldeg = json_ssfm['poldeg']  # polynomial degree for PRBS generator, between 5 and 28
modulation_format = json_spectral['modulation_format']
# modulation_format = modulation_format.upper()
Rs = json_spectral['baud_rate']  # symbol-rate, in Gbaud
Ts = 1/Rs  # symbol period
num_signals = 2**poldeg     # number of transmitted signals

'DAC'
sample_rate = 88e9  # sample/s
dac_len = 10
quant_bits = 8
'Raised cosine Params'
sps = 4
# sps = json_spectral['shaping_filter'][2]   # define samples per symbol (N.B. CHOOSE ONLY EVEN NUMBERS TO GET CORRECT RESULTS)
# N_taps = sps*num_signals
N_taps = 32*num_signals
# length of the filter in samples
beta = json_spectral['shaping_filter'][1]      # roll-off for raised cosine (ideal is 0, actually use 0.09)
samp_f = sps*Rs     # sampling frequency

channel_bw = (1 + beta)*Rs  # [Hz] Frequency 'amplitude' of the primary lobe
delta_f = channel_bw    # channel spacing

# sym_len = r_s**(-1)    # number of samples to represent a signal (bit duration, in ps)

npol = json_ssfm['npol']
if modulation_format == 2:
    # L = number of levels in each dimension
    n_bit = 2
    L = n_bit
    norm_factor = (np.sqrt(2)/2)
    v_drive = 2*v_pi
    evm_factor = 1
elif modulation_format == 3:
    n_bit = 3
    L = n_bit
    norm_factor = 0.75
    v_drive = v_pi
    evm_factor = 9/5
elif modulation_format == 4:
    n_bit = 4
    L = n_bit
    norm_factor = 0.75
    v_drive = v_pi
    evm_factor = 9/5
elif modulation_format == 5:
    n_bit = 5
    L = n_bit
    norm_factor = 0.75
    v_drive = v_pi
    evm_factor = 9/5
elif modulation_format == 6:
    n_bit = 8
    L = n_bit
    norm_factor = 0.75
    v_drive = v_pi
    evm_factor = 9/5

norm_rx = sps
n_bit_nrz = n_bit/2     # number of bits for sqrt(M)-PAM modulation
r_b = Rs*n_bit     # bit-rate, in Gbitps

M = 2**n_bit
# op_freq = 32e9     #GHz
spectral_efficiency = n_bit/(1 + beta)

"Driving voltage signal params"
v_tx_param = 0.5*v_pi/(np.sqrt(M) - 1)
driver_gain = 1
v_tx_param_griffin = 0.5*v_pi/(np.sqrt(M) - 1)

lpf_flag = json_modulator['lpf_flag']
plot_flag = False
classic_flag = False
