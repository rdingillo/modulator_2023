import numpy as np
import pandas as pd
import json
from pathlib import Path
from mzm_model.core.elements_ssfm_clean import SSFMLightSource
from mzm_model.core.modulator_ssfm_params import json_spectral, json_soa, frequency, current_in_pre_soa, current_in_post_soa
root = Path('~/PycharmProjects/modulator_2023/').expanduser()
json_source = root.parent / 'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'

# json_file = json_source / 'ssi_cut0_pump0.json'
# input_folder = root / 'mzm_model/resources'
# json_params = json.load(open(json_file, 'r'))
# input_file = input_folder / 'FreqResponse_belmonte.csv'
# folder_results = root / 'mzm_model' / 'results'
# json_out = folder_results / 'json_params.json'
#
# with open(json_out, 'w') as outfile:
#     json.dump(json_params, outfile, indent=4)

# json_spectral = json_params['spectral_info']
# json_modulator = json_params['modulator_params']
# retrieve optical input source
source = SSFMLightSource(json_spectral)

# evaluate optical input power in dBm
laser_input_power_dbm = source.input_power

'''Input data'''
# power_in_pre_soa = np.arange(8, 20)
power_in = laser_input_power_dbm  # dBm, typically 13 dBm

# mod_loss_set = np.arange(6, 15)                          # dB
pbo_loss = json_soa["pbo_loss"]  # dB, previously called modulation loss
quad_loss = json_soa["quad_loss"]  # dB, quadrature loss
loss_coupling = json_soa["loss_coupling"]  # dB, loss of coupler
loss_split_xy = json_soa["loss_split_xy"]  # dB, loss of polarization splitter
loss_wg_pre_soa = json_soa["loss_wg_pre_soa"]  # dB, loss of waveguide BEFORE pre-SOA
loss_wg_post_soa = json_soa["loss_wg_post_soa"]  # dB, loss of waveguide BEFORE post-SOA
# loss_wvl_dep_wg_data_post_soa = np.array(
#     [1.7, 1.3, 1.2, 1.4, 1.9, 2.8])  # wavelength dependent losses in wg after post soa


# evaluate loss BEFORE preSOA, which is a sort of Insertion LOSS
pre_soa_loss = loss_wg_pre_soa + loss_split_xy + loss_coupling  # dB, typically 7 dB

power_in_pre_soa = power_in - pre_soa_loss  # effective optical input power at pre-SOA input

channel = frequency*1e-12  # THz

f_min = 191.1
f_max = 196.1

beta0 = 10 ** 0.1 - 1

A0 = np.array([2.26e-2, -3.448, -6072.9])
A1 = np.array([-1.2241e-4, 0.019023, 63.155])
A2 = np.array([0, 0, -0.16421])

B0 = np.array([1.038 * 1e-2, -1.844, 153.48])
B1 = np.array([-5.626 * 1e-5, 1.02 * 1e-2, -0.7917])

# channels_expand = np.linspace(f_min, f_max, 601)  # choose multiple of 6 + 1 to get the 6 main channels

loss_wvl_dep_wg_data_post_soa = 0.1543*(channel**2) - 59.543*channel + 5745.5
post_soa_loss = pbo_loss + quad_loss + loss_wg_post_soa + loss_wvl_dep_wg_data_post_soa

# Pre and Post SOA params
a0 = A2[2] * channel ** 2 + A1[2] * channel + A0[2]
a1 = A1[1] * channel + A0[1]
a2 = A1[0] * channel + A0[0]

b0 = B1[2] * channel + B0[2]
b1 = B1[1] * channel + B0[1]
b2 = B1[0] * channel + B0[0]


def evaluate_gs(a0, a1, a2, current_in):
    # gs_df = pd.DataFrame(columns=channels_expand)
    # gs_df.insert(loc=0, column='Current_In', value=current_in)
    gs = a2 * current_in ** 2 + a1 * current_in + a0
    return gs


def evaluate_p1db_pre(b0, b1, b2, current_in):
    p1db = b2 * (current_in*0.7) ** 2 + b1 * (current_in*0.8) + b0
    return p1db


def evaluate_p1db_post(b0, b1, b2, current_in):
    p1db = b2 * current_in ** 2 + b1 * current_in + b0
    return p1db


def evaluate_beta(P1dB, Gs):
    beta = beta0 / 10 ** ((P1dB - Gs) / 10)
    return beta


def evaluate_output_power(p_in, gs, beta):
    p_out = p_in + gs + 10 * np.log10(1 / (1 + beta * 10 ** (p_in / 10)))
    return p_out


def evaluate_osnr_db(p_in, channel):
    osnr = 60.95 + p_in - 1 - 0.019712*(channel**2) + 7.297*channel - 680.3
    return osnr


gs_pre_soa = 1.2*evaluate_gs(a0, a1, a2, current_in_pre_soa)
p1db_pre_soa = evaluate_p1db_pre(b0, b1, b2, current_in_pre_soa)
beta_pre_soa = evaluate_beta(p1db_pre_soa, gs_pre_soa)

p_out_pre_soa = evaluate_output_power(power_in_pre_soa, gs_pre_soa, beta_pre_soa)


def pre_soa_out_power():
    return p_out_pre_soa

gs_post_soa = evaluate_gs(a0, a1, a2, current_in_post_soa)
p1db_post_soa = evaluate_p1db_post(b0, b1, b2, current_in_post_soa)
beta_post_soa = evaluate_beta(p1db_post_soa, gs_post_soa)

power_in_post_soa = p_out_pre_soa - post_soa_loss

p_out_post_soa = evaluate_output_power(power_in_post_soa, gs_post_soa, beta_post_soa)
osnr_db = evaluate_osnr_db(power_in_post_soa, channel)
print()


def post_soa_out_power(p_in_dbm):
    power_in = p_in_dbm - post_soa_loss
    power_out = evaluate_output_power(power_in, gs_post_soa, beta_post_soa)
    return power_out
