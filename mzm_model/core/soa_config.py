import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
import csv
from pathlib import Path
from mzm_model.core.elements_ssfm_clean import SSFMLightSource

root = Path('~/PycharmProjects/modulator_2023/').expanduser()
json_source = root.parent / 'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'

json_file = json_source / 'ssi_cut0_pump0.json'
input_folder = root / 'mzm_model/resources'
json_params = json.load(open(json_file, 'r'))
input_file = input_folder / 'FreqResponse_belmonte.csv'
folder_results = root / 'mzm_model' / 'results'
json_out = folder_results / 'json_params.json'

with open(json_out, 'w') as outfile:
    json.dump(json_params, outfile, indent=4)

json_spectral = json_params['spectral_info']

# retrieve optical input source
source = SSFMLightSource(json_spectral)

# evaluate optical input power in dBm
laser_input_power_dbm = source.input_power

# matplotlib.use('Qt5Agg')

'''Input data'''
# power_in_pre_soa = np.arange(8, 20)
power_in = laser_input_power_dbm  # dBm, typically 13 dBm
current_in_pre_soa = np.arange(0, 201)  # mA
# mod_loss_set = np.arange(6, 15)                          # dB
pbo_loss = 10  # dB, previously called modulation loss
quad_loss = 3  # dB, quadrature loss
loss_coupling = 3  # dB, loss of coupler
loss_split_xy = 3  # dB, loss of polarization splitter
loss_wg_pre_soa = 1  # dB, loss of waveguide BEFORE pre-SOA
loss_wg_post_soa = 3  # dB, loss of waveguide BEFORE post-SOA
# loss_wvl_dep_wg_data_post_soa = np.array(
#     [1.7, 1.3, 1.2, 1.4, 1.9, 2.8])  # wavelength dependent losses in wg after post soa


# evaluate loss BEFORE preSOA, which is a sort of Insertion LOSS
pre_soa_loss = loss_wg_pre_soa + loss_split_xy + loss_coupling  # dB, typically 7 dB

power_in_pre_soa = power_in - pre_soa_loss  # effective optical input power at pre-SOA input

current_in_post_soa = np.arange(0, 201)  # mA  # mA
channels = np.arange(191.1, 197.1, 1)  # THz

f_min = 193.5
f_max = 193.5

beta0 = 10 ** 0.1 - 1

A0 = np.array([2.26e-2, -3.448, -6072.9])
A1 = np.array([-1.2241e-4, 0.019023, 63.155])
A2 = np.array([0, 0, -0.16421])

B0 = np.array([1.038 * 1e-2, -1.844, 153.48])
B1 = np.array([-5.626 * 1e-5, 1.02 * 1e-2, -0.7917])

channels_expand = np.linspace(f_min, f_max, 1)  # choose multiple of 6 + 1 to get the 6 main channels

loss_wvl_dep_wg_data_post_soa = 0.1543*(channels_expand**2) - 59.543*channels_expand + 5745.5
post_soa_loss = pbo_loss + quad_loss + loss_wg_post_soa + loss_wvl_dep_wg_data_post_soa

# Pre and Post SOA params
a0 = A2[2] * channels_expand ** 2 + A1[2] * channels_expand + A0[2]
a1 = A1[1] * channels_expand + A0[1]
a2 = A1[0] * channels_expand + A0[0]

b0 = B1[2] * channels_expand + B0[2]
b1 = B1[1] * channels_expand + B0[1]
b2 = B1[0] * channels_expand + B0[0]

# plt.figure()
# plt.plot(channels_expand, a0, label='a0')
# plt.figure()
# plt.plot(channels_expand, a1, label='a1')
# plt.figure()
# plt.plot(channels_expand, a2, label='a2')
#
# plt.figure()
# plt.plot(channels_expand, b0, label='b0')
# plt.figure()
# plt.plot(channels_expand, b1, label='b1')
# plt.figure()
# plt.plot(channels_expand, b2, label='b2')

plt.show(block=False)


# a0 = np.array([-7.99*1e-1, -5.69*1e-1, -6.68*1e-1, -1.1, -1.85, -2.93])
# a1 = np.array([1.87*1e-1, 2.06*1e-1, 2.25*1e-1, 2.44*1e-1, 2.63*1e-1, 2.82*1e-1])
# a2 = np.array([-7.93*1e-4, -9.15*1e-4, -1.04*1e-3, -1.16*1e-3, -1.28*1e-3, -1.4*1e-3])
#
# b0 = np.array([2.19, 1.39, 6.03*1e-1, -1.89*1e-1, -9.81*1e-1, -1.77])
# b1 = np.array([1.05*1e-1, 1.15*1e-1, 1.26*1e-1, 1.36*1e-1, 1.46*1e-1, 1.56*1e-1])
# b2 = np.array([-3.71*1e-4, -4.28*1e-4, -4.84*1e-4, -5.40*1e-4, -5.96*1e-4, -6.53*1e-4])


def evaluate_gs(a0, a1, a2, current_in):
    gs_list = []
    gs_df = pd.DataFrame(columns=channels_expand)
    # gs_df.insert(loc=0, column='Current_In', value=current_in)
    for i in current_in:
        gs = a2 * (i) ** 2 + a1 * i + a0
        # gs_df.loc[i] = gs
        gs_list.append(gs)
    return gs_list


def evaluate_p1db_pre(b0, b1, b2, current_in):
    p1db_list = []
    for i in current_in:
        p1db = b2 * (i*0.7) ** 2 + b1 * (i*0.8) + b0
        p1db_list.append(p1db)
    return p1db_list


def evaluate_p1db_post(b0, b1, b2, current_in):
    p1db_list = []
    for i in current_in:
        p1db = b2 * i ** 2 + b1 * i + b0
        p1db_list.append(p1db)
    return p1db_list


def evaluate_beta(P1dB, Gs):
    beta_list = []
    for i in range(len(P1dB)):
        beta = beta0 / 10 ** ((P1dB[i] - Gs[i]) / 10)
        beta_list.append(beta)
    return beta_list


def evaluate_output_power_pre(p_in, gs, beta):
    p_out_list = []
    for i in range(len(gs)):
        p_out = p_in + gs[i] + 10 * np.log10(1 / (1 + beta[i] * 10 ** (p_in / 10)))
        p_out_list.append(p_out)
    return p_out_list

def evaluate_output_power_post(p_in, gs, beta):
    p_out_list = []
    for i in range(len(gs)):
        p_out = p_in[i] + gs[i] + 10 * np.log10(1 / (1 + beta[i] * 10 ** (p_in[i] / 10)))
        p_out_list.append(p_out)
    return p_out_list

def evaluate_osnr_db(p_in, channels):
    osnr_list = []
    for i in range(len(p_in)):
        osnr = 60.95 + p_in[i] - 1 - 0.019712*(channels[i]**2) + 7.297*channels[i] - 680.3
        osnr_list.append(osnr)
    return np.array(osnr_list)


gs_pre_soa = 1.2*np.array(evaluate_gs(a0, a1, a2, current_in_pre_soa))
p1db_pre_soa = np.array(evaluate_p1db_pre(b0, b1, b2, current_in_pre_soa))
beta_pre_soa = np.array(evaluate_beta(p1db_pre_soa, gs_pre_soa))

p_out_pre_soa = np.array(evaluate_output_power_pre(power_in_pre_soa, gs_pre_soa, beta_pre_soa))

plt.figure()
plt.plot(current_in_pre_soa, p_out_pre_soa)
plt.show(block=True)

gs_post_soa = np.array(evaluate_gs(a0, a1, a2, current_in_post_soa))
p1db_post_soa = np.array(evaluate_p1db_post(b0, b1, b2, current_in_post_soa))
beta_post_soa = np.array(evaluate_beta(p1db_post_soa, gs_post_soa))

power_in_post_soa = p_out_pre_soa - post_soa_loss

p_out_post_soa = np.array(evaluate_output_power_post(power_in_post_soa, gs_post_soa, beta_post_soa))

plt.figure()
plt.plot(current_in_post_soa, p_out_post_soa)
plt.show(block=True)

osnr_flag = 0

if osnr_flag == 1:
    osnr_db = evaluate_osnr_db(power_in_post_soa, channels)
print()
