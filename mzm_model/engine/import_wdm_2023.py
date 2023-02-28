"""iq_mzm.py"""

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns; sns.set_theme()
import json
from pathlib import Path
from mzm_model.core.elements_ssfm_clean import SSFMLightSource, InP_MZM, Combiner
from mzm_model.core.modulator_ssfm_params import v_pi_values, v_diff_i, v_diff_q, b, c, gamma_1, gamma_2, \
    bias_offset_i, bias_offset_q, plot_flag, npol, vcm_bias, vcm_phase, lambda_wave, v_diff_ph, driver_gain_i, \
    driver_gain_q
from mzm_model.core.math_utils import dbm2lin
# from mzm_model.core.utils_ssfm import plot_constellation
from mzm_model.core.soa_config_singlechannel import pre_soa_out_power, post_soa_out_power
from mzm_model.core.science_utils_ssfm import evaluate_field, griffin_phase_electrode, field_to_power_dbm

start_time = time.time()

root = Path('~/PycharmProjects/modulator_2023/').expanduser()
json_source = root.parent/'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'

json_file = json_source/'ssi_cut0_pump0_2023.json'
input_folder = root/'mzm_model/resources'
json_params = json.load(open(json_file, 'r'))
input_file = input_folder/'FreqResponse_128g.csv'
folder_results = root/'mzm_model'/'results'
json_out = folder_results/'json_params.json'

wdm_folder = json_source.parent

with open(json_out, 'w') as outfile:
    json.dump(json_params, outfile, indent=4)

json_spectral = json_params['spectral_info']


# for now work just for bias and upload RF values in single polarization
def bias_mzm():
    # retrieve optical input source
    source = SSFMLightSource(json_spectral)

    # evaluate optical input power in dBm
    input_power_dbm = source.input_power  # dBm
    input_power = dbm2lin(input_power_dbm)  # Watt
    source.out_field = source.calculate_optical_input_power(input_power_dbm)
    input_field = source.out_field

    """NB ARRIVATI A QUESTO PUNTO SIAMO AL PRIMO SPLITTER DOVE SI DIVIDONO X E Y"""
    """ QUI VANNO INSERITI I VALORI DEI PRE-SOA"""
    pol_power_dbm = pre_soa_out_power()
    xi_power_dbm = pol_power_dbm - 3
    xq_power_dbm = xi_power_dbm

    xi_input_field = evaluate_field(xi_power_dbm)
    xq_input_field = evaluate_field(xq_power_dbm)

    # For MZi values, consider that MZ1 = YQ (0), MZ2 = YI (1), MZ3 = XQ (2), MZ4 = XI (3)
    vpi_phase = np.mean([v_pi_values[2], v_pi_values[3]])
    # mz_xi = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[3], v_diff_i, gamma_1, gamma_2, b, c, driver_gain_i,
    #                 np.array([0]))
    # mz_xq = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[2], v_diff_q, gamma_1, gamma_2, b, c, driver_gain_q,
    #                 np.array([0]))

    mz_xi = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[3] , v_pi_values[3], gamma_1, gamma_2, b, c, driver_gain_i,
                    np.array([0]))
    mz_xq = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[2], v_pi_values[2], gamma_1, gamma_2, b, c, driver_gain_q,
                    np.array([0]))

    # evaluate the electric fields of MZi at output
    mz_xi_field = mz_xi.griffin_eo_tf_field() * xi_input_field
    mz_xq_field = mz_xq.griffin_eo_tf_field() * xq_input_field

    mz_xi_power = field_to_power_dbm(mz_xi_field)  # [dBm]
    mz_xq_power = field_to_power_dbm(mz_xq_field)  # [dBm]

    return float(mz_xi.vpi), float(mz_xq.vpi), mz_xi_power, mz_xq_power


def inp_modulator(xi, xq, yi, yq):

    xi = np.array(xi)
    xq = np.array(xq)
    yi = np.array(yi)
    yq = np.array(yq)

    with open(json_out, 'w') as outfile:
        json.dump(json_params, outfile, indent=4)

    json_spectral = json_params['spectral_info']

    # print("--- Elapsed Time in Python Module: %s seconds ---" % (time.time() - start_time))
    return 0
    # return {'xi': np.array(griffin_xi), 'xq': np.array(griffin_xq), 'yi': np.array(griffin_yi), 'yq': np.array(griffin_yq)}


def update_modulator(rf_i, rf_q):
    # retrieve optical input source
    source = SSFMLightSource(json_spectral)

    # evaluate optical input power in dBm
    input_power_dbm = source.input_power  # dBm
    input_power = dbm2lin(input_power_dbm)  # Watt
    source.out_field = source.calculate_optical_input_power(input_power_dbm)
    input_field = source.out_field

    """NB ARRIVATI A QUESTO PUNTO SIAMO AL PRIMO SPLITTER DOVE SI DIVIDONO X E Y"""
    """ QUI VANNO INSERITI I VALORI DEI PRE-SOA"""
    pol_power_dbm = pre_soa_out_power()
    xi_power_dbm = pol_power_dbm - 3
    xq_power_dbm = xi_power_dbm

    xi_input_field = evaluate_field(xi_power_dbm)
    xq_input_field = evaluate_field(xq_power_dbm)

    # For MZi values, consider that MZ1 = YQ (0), MZ2 = YI (1), MZ3 = XQ (2), MZ4 = XI (3)
    vpi_phase = np.mean([v_pi_values[2], v_pi_values[3]])
    # mz_xi = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[3], v_diff_i, gamma_1, gamma_2, b, c, driver_gain_i,
    #                 np.array([rf_i]))
    # mz_xq = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[2], v_diff_q, gamma_1, gamma_2, b, c, driver_gain_q,
    #                 np.array([rf_q]))
    mz_xi = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[3], v_pi_values[3], gamma_1, gamma_2, b, c, driver_gain_i,
                    np.array([rf_i]))
    mz_xq = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[2], v_pi_values[2], gamma_1, gamma_2, b, c, driver_gain_q,
                    np.array([rf_q]))
    mz_phase = InP_MZM(lambda_wave, vcm_phase, vcm_bias, vpi_phase, v_diff_ph, gamma_1, gamma_2, b, c,
                       np.array([0]), np.array([0]))

    # evaluate the electric fields of MZi at output
    mz_xi_field = mz_xi.griffin_eo_tf_field() * xi_input_field
    mz_xq_field = mz_xq.griffin_eo_tf_field() * xq_input_field

    mz_xi_power = field_to_power_dbm(mz_xi_field)  # [dBm]
    mz_xq_power = field_to_power_dbm(mz_xq_field)  # [dBm]

    # Apply the Phase Offset considering the vdiff_phase to be applied to the fields
    # At first evaluate the phases
    phases_shifts = griffin_phase_electrode(mz_phase.vl, mz_phase.vr, v_pi_values[3], v_pi_values[2], b)
    # Apply Phase Electrode phases independently to each XI and XQ arm
    # mz_xi_ph_field = mz_xi_field * np.exp(phases_shifts[0] * 1j)
    mz_xi_ph_field = mz_xi_field
    # mz_xq_ph_field = mz_xq_field * np.exp(phases_shifts[1] * 1j)
    mz_xq_ph_field = mz_xq_field
    # combiner output field InP MZM Single Pol.
    combiner_single_pol = Combiner(mz_xi_ph_field, mz_xq_ph_field)
    out_fields_combiner_single_pol = combiner_single_pol.combiner_out_field(combiner_single_pol.in_A,
                                                                            combiner_single_pol.in_B)
    # collection and normalization of constellation points
    constellation_samples_xi = np.array(out_fields_combiner_single_pol) / \
                                    np.sqrt(np.mean((np.array(np.abs(out_fields_combiner_single_pol))) ** 2))
    inp_x_df = pd.DataFrame(constellation_samples_xi)
    inp_xi = inp_x_df.values.real
    inp_xq = inp_x_df.values.imag
    inp_x_fields = inp_x_df.values
    return mz_xi, mz_xq, np.array(inp_xi), np.array(inp_xq), np.array(inp_x_fields)

# import random
# prova_arr1 = [round(random.uniform(-v_pi_values[3], v_pi_values[3]), 2)for i in range(0, 2048)]
# prova_arr2 = [round(random.uniform(-v_pi_values[3], v_pi_values[3]), 2)for i in range(0, 2048)]
#
# rf_mz = update_modulator(prova_arr1, prova_arr2)
# from mzm_model.core.utils import plot_constellation
# import matplotlib.pyplot as plt
# plot_constellation(rf_mz[4], 'InP MZM Constellation at RX side')
# plt.show(block=True)
# print()