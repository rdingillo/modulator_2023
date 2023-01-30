"""iq_mzm.py"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns; sns.set_theme()
import json
from pathlib import Path
from mzm_model.core.elements_ssfm_clean import SSFMLightSource, Splitter, Combiner, MZMSingleElectrode, InP_MZM
from mzm_model.core.modulator_ssfm_params import v_pi, phase_offset, b, c, gamma_1, gamma_2, num_signals, v_tx_param, \
    norm_factor, bias_offset_i, bias_offset_q, plot_flag, classic_flag, npol
from mzm_model.core.math_utils import dbm2lin
from mzm_model.core.utils_ssfm import plot_constellation

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


def inp_modulator(xi, xq, yi, yq):

    xi = np.array(xi)
    xq = np.array(xq)
    yi = np.array(yi)
    yq = np.array(yq)

    with open(json_out, 'w') as outfile:
        json.dump(json_params, outfile, indent=4)

    json_spectral = json_params['spectral_info']

    # retrieve optical input source
    source = SSFMLightSource(json_spectral)

    # evaluate optical input power in dBm
    input_power_dbm = source.input_power        # dBm
    """TODO: QUI APPLICARE IL PRE-SOA POWER"""
    input_power = dbm2lin(input_power_dbm)       # Watt
    source.out_field = source.calculate_optical_input_power(input_power_dbm)
    input_field = source.out_field

    # define TX amplitude parameter in time (for the field)
    k_tx = (np.sqrt(input_power)/(np.sqrt(2)))*(np.pi/(2*v_pi))

    # evaluate electric fields in splitter
    splitter = Splitter(source.input_power, source.out_field)
    splitter_fields = splitter.calculate_arms_input_fields()
    # save p and q field for the 2 arms (I, Q)
    p_field = splitter_fields[0]   # [mV/m]
    q_field = splitter_fields[1]   # [mV/m]
    splitter.A_in_field = p_field
    splitter.B_in_field = q_field

    i_sig = xi
    q_sig = xq

    p_sign = i_sig * v_tx_param
    q_sign = q_sig * v_tx_param

    # v_bias = -(np.pi/v_pi)*cpe_griffin
    v_bias = -(np.pi/v_pi)
    # inserting *np.arcsin(np.sqrt(1/(1+er_i))) we get the dependence on ER
    i_Griffin_list = [GriffinMZM(v_bias + bias_offset_i,
                                 (v_p_tem)/2 + 0.5*v_pi + bias_offset_i/2,
                                 -0.5*v_pi + (-v_p_tem)/2 - bias_offset_i/2, gamma_1, gamma_2,
                                 phase_offset, b, c, er_i) for v_p_tem in p_sign]
    i_Griffin_field_list = [i_Griffin.griffin_eo_tf_field()*p_field for i_Griffin in i_Griffin_list]
    q_Griffin_list = [GriffinMZM(v_bias + bias_offset_q,
                                 0.5*v_pi + (v_q_tem)/2 + bias_offset_q/2,
                                 -0.5*v_pi + (-v_q_tem)/2 - bias_offset_q/2, gamma_1, gamma_2,
                                 phase_offset, b, c, er_q) for v_q_tem in q_sign]

    # for Q field evaluation, apply Phase Modulator effect rotating it, multiplying it for -1j
    q_Griffin_field_list = [q_Griffin.griffin_eo_tf_field()*q_field*(-1)*(1j) for q_Griffin
                            in q_Griffin_list]


    # combiner output field Griffin
    combiner_Griffin_list = [Combiner(i_Griffin_field_list[i], q_Griffin_field_list[i]) for i
                             in range(len(i_Griffin_field_list))]
    out_fields_combiner_Griffin_list = [combiner.combiner_out_field(combiner.in_A, combiner.in_B) for combiner
                                        in combiner_Griffin_list]

    out_fields_samples_Griffin = [out_fields_combiner_Griffin_list for i in range(num_signals)]

    # collection and normalization of constellation points
    constellation_samples_Griffin = np.array(out_fields_combiner_Griffin_list)/\
                                    np.sqrt(np.mean((np.array(np.abs(out_fields_combiner_Griffin_list)))**2))

    griffin_tx_const = constellation_samples_Griffin/\
                       (np.sqrt(input_power))/(v_tx_param/v_pi)/norm_factor


    griffin_df = pd.DataFrame(constellation_samples_Griffin)
    if classic_flag:
        classical_df = pd.DataFrame(np.array(constellation_samples_classic[0]))
        linb_real = classical_df.values.real
        linb_imag = classical_df.values.imag

    griffin_real = griffin_df.values.real
    griffin_imag = griffin_df.values.imag
    # np.savetxt(output_csv_xi, griffin_real, delimiter=',')
    # np.savetxt(output_csv_xq, griffin_imag, delimiter=',')
    # xi_dict = {'xi': griffin_real}
    # xq_dict = {'xq': griffin_imag}
    # griffin_xi = savemat('xi.mat', xi_dict)
    # griffin_xq = savemat('xq.mat', xq_dict)
    griffin_xi = griffin_real
    griffin_xq = griffin_imag

    if npol == 2:
        # i_sig = mod_sign_df['yi']
        # q_sig = mod_sign_df['yq']

        i_sig = yi
        q_sig = yq

        p_sign = i_sig * v_tx_param
        q_sign = q_sig * v_tx_param

        i_Griffin_list = [GriffinMZM(v_bias + bias_offset_i,
                                     (v_p_tem) / 2 + 0.5 * v_pi + bias_offset_i / 2,
                                     -0.5 * v_pi + (-v_p_tem) / 2 - bias_offset_i / 2, gamma_1, gamma_2,
                                     phase_offset, b, c, er_i) for v_p_tem in p_sign]
        i_Griffin_field_list = [i_Griffin.griffin_eo_tf_field() * p_field for i_Griffin in i_Griffin_list]
        q_Griffin_list = [GriffinMZM(v_bias + bias_offset_q,
                                     0.5 * v_pi + (v_q_tem) / 2 + bias_offset_q / 2,
                                     -0.5 * v_pi + (-v_q_tem) / 2 - bias_offset_q / 2, gamma_1, gamma_2,
                                     phase_offset, b, c, er_q) for v_q_tem in q_sign]

        # for Q field evaluation, apply Phase Modulator effect rotating it, multiplying it for -1j
        q_Griffin_field_list = [q_Griffin.griffin_eo_tf_field() * q_field * (-1) * (1j) for q_Griffin
                                in q_Griffin_list]

        # combiner output field Griffin
        combiner_Griffin_list = [Combiner(i_Griffin_field_list[i], q_Griffin_field_list[i]) for i
                                 in range(len(i_Griffin_field_list))]
        out_fields_combiner_Griffin_list = [combiner.combiner_out_field(combiner.in_A, combiner.in_B) for combiner
                                            in combiner_Griffin_list]

        out_fields_samples_Griffin = [out_fields_combiner_Griffin_list for i in range(num_signals)]

        # collection and normalization of constellation points
        constellation_samples_Griffin = np.array(out_fields_combiner_Griffin_list) / \
                                        np.sqrt(np.mean((np.array(np.abs(out_fields_combiner_Griffin_list))) ** 2))

        griffin_tx_const = constellation_samples_Griffin / \
                           (np.sqrt(input_power)) / (v_tx_param / v_pi) / norm_factor

        griffin_df = pd.DataFrame(constellation_samples_Griffin)
        if classic_flag:
            classical_df = pd.DataFrame(np.array(constellation_samples_classic[0]))
            linb_real = classical_df.values.real
            linb_imag = classical_df.values.imag

        griffin_real = griffin_df.values.real
        griffin_imag = griffin_df.values.imag
        # np.savetxt(output_csv_yi, griffin_real, delimiter=',')
        # np.savetxt(output_csv_yq, griffin_imag, delimiter=',')
        # yi_dict = {'yi': griffin_real}
        # yq_dict = {'yq': griffin_imag}
        # griffin_yi = savemat('yi.mat', yi_dict)
        # griffin_yq = savemat('yq.mat', yq_dict)

        griffin_yi = griffin_real
        griffin_yq = griffin_imag

    else:
        griffin_real = np.zeros(len(griffin_real))
        griffin_imag = np.zeros(len(griffin_imag))
        # np.savetxt(output_csv_yi, griffin_real, delimiter=',')
        # np.savetxt(output_csv_yq, griffin_imag, delimiter=',')
        # yi_dict = {'yi': griffin_real}
        # yq_dict = {'yq': griffin_imag}
        # griffin_yi = savemat('yi.mat', yi_dict)
        # griffin_yq = savemat('yq.mat', yq_dict)
        griffin_yi = griffin_real
        griffin_yq = griffin_imag
    if plot_flag:
        plot_constellation(constellation_samples_Griffin, 'InP MZM Constellation at RX side')

        plt.show(block=True)

    # print("--- Elapsed Time in Python Module: %s seconds ---" % (time.time() - start_time))

    return {'xi': np.array(griffin_xi), 'xq': np.array(griffin_xq), 'yi': np.array(griffin_yi), 'yq': np.array(griffin_yq)}
