"""iq_mzm.py"""
import math

import matplotlib

import numpy as np
import matplotlib.pyplot as plt
# from scipy.io import savemat, loadmat
import pandas as pd
import time
import seaborn as sns; sns.set_theme()
import json
# import csv
from pathlib import Path
from mzm_model.core.elements_ssfm import SSFMLightSource, Splitter, Combiner, MZMSingleElectrode, GriffinMZM
from mzm_model.core.modulator_ssfm_params import v_pi, insertion_loss, phase_offset, b, c,\
    gamma_1, gamma_2, num_signals, std, sps, Ts, v_tx_param, noise_flag, norm_factor, er_i, er_q,\
    bias_offset_i, bias_offset_q, plot_flag, classic_flag, npol
from mzm_model.core.math_utils import dbm2lin
from mzm_model.core.utils_ssfm import plot_constellation

start_time = time.time()

# matplotlib.use('Qt5Agg')

root = Path('~/PycharmProjects/optical_wideband_iq_modulator/').expanduser()
json_source = root.parent/'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'
# root = Path('~/github/optical_wideband_iq_modulator/').expanduser()
# json_source = root.parent/'ssfm/spectral_information'


# dirname = Path(os.path.dirname(__file__))
# root = dirname.parent.parent
# json_source = root.parent/'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'

json_file = json_source/'ssi_cut0_pump0.json'
input_folder = root/'mzm_model/resources'
json_params = json.load(open(json_file, 'r'))
input_file = input_folder/'FreqResponse_128g.csv'
folder_results = root/'mzm_model'/'results'
json_out = folder_results/'json_params.json'
# output_csv_xi = folder_results/"out_tx_time_xi.csv"
# output_csv_xq = folder_results/"out_tx_time_xq.csv"
# output_csv_yi = folder_results/"out_tx_time_yi.csv"
# output_csv_yq = folder_results/"out_tx_time_yq.csv"

wdm_folder = json_source.parent


def inp_modulator(xi, xq, yi, yq):
    # xi_mat = loadmat('xi_pre.mat')['xi_pre'][:, 0]
    # xq_mat = loadmat('xq_pre.mat')['xq_pre'][:, 0]
    # yi_mat = loadmat('yi_pre.mat')['yi_pre'][:, 0]
    # yq_mat = loadmat('yq_pre.mat')['yq_mat'][:, 0]

    xi = np.array(xi)
    xq = np.array(xq)
    yi = np.array(yi)
    yq = np.array(yq)


    # xi = wdm_folder/'xi.csv'
    # yi = wdm_folder/'yi.csv'
    # xq = wdm_folder/'xq.csv'
    # yq = wdm_folder/'yq.csv'
    # list_sign = [xi, xq, yi, yq]
    # file = open(xi, "r")
    # csv_reader = csv.reader(file, delimiter=',')
    # data_xi = np.array(list(csv_reader)).astype(float)
    # mod_sign_df = pd.DataFrame(columns=['xi','xq','yi','yq'], index=range(len(data_xi)))
    #
    # for filename in list_sign:
    #
    #     file = open(filename, "r")
    #     csv_reader = csv.reader(file, delimiter=',')
    #     data = np.array(list(csv_reader)).astype(float)
    #     if not math.isnan(data[0][0]):
    #         pass
    #     else:
    #         data = np.zeros(len(data_xi))
    #     mod_sign_df[filename.stem] = data


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

    # initialize two lists, one for i_field and one for q_field
    i_field_list = []
    q_field_list = []
    out_iq_list = []
    vp_list = []
    vq_list = []
    vp_norm_list = []
    vq_norm_list = []
    # add some random AWG noise to signal
    # 0 is the mean of the normal distribution I am choosing for
    # std is the standard deviation of normal distribution
    # num_signals is the number of signals where I want to add AWGN
    # if noise_flag == True:
    #     noise_i = np.random.normal(0, std, 2*num_signals*sps-1)
    #     noise_q = np.random.normal(0, std, 2*num_signals*sps-1)
    # else:
    #     noise_i = 0
    #     noise_q = 0

    # define time axis for the 'square wave'
    # t1 = np.linspace(0, num_signals*Ts*1e9, mod_sign_df.shape[0])

    # i_sig = mod_sign_df['xi']
    # q_sig = mod_sign_df['xq']

    i_sig = xi
    q_sig = xq

    # if plot_flag:
    #     fig, axs = plt.subplots(2, 1, figsize=(9, 10))
    #     fig.suptitle('I and Q samples')
    #     axs[0].set_title('I samples')
    #     axs[0].set(xlabel='Time [ns] ', ylabel='Bit values')
    #     axs[0].plot(t1, i_sig, label='I')
    #     axs[1].set_title('Q samples')
    #     axs[1].set(xlabel='Time [ns] ', ylabel='Bit values')
    #     axs[1].plot(t1, q_sig, label='Q')
    #     plt.grid(True)
    #
    # if lpf_flag:
    # # if 1==1:
    #     input_power_wave = input_power_dbm * np.sin(2 * np.pi * frequency * t1)
    #     power_wave_freq = (np.fft.fft(input_power_wave))
    #     magn_pow = np.abs(power_wave_freq)
    #     # Create our root-raised-cosine (RRC) filter
    #     # rrcos params:
    #     # N = N_taps = length of the filter in samples = samples per symbol * number of transmitted signals
    #     # beta = roll-off
    #     # Ts = symbol period (in seconds)
    #     # samp_f = sampling frequency = Rs*sps
    #     # rrcos_filter = flt.rrcosfilter(N_taps, beta, Ts, samp_f)
    #     rrcos_filter = flt.rrcosfilter(mod_sign_df['xi'].size, beta, Ts, samp_f)
    #
    #     # Perform FFT of RRCOS filter and retrieve the frequencies associated
    #     rrcos_fft = np.fft.fft(rrcos_filter[1])
    #     rrcos_fft_mag = 20*np.log10(np.abs(rrcos_fft))
    #     rrcos_freq = np.fft.fftfreq(n=rrcos_filter[1].size, d=1/samp_f)
    #
    #     # Everytime perform an FFT, perform the shift in order to make first FFT of negative freqs
    #     # and then of the positive ones, contrary to default approach of numpy FFT
    #     rrcos_fft = np.fft.fftshift(rrcos_fft)
    #     rrcos_freq = np.fft.fftshift(rrcos_freq)
    #
    #     # here we need to create the custom LPF generated by measured values and apply it at the input of the modulator
    #     lpf = pd.read_csv(input_file, sep=";", usecols= ['Freq [GHz]','dB20(S21diff) norm2'], index_col=False)
    #     lpf['Freq [GHz]'] = lpf['Freq [GHz]']*1e9
    #
    #     freqs_filter = lpf['Freq [GHz]']
    #     pow_filter = lpf['dB20(S21diff) norm2']
    #     # Add negative frequencies to filter to perform multiplication with signal FFT later
    #     freqs_no_zero = freqs_filter[1::]
    #     pow_no_zero = pow_filter[1::]
    #     negative_freqs = freqs_no_zero[::-1]*(-1)
    #     negative_pows = pow_no_zero[::-1]
    #     tot_freqs = negative_freqs.append(freqs_filter)
    #     tot_pows = negative_pows.append(pow_filter)
    #
    #     # Take the amplitude values to perform FFT
    #     magn_filter = control.db2mag(tot_pows)
    #     # magn_db = 10*np.log10(magn_filter)
    #     lpf_dict_magn = {'Frequency': tot_freqs, 'Amplitude': magn_filter}
    #
    #     lpf_mag_df = pd.DataFrame(lpf_dict_magn)
    #
    #     # retrieve the frequency values of RRCOS associated to LPF frequencies to obtain a new set of
    #     # frequencies useful to interpolate data of the custom LPF
    #     boolean_array = np.logical_and(rrcos_freq >= min(np.array(tot_freqs)), rrcos_freq <= max(np.array(tot_freqs)))
    #     new_array = rrcos_freq[np.where(boolean_array)]
    #     # take the values for interpolation
    #     x = rrcos_fft[np.where(boolean_array)]
    #     y = new_array
    #     f = scipy.interpolate.griddata(np.array(tot_freqs), np.array(lpf_mag_df['Amplitude']), new_array)
    #     # create the lists related to empty values of frequencies, and give them the minimum value. Then insert in df
    #     lost_freqs = np.array([x for x in rrcos_freq if x not in new_array])
    #     min_amp_array = np.array([min(f) for x in range(len(lost_freqs))])
    #     lpf_dict_magn_interp = {'Freqs': y, 'Amplitude': f}
    #
    #     lpf_mag_df_interp = pd.DataFrame(lpf_dict_magn_interp)
    #
    #     fill_values_df_dict = {'Freqs': lost_freqs, 'Amplitude': min_amp_array}
    #
    #     fill_values_df = pd.DataFrame(fill_values_df_dict)
    #
    #     # merge the two dfs to have the same dimension of the RRCOS filter
    #     lpf_filled_df = pd.merge_ordered(fill_values_df, lpf_mag_df_interp)
    #
    #     # Perform FFT of signals and retrieve the frequencies
    #     # FFT performed only on the wanted samples
    #     i_sig = mod_sign_df['xi']
    #     q_sig = mod_sign_df['xq']
    #
    #     plot_constellation(i_sig + q_sig*1j, 'TX Constellation (standard modulator, no coherent RX)')
    #
    #     fig, axs = plt.subplots(2, 1, figsize=(9, 10))
    #
    #     # if noise_flag == True:
    #     #     i_samples = [i_sig[i*int(sps)] for i in range(num_signals)]
    #     #     q_samples = [q_sig[i*int(sps)] for i in range(num_signals)]
    #     #     # i_samples = [mod_sign_df['xi'][i * int(sps/2)] for i in range(int(num_signals/2))]
    #     #     # q_samples = [mod_sign_df['xq'][i * int(sps/2)] for i in range(int(num_signals/2))]
    #     # else:
    #     #     i_samples = [mod_sign_df['xi'][i*sps] for i in range(num_signals/2)]
    #     #     q_samples = [mod_sign_df['xq'][i*sps] for i in range(num_signals/2)]
    #
    #     i_sig_fft = np.fft.fft(i_sig)
    #     q_sig_fft = np.fft.fft(q_sig)
    #     i_sig_fft = np.fft.fftshift(i_sig_fft)
    #     q_sig_fft = np.fft.fftshift(q_sig_fft)
    #     i_sig_fft_mag = 20*np.log10(np.abs(i_sig_fft))
    #     q_sig_fft_mag = 20*np.log10(np.abs(q_sig_fft))
    #
    #     sig_freq = np.fft.fftfreq(n=mod_sign_df['xi'].size, d=1/samp_f)
    #     sig_freq = np.fft.fftshift(sig_freq)
    #
    #     fig, axs = plt.subplots(2, 1, figsize=(9, 10))
    #     fig.suptitle('I and Q Signals FFT')
    #     axs[0].set_title('I Signal (FFT)')
    #     axs[0].plot(sig_freq, i_sig_fft_mag, label='I')
    #
    #     axs[1].set_title('Q Signal (FFT)')
    #     axs[1].plot(sig_freq, q_sig_fft_mag, label='Q')
    #     plt.grid(True)
    #
    #     # Filter our signals, in order to apply the pulse shaping
    #     # These are the shaped signals we can use
    #     # add some noise generated randomly before
    #
    #     # in time domain
    #     i_shaped_t = scipy.signal.fftconvolve(i_sig, rrcos_filter[1])
    #     q_shaped_t = scipy.signal.fftconvolve(q_sig, rrcos_filter[1])
    #     #
    #     # # in frequency domain
    #     i_shaped = np.multiply(i_sig_fft, rrcos_fft)
    #     q_shaped = np.multiply(q_sig_fft, rrcos_fft)
    #     i_shaped_mag = 20*np.log10(np.abs(i_shaped))
    #     q_shaped_mag = 20*np.log10(np.abs(q_shaped))
    #
    #     # here we need to create the custom LPF generated by measured values and apply it at the input of the modulator
    #     # Apply LPF, but before that convert the dB20 values to magnitude values
    #     bandcut_i = np.multiply(i_shaped, lpf_filled_df['Amplitude'])
    #     bandcut_q = np.multiply(q_shaped, lpf_filled_df['Amplitude'])
    #
    #     bandcut_i_mag = 20*np.log10(np.abs(bandcut_i))
    #     bandcut_q_mag = 20*np.log10(np.abs(bandcut_q))
    #
    #     fig, axs = plt.subplots(2, 1, figsize=(9, 10))
    #     fig.suptitle('I and Q PSD after LPF filtering (in frequency)')
    #     axs[0].set_title('I pulse shaped')
    #     axs[0].plot(rrcos_freq, i_shaped_mag, label='I')
    #     axs[0].set_xlabel('Frequency')
    #     axs[0].set_ylabel('PSD')
    #     axs[1].set_title('Q pulse shaped')
    #     axs[1].plot(rrcos_freq, q_shaped_mag, label='Q')
    #     axs[1].set_xlabel('Frequency')
    #     axs[1].set_ylabel('PSD')
    #     plt.show(block=False)
    #
    #     # here we add AWGN noise to see what happens to the signal when a noise is present
    #     # if we add it before the IFFT, no effect is visible
    #     if noise_flag == True:
    #         i_shaped_time = np.fft.ifft(i_shaped)
    #         q_shaped_time = np.fft.ifft(q_shaped)
    #     else:
    #         i_shaped_time = np.fft.ifft(i_shaped)
    #         q_shaped_time = np.fft.ifft(q_shaped)
    #

    p_sign = i_sig * v_tx_param
    q_sign = q_sig * v_tx_param

    if classic_flag:
        # define Single Electrode Classic MZM for both I and Q arms
        i_MZM_list = [MZMSingleElectrode(p_field, p_signal, v_pi) for p_signal in p_sign]
        i_field_list = [i_MZM.out_mzm_field_evaluation() for i_MZM in i_MZM_list]
        q_MZM_list = [MZMSingleElectrode(q_field, q_signal, v_pi) for q_signal in q_sign]
        # for Q field evaluation, apply Phase Modulator effect rotating it, multiplying it for -1j
        q_field_list = [q_MZM.out_mzm_field_evaluation()*(-1)*(1j) for q_MZM in q_MZM_list]

        # combiner output field
        combiner_list = [Combiner(i_field_list[i], q_field_list[i]) for i in range(len(i_field_list))]
        out_fields_combiner_list = [combiner.combiner_out_field(combiner.in_A, combiner.in_B) for combiner in combiner_list]
        out_fields_samples = [out_fields_combiner_list for i in range(num_signals)]

        # plot constellations
        constellation_samples_classic = [out_fields_combiner_list]
        # constellation_samples_classic = [out_fields_combiner_list[i*int(sps/2)] for i in range(int(num_signals))]

        # plot constellations, normalized wrt a constant value 0.75, v_pi,
        # the constant transmission parameter and insertion loss
        classic_tx_const = (constellation_samples_classic/
                           (np.sqrt(input_power)))/(v_tx_param/v_pi)*(1/np.sqrt(insertion_loss))/norm_factor

    # create a list of Griffin MZMs using this input voltage interval
    # take into account the non-ideal params as b, c
    # for this evaluation, consider common mode voltage at v_bias

    # bias_offset_i = np.exp((b*(1.5*v_pi)**2)*1j)
    cpe_griffin = np.exp((b*(1.5*v_pi)**2)*1j)

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
