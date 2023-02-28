import numpy as np
import pandas as pd
from matplotlib import use
import matplotlib.pyplot as plt
import time
from cmath import phase
import seaborn as sns; sns.set_theme()
from pathlib import Path
from math import radians, degrees
from mzm_model.core.elements_ssfm_clean import SSFMLightSource, Splitter, Waveguide, Combiner, InP_MZM
from mzm_model.core.modulator_ssfm_params import h, frequency, q, v_pi,phase_offset, v_off, b, c, gamma_1, gamma_2,\
    v_pi_values, lambda_wave, vcm_phase, vcm_bias, v_diff_i, v_diff_q, v_diff_ph, plot_flag, eval_vpi, csv_vpi
from mzm_model.core.math_utils import lin2db, lin2dbm, db2lin, dbm2lin, normalize
from mzm_model.core.soa_config_singlechannel import pre_soa_out_power, post_soa_out_power
from mzm_model.core.science_utils_ssfm import evaluate_field, griffin_phase_electrode, field_to_power_dbm
import json

start_time = time.time()
use('Qt5Agg')

root = Path(__file__).parent.parent
input_folder = root/'mzm_model'/'resources'
folder_results = root/'mzm_model'/'results'
json_out = folder_results/'json_params.json'
json_source = root.parent/'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'

json_file = json_source/'ssi_cut0_pump0_2023.json'

json_params = json.load(open(json_file, 'r'))

with open(json_out, 'w') as outfile:
    json.dump(json_params, outfile, indent=4)

json_spectral = json_params['spectral_info']
json_mod_params = json_params['modulator_params']
driver_gain_xi = json_mod_params['driver_gain_i']
rf_i = pd.read_csv(input_folder / 'xi.csv', header=None).values
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

xi_input_field = evaluate_field(xi_power_dbm)


vcm_array = np.array([3, 5, 7, 9])
b_array = np.arange(0.05, 0.21, 0.05)
ph_list = []
mz_xi = InP_MZM(lambda_wave, vcm_phase, vcm_bias, 1, 0, gamma_1, gamma_2, 0, 0, driver_gain_xi, rf_i)

for b_i in b_array:
    ph_array = []
    vsub_array = []
    for vcm in vcm_array:
        vpi = eval_vpi(str(lambda_wave*1e9), vcm, csv_vpi)[3]
        rf_i = normalize(-vpi, vpi, rf_i)
        # plt.plot(rf_i)
        # plt.show()
        ratio = vcm / vpi
        product = vcm * vpi
        b_eval = product / (2 * np.pi) - 1
        # For MZi values, consider that MZ1 = YQ (0), MZ2 = YI (1), MZ3 = XQ (2), MZ4 = XI (3)
        phase_vsub = mz_xi.griffin_single_phase_analysis(b_i, vpi, vcm, rf_i)
        # plt.plot(np.sort(phase_vsub[0]))
        # plt.show()
        phase = phase_vsub[0]
        vsub = phase_vsub[1]

        # plt.plot(phase)
        # plt.show()
        ph_mean = np.mean(phase)
        ph_array.append(ph_mean)
        vsub_array.append(vsub)
    plt.plot(vcm_array, ph_array)
    plt.show()
    ph_list.append(ph_array)
plt.plot(ph_list)
plt.show()
# evaluate the electric fields of MZi at output
mz_xi_field = mz_xi.griffin_eo_tf_field()*xi_input_field

mz_xi_power = field_to_power_dbm(mz_xi_field)

# Apply the Phase Offset considering the vdiff_phase to be applied to the fields
# At first evaluate the phases
phases_shifts = griffin_phase_electrode(mz_phase.vl, mz_phase.vr, v_pi_values[3], v_pi_values[2], b)
# Apply Phase Electrode phases independently to each XI and XQ arm
mz_xi_ph_field = mz_xi_field * np.exp(phases_shifts[0] * 1j)
mz_xq_ph_field = mz_xq_field * np.exp(phases_shifts[1] * 1j)

mz_xi_ph_power = field_to_power_dbm(mz_xi_ph_field)     # [dBm]
mz_xq_ph_power = field_to_power_dbm(mz_xq_ph_field)     # [dBm]
"TODO: considera di trovare tramite software il valore da applicare al PHASE MODULATOR in modo " \
"da ottenere una rotazione ideale pari a pi/2"
phase_diff_norm_pi_half = (phase(mz_xi_field) - phase(mz_xq_field))/(np.pi/2)
# combiner output field Griffin
combiner_single_pol = Combiner(mz_xi_ph_field, mz_xq_ph_field)
out_fields_combiner_single_pol = combiner_single_pol.combiner_out_field(combiner_single_pol.in_A,
                                                                        combiner_single_pol.in_B)
out_power_single_pol_pre_post_soa = field_to_power_dbm(out_fields_combiner_single_pol)      # [dBm]
"TODO: FIND THE PD1 VALUE AND CHECK IT IS ALMOST THE SAME AS THE PREVIOUS POWER RETRIEVED VALUE"

"QUI VA APPLICATA LA POTENZA DEL POST SOA"
"FARE ATTENZIONE ALLE LOSS DEL POST SOA, FORSE VA TOLTA QUELLA DI QUAD LOSS PERCHE' GIA' IL COMBINER DA' LOSS"
pout_post_soa = post_soa_out_power(out_power_single_pol_pre_post_soa)

# HERE A CORRECT EVALUATION OF EXTINCTION RATIO AND INSERTION LOSS IS PERFORMED SWEEPING ON ALL DIFFERENT POSSIBLE TPEs
vdiff_array = np.arange(-2.5, 2.52, 0.01)

il_list = []
er_list = []
mz_pow_lists = []
for i in range(4):
    mz_list = [InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[i], v_diff, gamma_1, gamma_2, b, c)
               for v_diff in vdiff_array]
    mz_pow_list = np.array([dbm2lin(field_to_power_dbm(mzm.griffin_eo_tf_field())) for mzm in mz_list])
    max_pow = max(mz_pow_list)
    min_pow = min(mz_pow_list)
    insertion_loss = - lin2db(max_pow)
    er = lin2db(max_pow/min_pow)
    il_list.append(insertion_loss)
    er_list.append(er)
    mz_pow_lists.append(lin2dbm(mz_pow_list))

mz_pow_lists = np.array(mz_pow_lists)
# if plot_flag == 1:
plt.figure()
for list in mz_pow_lists:
    plt.plot(vdiff_array, list)
plt.show(block=False)

b_list = np.arange(0.00, 0.21, 0.05)
c_list = [20, 7.3, 6.3, 5.3, 4.3]
vdiff_array = np.arange(-13, 3, 0.01)

b_phase_list = []
c_transmission_list = []

for b_new in b_list:
    mz_list_b = [InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[3], v_diff, gamma_1, gamma_2, b_new, c_list[0])
               for v_diff in vdiff_array]
    mz_phase_list = np.array([mzm.phase_vr for mzm in mz_list_b])
    b_phase_list.append(mz_phase_list)
b_phase_list = np.array(b_phase_list)

for c_new in c_list:
    mz_list_c = [InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[3], v_diff, gamma_1, gamma_2, b_list[0], c_new)
               for v_diff in vdiff_array]
    mz_transmission_list = np.array([mzm.transmission_vl for mzm in mz_list_c])
    c_transmission_list.append(mz_transmission_list)
c_transmission_list = np.array(c_transmission_list)

plt.figure()
for b_list in b_phase_list:
    plt.plot(vdiff_array, b_list)
plt.show(block=False)

plt.figure()
for c_list in c_transmission_list:
    plt.plot(vdiff_array, c_list)
plt.show(block=False)
plt.show(block=True)

print()
