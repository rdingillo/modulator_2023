import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import seaborn as sns; sns.set_theme()
from pathlib import Path
from mzm_model.core.elements_ssfm_clean import SSFMLightSource, Splitter, Waveguide, Combiner, InP_MZM
# from mzm_model.core.modulator_ssfm_params import h, frequency, q, v_pi, \
#     v_diff, v_in_limit, v_in_step, er, insertion_loss, phase_offset, v_off, v_bias, b, c, wavelength, gamma_1, gamma_2,\
#     v_pi_values
from mzm_model.core.modulator_ssfm_params import h, frequency, q, v_pi,phase_offset, v_off, b, c, gamma_1, gamma_2,\
    v_pi_values, lambda_wave, vcm_phase, vcm_bias, v_diff_i, v_diff_q, v_diff_ph
from mzm_model.core.math_utils import lin2db, lin2dbm, db2lin, dbm2lin
from mzm_model.core.soa_config_singlechannel import pre_soa_out_power, post_soa_out_power
from mzm_model.core.science_utils_ssfm import evaluate_field, griffin_phase_electrode, field_to_power_dbm
import json

start_time = time.time()
# matplotlib.use('Qt5Agg')

root = Path(__file__).parent.parent
input_folder = root/'resources'
folder_results = root/'mzm_model'/'results'
json_out = folder_results/'json_params.json'
json_source = root.parent/'optical-system-interface/resources/ssfm_test_configs/lumentum_modulator/spectral_information'

json_file = json_source/'ssi_cut0_pump0.json'

json_params = json.load(open(json_file, 'r'))

with open(json_out, 'w') as outfile:
    json.dump(json_params, outfile, indent=4)

json_spectral = json_params['spectral_info']

# retrieve optical input source
source = SSFMLightSource(json_spectral)

# evaluate optical input power in dBm
input_power_dbm = source.input_power  # dBm
input_power = dbm2lin(input_power_dbm)  # Watt
source.out_field = source.calculate_optical_input_power(input_power_dbm)
input_field = source.out_field

# # define TX amplitude parameter in time (for the field)
# k_tx = (np.sqrt(input_power) / (np.sqrt(2))) * (np.pi / (2 * v_pi))
#
# # evaluate electric fields in splitter
# splitter = Splitter(source.input_power, source.out_field)
# splitter_fields = splitter.calculate_arms_input_fields()
# # save p and q field for the 2 arms (I, Q)
# x_field = splitter_fields[0]  # [mV/m]
# y_field = splitter_fields[1]  # [mV/m]
# splitter.A_in_field = x_field
# splitter.B_in_field = y_field

"""NB ARRIVATI A QUESTO PUNTO SIAMO AL PRIMO SPLITTER DOVE SI DIVIDONO X E Y"""
""" QUI VANNO INSERITI I VALORI DEI PRE-SOA"""
pol_power_dbm = pre_soa_out_power()
xi_power_dbm = pol_power_dbm - 3
xq_power_dbm = xi_power_dbm

xi_input_field = evaluate_field(xi_power_dbm)
xq_input_field = evaluate_field(xq_power_dbm)

# combiner output field

# combiner = Combiner(arm_a.out_field, arm_b.out_field)
# out_field_combiner = combiner.combiner_out_field(combiner.in_A, combiner.in_B)
# combiner.out_field = out_field_combiner
# # to check that combiner field is the same wrt eo_tf, try to take the abs**2 of combiner_out_field/sqrt()input_power)
# power_tf_check = (np.abs(combiner.out_field)/np.sqrt(input_power))**2
# create a list of Griffin MZMs using this input voltage interval
# take into account the non-ideal params as b, c
# for this evaluation, consider common mode voltage at 0

# For MZi values, consider that MZ1 = YQ (0), MZ2 = YI (1), MZ3 = XQ (2), MZ4 = XI (3)
vpi_phase = np.mean([v_pi_values[2], v_pi_values[3]])
mz_xi = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[3], v_diff_i, gamma_1, gamma_2, phase_offset, b, c)
mz_xq = InP_MZM(lambda_wave, vcm_phase, vcm_bias, v_pi_values[2], v_diff_q, gamma_1, gamma_2, phase_offset, b, c)
mz_phase = InP_MZM(lambda_wave, vcm_phase, vcm_bias, vpi_phase, v_diff_ph, gamma_1, gamma_2, phase_offset, b, c)

# evaluate the electric fields of MZi at output
mz_xi_field = mz_xi.griffin_eo_tf_field()*xi_input_field
mz_xq_field = mz_xq.griffin_eo_tf_field()*xq_input_field

mz_xi_power = field_to_power_dbm(mz_xi_field)
mz_xq_power = field_to_power_dbm(mz_xq_field)

# Apply the Phase Offset considering the vdiff_phase to be applied to the fields
# At first evaluate the phases
phases_shifts = griffin_phase_electrode(mz_phase.vl, mz_phase.vr, v_pi_values[3], v_pi_values[2])
# Apply Phase Electrode phases independently to each XI and XQ arm
mz_xi_ph_field = mz_xi_field * np.exp(phases_shifts[0] * 1j)
mz_xq_ph_field = mz_xq_field * np.exp(phases_shifts[1] * 1j)

mz_xi_ph_power = field_to_power_dbm(mz_xi_ph_field)     # [dBm]
mz_xq_ph_power = field_to_power_dbm(mz_xq_ph_field)     # [dBm]
"TODO: considera di trovare tramite software il valore da applicare al PHASE MODULATOR in modo " \
"da ottenere una rotazione ideale"
# combiner output field Griffin
combiner_single_pol = Combiner(mz_xi_ph_field, mz_xq_ph_field)
out_fields_combiner_single_pol = combiner_single_pol.combiner_out_field(combiner_single_pol.in_A,
                                                                        combiner_single_pol.in_B)
out_power_single_pol_pre_post_soa = field_to_power_dbm(out_fields_combiner_single_pol)      # [dBm]
"TODO: QUI VA APPLICATA LA POTENZA DEL POST SOA"
"FARE ATTENZIONE ALLE LOSS DEL POST SOA, FORSE VA TOLTA QUELLA DI QUAD LOSS PERCHE' GIA' IL COMBINER DA' LOSS"
pout_post_soa = post_soa_out_power(out_power_single_pol_pre_post_soa)


# evaluate Griffin V_bias to have the same V_pi of the classic MZM
'Set a constant v_bias to get the same v_pi of the classic mzm, otherwise leave the variable one'
v_bias = np.pi / v_pi
# griffin_mzm_list = [GriffinMZM(v_bias, v_in/2 + v_pi/2, (2*v_bias - v_in/2 - v_pi/2), gamma_1, gamma_2, phase_offset, b, c) for v_in in v_in_range]
griffin_mzm_list = [InP_MZM(v_bias, v_in/2, 0, gamma_1, gamma_2, phase_offset, b, c, er) for v_in in v_in_range]
griffin_eo_tf_list = np.array([mzm.griffin_eo_tf() for mzm in griffin_mzm_list])
griffin_phase_vl_list = np.array([mzm.phase_vl for mzm in griffin_mzm_list])/np.pi
griffin_phase_vr_list = np.array([mzm.phase_vr for mzm in griffin_mzm_list])/np.pi
griffin_transmission_vl_list = np.array([mzm.transmission_vl for mzm in griffin_mzm_list])
griffin_transmission_vr_list = np.array([mzm.transmission_vr for mzm in griffin_mzm_list])
griffin_il_list = np.array([mzm.griffin_il_er()[0] for mzm in griffin_mzm_list])
griffin_er_list = np.array([mzm.griffin_il_er()[1] for mzm in griffin_mzm_list])
griffin_il_on_er_list = np.array([mzm.griffin_il_er()[2] for mzm in griffin_mzm_list])

b_list = np.arange(0.00, 0.21, 0.05)
c_list = [20, 7.3, 6.3, 5.3, 4.3]


griffin_b_list = [[GriffinMZM(v_bias, v_in, 2*v_bias - v_in, gamma_1, gamma_2, phase_offset, b_par, 20, er) for v_in in v_in_range] for b_par in b_list]
# griffin_phase_vin = np.array([[[mzm.griffin_phase(v_in) for mzm in griffin_b_list[i]] for v_in in v_in_range]for i in range(len(griffin_b_list))])
griffin_c_list = [[GriffinMZM(v_bias, v_in, 0, gamma_1, gamma_2, phase_offset, 0.00, c_par, er) for v_in in v_in_range] for c_par in c_list]

griffin_eo_tf_bvar = np.array([[mzm.griffin_eo_tf() for mzm in griffin_b_list[i]] for i in range(len(griffin_b_list))])
griffin_eo_tf_cvar = np.array([[mzm.griffin_eo_tf() for mzm in griffin_c_list[i]] for i in range(len(griffin_c_list))])



griffin_phase_vl_bvar = np.array([[mzm.phase_vl for mzm in griffin_b_list[i]] for i in range(len(griffin_b_list))])/np.pi
griffin_transmission_vl_cvar = np.array([[mzm.transmission_vl for mzm in griffin_c_list[i]] for i in range(len(griffin_c_list))])

v_pi_gr = griffin_mzm_list[0].v_pi
# plot EO TFs
non_ideal_mzm.eo_tf_draw(v_in_range, eo_tf_list, non_ideal_eo_tf_list, griffin_eo_tf_list, griffin_il_list, griffin_er_list, griffin_il_on_er_list, v_pi_gr)
griffin_mzm.phase_transmission_draw(v_in_range, griffin_phase_vl_list, griffin_transmission_vl_list)
griffin_mzm.phase_parametric_draw(v_in_range, b_list, griffin_phase_vl_bvar)
griffin_mzm.transmission_parametric_draw(v_in_range, c_list, griffin_transmission_vl_cvar)

plt.show(block=True)


print()
