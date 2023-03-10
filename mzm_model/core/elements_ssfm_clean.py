"elements.py"

import numpy as np

from mzm_model.core.modulator_ssfm_params import gamma_1, gamma_2, v_off, phi_laser
from mzm_model.core.math_utils import lin2db


class SSFMLightSource(object):
    def __init__(self, json_spectral):
        self._input_power = json_spectral['channel_powers_dBm']
        self._out_field = 0

    @property
    def input_power(self):
        return self._input_power

    @input_power.setter
    def input_power(self, input_power):
        self._input_power = input_power

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Ideal Light Source methods'''

    def calculate_optical_input_power(self, input_source):
        optical_input_power = lin2db(input_source)
        self.out_field = np.sqrt(optical_input_power) * np.exp(phi_laser * -2j)
        return optical_input_power


'''Splitter'''


class Splitter(object):
    def __init__(self, input_power, input_field):
        self._input_power = input_power
        self._input_field = input_field
        self._A_in_field = 0
        self._B_in_field = 0

    @property
    def input_power(self):
        return self._input_power

    @property
    def input_field(self):
        return self._input_field

    @property
    def A_in_field(self):
        return self._A_in_field

    @A_in_field.setter
    def A_in_field(self, A_in_field):
        self._A_in_field = A_in_field

    @property
    def B_in_field(self):
        return self._B_in_field

    @B_in_field.setter
    def B_in_field(self, B_in_field):
        self._B_in_field = B_in_field

    '''Splitter methods'''

    def calculate_arms_input_fields(self):
        A_in_field = self.input_field * gamma_1
        B_in_field = (self.input_field * gamma_1) * 1j
        return A_in_field, B_in_field


'''Waveguide'''


class Waveguide(object):
    def __init__(self, in_field):
        self._in_field = in_field
        self._out_field = 0

    @property
    def in_field(self):
        return self._in_field

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Waveguides methods'''

    def delta_phi_evaluation(self, v_arm, v_pi):
        delta_phi = np.pi*(v_arm/v_pi)
        return delta_phi

    def delta_phi_evaluation_k(self, v_arm, k_arm):
        delta_phi = v_arm*k_arm
        return delta_phi

    def out_field_evaluation(self, arm_field, v_arm, v_pi):
        delta_phi = self.delta_phi_evaluation(v_arm, v_pi)
        out_field = arm_field*np.exp(delta_phi*1j)
        return out_field


'''Combiner'''


class Combiner(object):
    def __init__(self, in_A, in_B):
        self._in_A = in_A
        self._in_B = in_B
        self._out_field = 0

    @property
    def in_A(self):
        return self._in_A

    @property
    def in_B(self):
        return self._in_B

    @property
    def out_field(self):
        return self._out_field
    
    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Combiner methods'''

    def combiner_out_field(self, in_A, in_B):
        comb_out_field = gamma_2*(in_A+in_B*1j)
        self.out_field = comb_out_field
        return comb_out_field


'''Single Electrode MZM'''


class MZMSingleElectrode(object):
    def __init__(self, in_field, v_in, v_pi):
        self._in_field = in_field
        self._v_in = v_in
        self._v_pi = v_pi
        self._out_field = 0

    @property
    def in_field(self):
        return self._in_field

    @property
    def v_in(self):
        return self._v_in

    @property
    def v_pi(self):
        return self._v_pi

    @property
    def out_field(self):
        return self._out_field

    @out_field.setter
    def out_field(self, out_field):
        self._out_field = out_field

    '''Single Electrode MZM methods'''

    def out_mzm_field_evaluation(self):
        v_pi = self.v_pi
        in_field = self.in_field
        v_in = self.v_in
        arg = (np.pi / 2) * ((v_in - v_off) / v_pi)
        # out_field = in_field*(np.sin(arg) - np.cos(arg)*(1/np.sqrt(er))*1j)
        out_field = in_field * np.sqrt(insertion_loss) * (np.sin(arg) - ((1 / np.sqrt(er)) * np.cos(arg)) * 1j)
        self.out_field = out_field
        return out_field

    from mzm_model.core.utils import eo_tf_draw


'''InP MZM'''


class InP_MZM(object):
    def __init__(self, lambda_mzm, vcm_phase, vcm_bias, vpi, vdiff, gamma_1, gamma_2, b, c, driver_gain, rf):
        # lambda in use
        self._lambda_mzm = lambda_mzm
        # vcm phase
        self._vcm_phase = vcm_phase
        # vcm phase
        self._vcm_bias = vcm_bias
        # driving voltages
        self._vdiff = vdiff
        self._vpi = vpi
        self._gamma_1 = gamma_1
        self._gamma_2 = gamma_2
        self._b = b
        self._c = c
        self._vl = np.array([vcm_phase - vdiff/2 - driver_gain*rf_i/2 for rf_i in rf])
        self._vr = np.array([vcm_phase + vdiff/2 + driver_gain*rf_i/2 for rf_i in rf])
        self._intensities = self.griffin_intensity_tf()
        self._phases = self.griffin_phase()
        self._phase_vl = self._phases[0]
        self._phase_vr = self._phases[1]
        self._transmission_vl = self._intensities[0]
        self._transmission_vr = self._intensities[1]
        self._rf = np.array(rf)
        self._driver_gain = driver_gain

    @property
    def lambda_mzm(self):
        return self._lambda_mzm

    @property
    def vcm_phase(self):
        return self._vcm_phase

    @property
    def vcm_bias(self):
        return self._vcm_bias

    @property
    def vdiff(self):
        return self._vdiff

    @property
    def vl(self):
        return self._vl

    @property
    def vr(self):
        return self._vr

    @property
    def vpi(self):
        return self._vpi

    @property
    def gamma_1(self):
        return self._gamma_1

    @property
    def gamma_2(self):
        return self._gamma_2

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def phase_vl(self):
        return self._phase_vl

    @phase_vl.setter
    def phase_vl(self, phase_vl):
        self._phase_vl = phase_vl

    @property
    def phase_vr(self):
        return self._phase_vr

    @phase_vr.setter
    def phase_vr(self, phase_vr):
        self._phase_vr = phase_vr

    @property
    def rf(self):
        return self._rf

    @rf.setter
    def rf(self, rf):
        self._rf = rf

    @property
    def driver_gain(self):
        return self._driver_gain

    @driver_gain.setter
    def driver_gain(self, driver_gain):
        self._driver_gain = driver_gain

    @property
    def transmission_vl(self):
        return self._transmission_vl

    @transmission_vl.setter
    def transmission_vl(self, transmission_vl):
        self._transmission_vl = transmission_vl

    @property
    def transmission_vr(self):
        return self._transmission_vr

    @transmission_vr.setter
    def transmission_vr(self, transmission_vr):
        self._transmission_vr = transmission_vr

    '''InP Model methods'''
    def eval_varm(self):
        vdiff = self.vdiff
        vph = self.vcm_phase
        driv_gain = self.driver_gain
        vl = self.vl
        vr = self.vr
        return [vl, vr]

    # define V-dependent intensity tf
    def griffin_intensity_tf(self):
        c = self.c
        tf_l = (1 + np.exp((self.vl - c) / 0.8)) ** -1.25
        tf_r = (1 + np.exp((self.vr - c) / 0.8)) ** -1.25

        return [tf_l, tf_r]

    def griffin_phase(self):
        b = self.b
        v_pi = self.vpi
        vcm_phase = self.vcm_phase
        # vl = self.eval_varm()[0]
        # vr = self.eval_varm()[1]
        phase_l = ((2 * b * vcm_phase * v_pi - np.pi) / v_pi) * self.vl - b * self.vl ** 2
        phase_r = ((2 * b * vcm_phase * v_pi - np.pi) / v_pi) * self.vr - b * self.vr ** 2

        return [phase_l, phase_r]

    def griffin_eo_tf(self):
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        transmission_vl = self.transmission_vl
        transmission_vr = self.transmission_vr
        phase_vl = self.phase_vl
        phase_vr = self.phase_vr

        eo_tf_field = gamma_1 * gamma_2 * np.sqrt(transmission_vl) * np.exp(phase_vl * 1j) + \
                      np.sqrt(np.abs(1 - gamma_1 ** 2) * np.abs(1 - gamma_2 ** 2)) * np.sqrt(transmission_vr) * \
                      np.exp(phase_vr * 1j)
        eo_field_conj = np.conjugate(eo_tf_field)
        eo_tf_power = eo_tf_field * eo_field_conj
        return eo_tf_power

    def griffin_eo_tf_field(self):
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        transmission_vl = self.transmission_vl
        transmission_vr = self.transmission_vr
        phase_vl = self.phase_vl
        phase_vr = self.phase_vr
        eo_tf_field = gamma_1 * gamma_2 * np.sqrt(transmission_vl) * np.exp(phase_vl * 1j) + \
                      np.sqrt(np.abs(1 - gamma_1 ** 2) * np.abs(1 - gamma_2 ** 2)) * np.sqrt(transmission_vr) * \
                      np.exp(phase_vr * 1j)
        return eo_tf_field

    def griffin_single_phase_analysis(self, b, v_pi, vcm, rf):
        vdd_oc = 3.7        # V
        # vsub = - (vcm + vdd_oc)
        vsub = vcm
        # v_arm = (vsub + 0.5 * rf[:, 0])
        v_arm = (vsub)
        param_1 = (2 * b * vsub * v_pi - np.pi) / v_pi
        phase_single_arm = param_1 * v_arm - b * (v_arm ** 2)

        return np.array(phase_single_arm), vsub