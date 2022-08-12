#%%
import pandas as pd
import numpy as np

import qiskit.pulse as pulse
from qiskit.circuit import Parameter

from qiskit_experiments.calibration_management.calibrations import Calibrations
import qiskit_dynamics.pulse.pulseSimClass
import importlib

from qiskit import IBMQ, schedule

#%%
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
ibm_backend = provider.get_backend('ibm_cairo')
#%%
# ibm_backend = provider.get_backend('fake_ibm_cairo')
# from qiskit.providers.fake_provider import FakeManila
# ibm_backend = FakeManila()


#%%
importlib.reload(qiskit_dynamics.pulse.pulseSimClass)
from qiskit_dynamics.pulse.pulseSimClass import PulseSimulator

backend = PulseSimulator.from_backend(ibm_backend, subsystem_list=[1,2])

cals = Calibrations.from_backend(backend)

#%%
qubit = 0  # The qubit we will work with
def setup_cals(backend) -> Calibrations:
    """A function to instantiate calibrations and add a couple of template schedules."""
    cals = Calibrations.from_backend(backend)

    dur = Parameter("dur")
    amp = Parameter("amp")
    sigma = Parameter("σ")
    beta = Parameter("β")
    drive = pulse.DriveChannel(Parameter("ch0"))

    # Define and add template schedules.
    with pulse.build(name="xp") as xp:
        pulse.play(pulse.Drag(dur, amp, sigma, beta), drive)

    with pulse.build(name="xm") as xm:
        pulse.play(pulse.Drag(dur, -amp, sigma, beta), drive)

    with pulse.build(name="x90p") as x90p:
        pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("β")), drive)

    cals.add_schedule(xp, num_qubits=1)
    cals.add_schedule(xm, num_qubits=1)
    cals.add_schedule(x90p, num_qubits=1)

    return cals

def add_parameter_guesses(cals: Calibrations):
    """Add guesses for the parameter values to the calibrations."""
    for sched in ["xp", "x90p"]:
        cals.add_parameter_value(80, "σ", schedule=sched)
        cals.add_parameter_value(0.5, "β", schedule=sched)
        cals.add_parameter_value(320, "dur", schedule=sched)
        cals.add_parameter_value(0.5, "amp", schedule=sched)

#%%
cals = setup_cals(backend)
add_parameter_guesses(cals)

#%%
from qiskit_experiments.calibration_management.basis_gate_library import FixedFrequencyTransmon
library = FixedFrequencyTransmon(default_values={"duration": 320})
cals = Calibrations.from_backend(backend, libraries=[library])
#%%
from qiskit_experiments.library.calibration.rough_frequency import RoughFrequencyCal

#%%
freq01_estimate = backend.defaults().qubit_freq_est[qubit]
frequencies = np.linspace(freq01_estimate -15e6, freq01_estimate + 15e6, 51)
spec = RoughFrequencyCal(qubit, cals, frequencies, backend=backend)
spec.set_experiment_options(amp=0.1)