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
from qiskit_ibm_provider import IBMProvider
# IBMProvider.save_account(token='2ea5ea951217c0dd712a85fb93e0dfbc9f22e211b141e86fca50a039627ef60b07f4c2ac5f96207805ae14c17df4e1dd23144dbc6826fc607be539f6041299ce')

#%%
# provider = IBMProvider.get_provider(hub='ibm-q-internal', group='deployed', project='default')
# ibm_backend = provider.get_backend('ibmq_manilla')
provider = IBMProvider()
ibm_backend = provider.get_backend('ibmq_lima')

#%%
# ibm_backend = provider.get_backend('fake_ibm_cairo')
# from qiskit.providers.fake_provider import FakeManila
# ibm_backend = FakeManila()


#%%
importlib.reload(qiskit_dynamics.pulse.pulseSimClass)
from qiskit_dynamics.pulse.pulseSimClass import PulseSimulator

# backend = PulseSimulator.from_backend(ibm_backend, subsystem_list=[0,1,2,3,4])
backend = PulseSimulator.from_backend(ibm_backend, subsystem_list=[0,1])
#%%
# from qiskit_dynamics.pulse.pulseSimClass import solver_from_backend
# subsystem_list=[0,1,2,3,4]
# pulseSim = PulseSimulator(solver=solver_from_backend(ibm_backend, subsystem_list))
#%%
# backend=ibm_backend
#%%
from qiskit.pulse import library

amp = 1
sigma = 10
num_samples = 128
#%%
gaus = pulse.library.Gaussian(num_samples, amp, sigma,
                              name="Parametric Gaus")
gaus.draw()

# %%
importlib.reload(qiskit_dynamics.pulse.pulseSimClass)
from qiskit_dynamics.pulse.pulseSimClass import PulseSimulator
with pulse.build() as schedule:
    pulse.play(gaus, backend.drive_channel(0))
    pulse.play(gaus, backend.drive_channel(1))
    pulse.play(gaus, backend.control_channel([0,1])[0])
    pulse.play(gaus, backend.control_channel([1,0])[0])
    pulse.play(gaus, backend.control_channel([1,2])[0])
    pulse.play(gaus, backend.control_channel([2,1])[0])
    pulse.play(gaus, backend.control_channel([1,3])[0])
    pulse.play(gaus, backend.control_channel([3,1])[0])
    # pulse.play(gaus, backend.drive_channel(2))
    # pulse.play(gaus, backend.drive_channel(3))
    # pulse.play(gaus, backend.drive_channel(4))

schedule.draw()
y0 = np.zeros(backend.solver.model.dim)
y0[0] = 1
t_span = np.array([0, num_samples * backend.solver._dt])
result = backend.run(schedule, y0=y0, t_span=t_span)



#%%
cals = Calibrations.from_backend(backend)

# spec = RoughFreqencyCal(qubit, cals, frequencies, backend=backend)
#%%
qubit = 1  # The qubit we will work with
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
pd.DataFrame(**cals.parameters_table(qubit_list=[qubit, ()]))

#%%
freq01_estimate = ibm_backend.defaults().qubit_freq_est[qubit]
frequencies = np.linspace(freq01_estimate -15e6, freq01_estimate + 15e6, 51)
spec = RoughFrequencyCal(qubit, cals, frequencies, backend=backend)
spec.set_experiment_options(amp=0.1)
# %%
circuit = spec.circuits()[0]
circuit.draw(output="mpl")

# %%
schedule(circuit, backend).draw()

# %%
spec_data = spec.run().block_for_results()

# %%



# %%
a
# %%
