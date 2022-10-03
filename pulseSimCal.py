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
# IBMProvider.save_account(token=<token>)
# needs to be installed from this source: https://github.com/Qiskit/qiskit-ibm-provider

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
    # Below need to fix the bug https://github.com/Qiskit/qiskit-dynamics/issues/127
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
result
#%%