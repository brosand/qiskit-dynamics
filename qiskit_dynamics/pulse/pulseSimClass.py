#%%
from typing import Dict, Iterable, Optional, Union
from qiskit.test.mock import FakeVigo
import numpy as np
device_backend = FakeVigo()

#%%
# from typing import Union
from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.pulse.channels import AcquireChannel, DriveChannel, MeasureChannel, ControlChannel

from qiskit_dynamics import Solver
from qiskit_dynamics.pulse.backend_parser.string_model_parser import parse_hamiltonian_dict

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import Schedule, ScheduleBlock

from qiskit.transpiler import Target
from qiskit.result import Result

def solver_from_backend(backend: Backend, subsystem_list: list[int]) -> 'PulseSimulator':
    """
    Create a solver object from a backend.
    :param backend: The backend to use.
    :param subsystem_list: The qubits to use for the simulation.
    :return: A Solver object.
    """

    if isinstance(backend, BackendV2):
        ham_dict = backend.hamiltonian
    else:
        ham_dict = backend.configuration().hamiltonian
    
    static_hamiltonian, hamiltonian_operators, reduced_channels, subsystem_dims = parse_hamiltonian_dict(ham_dict, subsystem_list)
    

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=hamiltonian_operators,
        rotating_frame=np.diag(static_hamiltonian)
    )
    return solver

# Do we want the hamiltonian and the operators to be separate?
# We could also have no init, and just have users init with a solver, or use simulator.from_
class PulseSimulator(BackendV2):
    def __init__(self, hamiltonian=None, operators=None, solver=None, acquire_channels=None, control_channels=None, measure_channels=None, drive_channels=None):
        super().__init__()
        if solver is None:
            self.solver = solver_from_hamiltonian(hamiltonian, operators)
        else:
            self.solver = solver
        # self.acquire_channels = acquire_channels
        # self.control_channels = control_channels
        # self.measure_channels = measure_channels
        # self.drive_channels = drive_channels
        # self.control_channels = control_channels
        # self.coupling_map = coupling_map

    @classmethod
    def from_backend(self, backend: BackendV2, subsystem_list: Optional[list[int]] = None) -> 'PulseSimulator':
        """
        Create a PulseSimulator object from a backend.
        :param backend: The backend to use.
        :param subsystem_list: The qubits to use for the simulation.
        :return: A PulseSimulator object.
        """
        pulseSim = PulseSimulator(solver=solver_from_backend(backend, subsystem_list))
        pulseSim.subsystem_list = subsystem_list
        pulseSim.solver = solver_from_backend(backend, subsystem_list)
        pulseSim.name = backend.name
        if isinstance(backend, BackendV1):
            pulseSim.qubit_properties = backend.properties().qubit_property
            pulseSim.target = Target()
            pulseSim.target.qubit_properties = [pulseSim.qubit_properties(i) if i in subsystem_list else None for i in range(backend.configuration().n_qubits)]
        else:
            pulseSim.qubit_properties = backend.qubit_properties
            pulseSim.target = backend.target
            pulseSim.drive_channel = backend.drive_channel
            pulseSim.control_channel = backend.control_channel

            # pulseSim.target = Target()
            # pulseSim.target.qubit_properties = [pulseSim.qubit_properties(i) if i in subsystem_list else None for i in range(backend.configuration().n_qubits)]
            # pulseSim.target.dt = backend.dt
            pulseSim._dtm = backend.dtm
            pulseSim._meas_map = backend.meas_map
            pulseSim.base_backend = backend
        return pulseSim
    
        # Set various attributes from backend
        # Instantiate a `Qiskit-Dynamics` solver
        # set an attribute for the subsystem of the device to simulate
    
    def acquire_channel(self, qubit: Iterable[int]) -> Union[int, AcquireChannel, None]:
        return self.base_backend.acquire_channel(qubit)
    
    def drive_channel(self, qubit: int) -> Union[int, DriveChannel, None]:
        return self.base_backend.drive_channel(qubit)

    def control_channel(self, qubit: int) -> Union[int, ControlChannel, None]:
        return self.base_backend.control_channel(qubit)
    
    def measure_channel(self, qubit: int) -> Union[int, MeasureChannel, None]:
        return self.base_backend.measure_channel(qubit)
    
    # def qubit_properties(self, qubit: int):
    #     return self.base_backend.qubit_properties(qubit)
    
    def run(self, run_input: Union[QuantumCircuit, Schedule, ScheduleBlock], **options) -> Result:
        return super().run(run_input, **options)

    def run(self, run_input, **options):
        """Run on the backend.

        This method returns a :class:`~qiskit.providers.Job` object
        that runs circuits. Depending on the backend this may be either an async
        or sync call. It is at the discretion of the provider to decide whether
        running should block until the execution is finished or not: the Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of
                :class:`~qiskit.circuits.QuantumCircuit,
                :class:`~qiskit.pulse.ScheduleBlock`, or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.
        Returns:
            Job: The job object for the run
        """
        pass


    def get_solver(self):
        return self.solver
    
    # def target(self):
    #     return Target()

    def _default_options(self):
        pass
    
    def max_circuits(self):
        pass

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals

        Returns:
            dtm: The output signal timestep in seconds.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                output signal timestep
        """
        return self._dtm


    @property
    def meas_map(self) -> list[list[int]]:
        """Return the grouping of measurements which are multiplexed

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            meas_map: The grouping of measurements which are multiplexed

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        return self._meas_map
    


 
# Below is for the case of a custom hamiltonian (separated temporarily)
class PulseSimulator1(BackendV2):
    def __init__(self, hamiltonian,acquire_channels=None, control_channels=None, measure_channels=None, drive_channels=None) -> None:
        self.hamiltonian = hamiltonian
        self.acquire_channels = acquire_channels
        self.control_channels = control_channels
        self.measure_channels = measure_channels
        self.drive_channels = drive_channels

    
        # Set various attributes from backend
        # Instantiate a `Qiskit-Dynamics` solver
        # set an attribute for the subsystem of the device to simulate
    
    def acquire_channel(self, qubit: int) -> AcquireChannel:
        return self.acquire_channels[qubit]
    
    def drive_channel(self, qubit: int) -> DriveChannel:
        return self.drive_channels[qubit]

    def control_channel(self, qubit: int) -> ControlChannel:
        return self.control_channels[qubit]
    
    def measure_channel(self, qubit: int) -> MeasureChannel:
        return self.measure_channels[qubit]
    
    def qubit_properties(self, qubit: int) -> dict:
        return self.base_backend.qubit_properties(qubit)
    
    def run(self, run_input: Union[QuantumCircuit, Schedule, ScheduleBlock], **options) -> Result:
        pass