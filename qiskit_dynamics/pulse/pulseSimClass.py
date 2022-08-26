#%%
from qiskit import QiskitError
import uuid
from random import sample
import datetime
import time
from typing import Dict, Iterable, List, Optional, Union
from qiskit.test.mock import FakeVigo
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import
from qiskit.result.models import ExperimentResult#, Result
import logging

from .pulse_to_signals import InstructionToSignals

#%%
# from typing import Union
from qiskit.providers.backend import Backend, BackendV1, BackendV2
from qiskit.pulse.channels import AcquireChannel, DriveChannel, MeasureChannel, ControlChannel

from qiskit_dynamics import Solver
from qiskit_dynamics.pulse.backend_parser.string_model_parser import parse_hamiltonian_dict

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import Schedule, ScheduleBlock#, block_to_schedule

from qiskit.transpiler import Target
from qiskit.result import Result

from qiskit_dynamics.pulse.pulse_utils import sample_counts, compute_probabilities, convert_to_dressed

#Logger
logger = logging.getLogger(__name__)

def validate_experiments(experiment):
    if isinstance(experiment, list):
        if isinstance(experiment[0], Schedule):
            return "Success"
        elif isinstance(experiment[0], ScheduleBlock):
            return "Success"
    elif isinstance(experiment, Schedule):
        return "Success"
    elif isinstance(experiment, ScheduleBlock):
        return "Success"
    else: 
        return f"Experiment type {type(experiment)} not yet supported"
def get_counts(state_vector: np.ndarray, n_shots: int, seed: int) -> Dict[str, int]:
    """
    Get the counts from a state vector.
    :param state_vector: The state vector.
    :return: The counts.
    """
    
    probs = compute_probabilities(state_vector, basis_states=convert_to_dressed(static_ham, subsystem_dims=subsystem_dims))
    counts = sample_counts(probs,n_shots=n_shots,seed=seed)
    return counts

def solver_from_backend(backend: Backend, subsystem_list: List[int]) -> 'PulseSimulator':
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

def ExperimentResult_from_sol(sol: OdeResult, return_type: str) -> ExperimentResult:
    """
    Get the data from a solver object.
    :param sol: The solver object.
    :param return_type: The type of data to return.
    :return: ExperimentResult.
    """

    if return_type == 'state_vector':
        result_data = {'state_vector': sol.y}
    elif return_type == 'unitary':
        result_data = {'unitary': sol.y}
    elif return_type == 'counts':
        result_data = {'counts': get_counts(sol.y)}
    else:
        raise NotImplementedError(f"Return type {return_type} not implemented.")


    return ExperimentResult.from_dict(result_data)

def result_from_sol(sol: Union[OdeResult, List[OdeResult]]) -> Result:
    """
    Create a result object from a solver object.
    :param sol: The solver object.
    :return: A Result object.
    """


    if isinstance(sol, list):
        result = Result()
        for s in sol:
            s = get_data_from_sol(s)
            result.add_counts(s.counts)
            result.add_data(s.data)
            result.add_final_state(s.final_state)
            result.add_initial_state(s.initial_state)
            result.add_time(s.time)
    else:
        sol = get_data_from_sol(sol)
        result = Result(counts=sol.counts, data=sol.data, final_state=sol.final_state, initial_state=sol.initial_state, time=sol.time)
    return result

# Do we want the hamiltonian and the operators to be separate?
# We could also have no init, and just have users init with a solver, or use simulator.from_
class PulseSimulator(BackendV2):
    # def __init__(self, solver: Solver, acquire_channels=None, control_channels=None, measure_channels=None, drive_channels=None):
    def __init__(self, solver: Solver=None, static_hamiltonian=None, hamiltonian_operators=None, dt=None, acquire_channels=None, control_channels=None, measure_channels=None, drive_channels=None):
        if Solver is None:
            if static_hamiltonian is None:
                raise QiskitError("Static hamiltonian must be defined")
            if hamiltonian_operators is None:
                raise QiskitError("Hamiltonian operators must be defined")
            if dt is None:
                raise QiskitError("dt must be defined")
            self.solver = Solver(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                rotating_frame=np.diag(static_hamiltonian),
                dt=dt
            )
        else:
            if static_hamiltonian is not None:
                logger.warn("Solver arg used, passing in static_hamiltonian will have no effect")
            if hamiltonian_operators is not None:
                logger.warn("Solver arg used, passing in hamiltonian_operators will have no effect")
            if dt is not None:
                logger.warn("Solver arg used, passing in dt will have no effect")
            self.solver = solver
        super().__init__()



    @classmethod
    def from_backend(cls, backend: BackendV2, subsystem_list: Optional[List[int]] = None) -> 'PulseSimulator':
        """
        Create a PulseSimulator object from a backend.
        :param backend: The backend to use.
        :param subsystem_list: The qubits to use for the simulation.
        :return: A PulseSimulator object.
        """
        pulseSim = cls(solver=solver_from_backend(backend, subsystem_list))
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
    
    @classmethod
    def from_hamiltonian(self, static_hamiltonian, hamiltonian_operators, dt) -> 'PulseSimulator':
        """
        Create a `PulseSimulator` object from a hamiltonian.
        :param static_hamiltonian: The static hamiltonian.
        :param hamiltonian_operators: The hamiltonian operators."""

        pulseSim = PulseSimulator(solver=solver_from_hamiltonian(static_hamiltonian, hamiltonian_operators), dt=dt)
        pulseSim.target = Target()
    
        # Set various attributes from backend
        # Instantiate a `Qiskit-Dynamics` solver
        # set an attribute for the subsystem of the device to simulate
    
    def acquire_channel(self, qubit: Iterable[int]) -> Union[int, AcquireChannel, None]:
        return self._acquire_channel(qubit)
    
    def drive_channel(self, qubit: int) -> Union[int, DriveChannel, None]:
        return self._drive_channel(qubit)

    def control_channel(self, qubit: int) -> Union[int, ControlChannel, None]:
        return self._control_channel(qubit)
    
    def measure_channel(self, qubit: int) -> Union[int, MeasureChannel, None]:
        return self._measure_channel(qubit)
    
    def run(self, run_input: Union[QuantumCircuit, Schedule, ScheduleBlock], **options) -> Result:
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

        # TODO move to validator
        if isinstance(run_input, QuantumCircuit):
            raise NotImplementedError('Pulse Simulator does not currently support quantum circuits')
        # elif isinstance(run_input, ScheduleBlock):
        #     run_input = block_to_schedule(run_input)
        elif isinstance(run_input, Schedule):
            pass
        else:
            raise NotImplementedError(f'Pulse Simulator does not currently support run inputs of type {type(run_input)}')
        
        # Convert the schedule to signals
        # signalMapper = InstructionToSignals(self.dt, carriers=self.carrier_freqs, channels=self.channel_list)

        # signals = signalMapper.get_signals(run_input)
        # sol = self.solver.solve(t_span=None, y0=None, signals=run_input)
        # result = result_from_sol(sol, output_type, **kwargs)

        # Run the schedule
        experiments = self._transpile(circuits)# unsure something like this, convert whatever input to the schedules I guess
        validation = validate_experiments(experiments)
        if validation != "success":
            raise QiskitError(f"Validation of experiment failed with error message: {validation}")

        job_id = str(uuid.uuid4())
        # if isinstance(experiments, list):
        #     aer_job = AerJobSet(self, job_id, self._run, experiments, executor)
        # else:
        pulse_job = pulseJob(self, job_id, self._run, experiments)
        pulse_job.submit(experiments)

        return pulse_job
    def _run(self, experiments, job_id='', format_result=True):
        """Run a job"""
        # Start timer
        start = time.time()

        # # Take metadata from headers of experiments to work around JSON serialization error
        # metadata_list = []
        # metadata_index = 0
        # for expr in qobj.experiments:
        #     if hasattr(expr.header, "metadata"):
        #         metadata_copy = expr.header.metadata.copy()
        #         metadata_list.append(metadata_copy)
        #         expr.header.metadata.clear()
        #         if "id" in metadata_copy:
        #             expr.header.metadata["id"] = metadata_copy["id"]
        #         expr.header.metadata["metadata_index"] = metadata_index
                # metadata_index += 1

        # Run simulation
        output = self._execute(experiments)

        # Recover metadata
        # metadata_index = 0
        # for expr in qobj.experiments:
        #     if hasattr(expr.header, "metadata"):
        #         expr.header.metadata.clear()
        #         expr.header.metadata.update(metadata_list[metadata_index])
        #         metadata_index += 1

        # Validate output
        if not isinstance(output, dict):
            logger.error("%s: simulation failed.", self.name())
            if output:
                logger.error('Output: %s', output)
            raise AerError(
                "simulation terminated without returning valid output.")

        # Format results
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version

        # Add execution time
        output["time_taken"] = time.time() - start

        # Display warning if simulation failed
        if not output.get("success", False):
            msg = "Simulation failed"
            if "status" in output:
                msg += f" and returned the following error message:\n{output['status']}"
            logger.warning(msg)
        if format_result:
            return self._format_results(output)
        return output
    
    def _execute(self, schedule: Schedule):
        return format_output(self.solver.solve(t_span = self.t_span, y0=self.y0, signals=schedule))

    def get_solver(self):
        return self.solver
    
    def _default_options(self):
        pass
    
    def max_circuits(self):
        pass

    def target(self):
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
    def meas_map(self) -> List[List[int]]:
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
    


from qiskit.providers import JobV1 as Job
class pulseJob(Job):
    def __init__(self, backend: Backend, job_id: str, **kwargs) -> None:
        super().__init__(backend=backend, job_id=job_id, **kwargs)
    
    @requires_submit
    def submit(self, exp):
        self.result = self.backend.run(exp)

    @requires_submit
    def result(self):
        return self.result
    
    def status(self):
        raise NotImplementedError

