#%%
from functools import reduce
from qiskit import QiskitError
import uuid
from random import sample
import datetime
import time
from typing import Dict, Iterable, List, Optional, Union
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import
from qiskit.result.models import ExperimentResult#, Result
import logging
from qiskit.compiler import transpile

from .utils import requires_submit

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

def pad_schedules(backend, schedules):
    return schedules



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
    elif isinstance(experiment, QuantumCircuit):
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
    
    # Remove control channels until I figure them out
    drive_channels = [chan for chan in reduced_channels if 'd' in chan]
    control_channels = [chan for chan in reduced_channels if 'u' in chan]

    channel_freq_dict = {channel: freq for channel, freq in zip(reduced_channels, [freq for i,freq in enumerate(backend.defaults().qubit_freq_est) if i in subsystem_list])}
    # control_freq_dict = {channel: freq for }
    for edge in backend.coupling_map.get_edges():
        channel_freq_dict[backend.control_channel(edge)[0].name] = backend.defaults().qubit_freq_est[edge[0]]
    
    channel_freq_dict


    
    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=hamiltonian_operators,
        rotating_frame=np.diag(static_hamiltonian),
        dt = backend.dt,
        hamiltonian_channels=reduced_channels,
        channel_carrier_freqs=channel_freq_dict
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
    
    def acquire_channel(self, qubit: Iterable[int]) -> Union[int, AcquireChannel, None]:
        return self._acquire_channel(qubit)
    
    def drive_channel(self, qubit: int) -> Union[int, DriveChannel, None]:
        return self._drive_channel(qubit)

    def control_channel(self, qubit: int) -> Union[int, ControlChannel, None]:
        return self._control_channel(qubit)
    
    def measure_channel(self, qubit: int) -> Union[int, MeasureChannel, None]:
        return self._measure_channel(qubit)
    
    def run(self, run_input: Union[QuantumCircuit, Schedule, ScheduleBlock], y0, t_span, **options) -> Result:
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

        # Run the schedule

        validation = validate_experiments(run_input)
        if validation != "Success":
            raise QiskitError(f"Validation of experiment failed with error message: {validation}")
        # Transpile the circuits? Do we validate before transpilation? what doe transpilation even mean here?
        # experiments = self._transpile(run_input)# unsure something like this, convert whatever input to the schedules I guess
        if isinstance(run_input, QuantumCircuit):
            raise NotImplementedError("No quantum circuits yet")
            experiments = transpile(run_input, backend=self)
        else:
            experiments = run_input

        job_id = str(uuid.uuid4())
        # pulse_job = pulseJob(experiments=experiments, backend=self, job_id=job_id)
        # pulse_job.submit()
        # print(experiments)
        output = self._run(experiments, y0, t_span, job_id)
        return output
        

        # return pulse_job


    def _run(self, experiments, y0, t_span, job_id='', format_result=True,):
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
        # schedules = pad_schedules(self, schedules)
        # schedu
        output = self._execute(experiments, y0=y0, t_span=t_span)

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
            raise QiskitError(
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
    
    def _execute(self, schedule: Schedule, y0, t_span):
        return ExperimentResult_from_sol(self.solver.solve(t_span = t_span, y0=y0, signals=schedule))

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
        """Return the system time resolGution of output signals

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
from qiskit.providers import JobStatus, JobError
from .utils import DEFAULT_EXECUTOR, requires_submit
class pulseJob(Job):
    def __init__(self, experiments, backend: Backend, job_id: str, **kwargs) -> None:
        super().__init__(backend=backend, job_id=job_id, **kwargs)
        self._executor = DEFAULT_EXECUTOR
        self.experiments = experiments
        self._future = None
        self._fn = backend.run
    
    def submit(self):
        """Submit the job to the backend for execution.

        Raises:
            QobjValidationError: if the JSON serialization of the Qobj passed
            during construction does not validate against the Qobj schema.
            JobError: if trying to re-submit the job.
        """
        if self._future is not None:
            raise JobError("Aer job has already been submitted.")
        self._future = self._executor.submit(self._fn, self.experiments, self._job_id)
        

    @requires_submit
    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        """Get job result. The behavior is the same as the underlying
        concurrent Future objects,

        https://docs.python.org/3/library/concurrent.futures.html#future-objects

        Args:
            timeout (float): number of seconds to wait for results.

        Returns:
            qiskit.Result: Result object

        Raises:
            concurrent.futures.TimeoutError: if timeout occurred.
            concurrent.futures.CancelledError: if job cancelled before completed.
        """
        return self._future.result(timeout=timeout)

    @requires_submit
    def cancel(self):
        """Attempt to cancel the job."""
        return self._future.cancel()

    @requires_submit
    def status(self):
        """Gets the status of the job by querying the Python's future

        Returns:
            JobStatus: The current JobStatus

        Raises:
            JobError: If the future is in unexpected state
            concurrent.futures.TimeoutError: if timeout occurred.
        """
        # The order is important here
        if self._future.running():
            _status = JobStatus.RUNNING
        elif self._future.cancelled():
            _status = JobStatus.CANCELLED
        elif self._future.done():
            _status = JobStatus.DONE if self._future.exception() is None else JobStatus.ERROR
        else:
            # Note: There is an undocumented Future state: PENDING, that seems to show up when
            # the job is enqueued, waiting for someone to pick it up. We need to deal with this
            # state but there's no public API for it, so we are assuming that if the job is not
            # in any of the previous states, is PENDING, ergo INITIALIZING for us.
            _status = JobStatus.INITIALIZING
        return _status

    def backend(self):
        """Return the instance of the backend used for this job."""
        return self._backend

    def qobj(self):
        """Return the Qobj submitted for this job.

        Returns:
            Qobj: the Qobj submitted for this job.
        """
        return self._qobj

    def executor(self):
        """Return the executor for this job"""
        return self._executor


# %%
