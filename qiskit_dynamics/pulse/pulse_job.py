
from qiskit.providers import JobV1 as Job
from qiskit.providers.backend import Backend, BackendV1, BackendV2
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

