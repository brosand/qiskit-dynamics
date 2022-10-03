# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for Aer job management."""
import uuid
import copy
from math import ceil
from functools import singledispatch, update_wrapper, wraps
from concurrent.futures import ThreadPoolExecutor

from qiskit.result import ProbDistribution
from qiskit.quantum_info import Clifford
from .compatibility import Statevector, DensityMatrix, StabilizerState, Operator, SuperOp

from qiskit.providers import JobError
from functools import wraps

DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def requires_submit(func):
    """
    Decorator to ensure that a submit has been performed before
    calling the method.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._future is None:
            raise JobError("Job not submitted yet!. You have to .submit() first!")
        return func(self, *args, **kwargs)

    return _wrapper


def methdispatch(func):
    """
    Returns a wrapper function that selects which registered function
    to call based on the type of args[2]
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[2].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def format_save_type(data, save_type, save_subtype):
    """Format raw simulator result data based on save type."""
    init_fns = {
        "save_statevector": Statevector,
        "save_density_matrix": DensityMatrix,
        "save_unitary": Operator,
        "save_superop": SuperOp,
        "save_stabilizer": (lambda data: StabilizerState(Clifford.from_dict(data))),
        "save_clifford": Clifford.from_dict,
        "save_probabilities_dict": ProbDistribution,
    }
    # Non-handled cases return raw data
    if save_type not in init_fns:
        return data

    if save_subtype in ["list", "c_list"]:

        def func(data):
            init_fn = init_fns[save_type]
            return [init_fn(i) for i in data]

    else:
        func = init_fns[save_type]

    # Conditional save
    if save_subtype[:2] == "c_":
        return {key: func(val) for key, val in data.items()}

    return func(data)
