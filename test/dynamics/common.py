# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,isinstance-second-argument-not-valid-type

"""
Shared functionality and helpers for the unit tests.
"""

import unittest
from typing import Callable, Iterable
import numpy as np
from scipy.sparse import issparse

try:
    from jax import jit, grad
except ImportError:
    pass

from qiskit_dynamics import dispatch
from qiskit_dynamics.array import Array, wrap


class QiskitDynamicsTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""

    def assertAllClose(self, A, B, rtol=1e-8, atol=1e-8):
        """Call np.allclose and assert true."""
        A = Array(A)
        B = Array(B)

        self.assertTrue(np.allclose(A, B, rtol=rtol, atol=atol))

    def assertAllCloseSparse(self, A, B, rtol=1e-8, atol=1e-8):
        """Call np.allclose and assert true. Converts A and B to arrays and then calls np.allclose.
        Assumes that A and B are either sparse matrices or lists of sparse matrices"""

        if issparse(A):
            A = A.toarray()
            B = B.toarray()
        elif isinstance(A, Iterable) and issparse(A[0]):
            A = [item.toarray() for item in A]
            B = [item.toarray() for item in B]

        self.assertTrue(np.allclose(A, B, rtol=rtol, atol=atol))


class TestJaxBase(unittest.TestCase):
    """Base class with setUpClass and tearDownClass for setting jax as the
    default backend.

    Test cases that inherit from this class will automatically work with jax
    backend.
    """

    @classmethod
    def setUpClass(cls):
        try:
            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

        dispatch.set_default_backend("jax")

    @classmethod
    def tearDownClass(cls):
        """Set numpy back to the default backend."""
        dispatch.set_default_backend("numpy")

    def jit_wrap(self, func_to_test: Callable) -> Callable:
        """Wraps and jits func_to_test.
        Args:
            func_to_test: The function to be jited.
        Returns:
            Wrapped and jitted function."""
        wf = wrap(jit, decorator=True)
        return wf(wrap(func_to_test))

    def jit_grad_wrap(self, func_to_test: Callable) -> Callable:
        """Tests whether a function can be graded. Converts
        all functions to scalar, real functions if they are not
        already.
        Args:
            func_to_test: The function whose gradient will be graded.
        Returns:
            JIT-compiled gradient of function."""
        wf = wrap(lambda f: jit(grad(f)), decorator=True)
        f = lambda *args: np.sum(func_to_test(*args)).real
        return wf(f)


class TestQutipBase(unittest.TestCase):
    """Base class with setUpClass for tests that utilize Qutip

    Test cases that inherit from this class will automatically work with jax
    backend.
    """

    @classmethod
    def setUpClass(cls):
        try:
            # pylint: disable=import-outside-toplevel,unused-import
            import qutip
        except Exception as err:
            raise unittest.SkipTest("Skipping qutip tests.") from err
