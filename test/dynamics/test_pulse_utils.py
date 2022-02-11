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
# pylint: disable=invalid-name

"""Tests for pulse_utils.py."""

from re import sub
import numpy as np
from collections import Counter
from qiskit_dynamics.pulse.pulse_utils import (
    compute_probabilities,
    convert_to_dressed,
    labels_generator,
    sample_counts,
)
from typing import List
from .common import QiskitDynamicsTestCase

RANDOM_SEED=123


def basis_vec(ind, dimension):
    vec = np.zeros(dimension, dtype=complex)
    vec[ind] = 1.0
    return vec


def two_q_basis_vec(inda, indb, dimension):
    vec_a = basis_vec(inda, dimension)
    vec_b = basis_vec(indb, dimension)
    return np.kron(vec_a, vec_b)


def get_dressed_state_index(inda, indb, dimension, evectors):
    b_vec = two_q_basis_vec(inda, indb, dimension)
    overlaps = np.abs(evectors @ b_vec)
    return overlaps.argmax()


def get_dressed_state_and_energy_3x3(evals, inda, indb, dimension, evecs):
    ind = get_dressed_state_index(inda, indb, dimension, evecs)
    return evals[ind], evecs[ind]


def generate_ham(subsystem_dims: List[int]) -> np.ndarray:
    """Generate a hamiltonian of up to 3 subsystems with arbitrary dimensions and preset variables.

    Args:
        subsystem_dims (List[int]): Dimensions of the subsystems of the hamiltonian.

    Returns:
        np.ndarray: Some hamiltonian.
    """
    dim = subsystem_dims[0]
    if len(subsystem_dims) > 1:
        dim1 = subsystem_dims[1]
        ident2q = np.eye(dim * dim1)
        a1 = np.diag(np.sqrt(np.arange(1, dim1)), 1)
        adag1 = a1.transpose()
        N_1 = np.diag(np.arange(dim1))
        ident1 = np.eye(dim1)
    if len(subsystem_dims) == 3:
        dim2 = subsystem_dims[2]
        ident3q = np.eye(dim * dim1 * dim2)
        a2 = np.diag(np.sqrt(np.arange(1, dim2)), 1)
        adag2 = a2.transpose()
        N_2 = np.diag(np.arange(dim2))
        ident2 = np.eye(dim2)

    w_c = 2 * np.pi * 5.105
    w_t = 2 * np.pi * 5.033
    w_2 = 2 * np.pi * 5.53
    alpha_c = 2 * np.pi * (-0.33534)
    alpha_t = 2 * np.pi * (-0.33834)
    alpha_2 = 2 * np.pi * (-0.33234)
    J = 2 * np.pi * 0.002
    J2 = 2 * np.pi * 0.0021

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.transpose()
    N = np.diag(np.arange(dim))
    ident = np.eye(dim)

    if len(subsystem_dims) == 1:
        # operators on the control qubit (first tensor factor)
        N0 = N

        H0 = w_c * N0 + 0.5 * alpha_c * N0 @ (N0 - ident)

    elif len(subsystem_dims) == 2:

        # operators on the control qubit (first tensor factor)
        a0 = np.kron(a, ident1)
        adag0 = np.kron(adag, ident1)
        N0 = np.kron(N, ident1)

        # operators on the target qubit (first tensor factor)
        a1 = np.kron(ident, a1)
        adag1 = np.kron(ident, adag1)
        N1 = np.kron(ident, N_1)
        H0 = (
            w_c * N0
            + 0.5 * alpha_c * N0 @ (N0 - ident2q)
            + w_t * N1
            + 0.5 * alpha_t * N1 @ (N1 - ident2q)
            + J * (a0 @ adag1 + adag0 @ a1)
        )
    elif len(subsystem_dims) == 3:

        # operators on the control qubit (first tensor factor)
        a0 = np.kron(a, ident1)
        a0 = np.kron(a0, ident2)
        adag0 = np.kron(adag, ident1)
        adag0 = np.kron(adag0, ident2)
        N0 = np.kron(N, ident1)
        N0 = np.kron(N0, ident2)

        # operators on the target qubit (first tensor factor)
        a1 = np.kron(ident, a1)
        a1 = np.kron(a1, ident2)
        adag1 = np.kron(ident, adag1)
        adag1 = np.kron(adag1, ident2)
        N1 = np.kron(ident, N_1)
        N1 = np.kron(N_2, N1)

        # operators on the third qubit (first tensor factor)
        a2 = np.kron(ident1, a2)
        a2 = np.kron(ident, a2)
        adag2 = np.kron(ident1, adag2)
        adag2 = np.kron(ident, adag2)
        N2 = np.kron(ident1, N_2)
        N2 = np.kron(ident, N2)

        H0 = (
            w_c * N0
            + 0.5 * alpha_c * N0 @ (N0 - ident3q)
            + w_t * N1
            + 0.5 * alpha_t * N1 @ (N1 - ident3q)
            + w_2 * N2
            + 0.5 * alpha_2 * N2 @ (N2 - ident3q)
            + J * (a0 @ adag1 + adag0 @ a1)
            + J2 * (a1 @ adag2 + adag1 @ a2)
        )
    return H0
class TestDressedStateConverter(QiskitDynamicsTestCase):
    """DressedStateConverter tests"""

    def assertAllClose(self, A, B, rtol=1e-8, atol=1e-8):
        """Call np.allclose and assert true."""

        self.assertTrue(np.allclose(A, B, rtol=rtol, atol=atol))

    def dressed_tester(self, dressed_states, subsystem_dims):
        labels = labels_generator(subsystem_dims, array=True)
        str_labels = labels_generator(subsystem_dims, array=False)
        for str_label, label in zip(str_labels, labels):
            id = np.argmax(np.abs(dressed_states[str_label]))
            labels[id]
            self.assertTrue((labels[id] == label))

    def test_convert_to_dressed_single_q(self):
        """Test convert_to_dressed with a single 3 level qubit system."""
        # How to test dressed frequencies without just rewriting my code?

        subsystem_dims = [3]
        H0 = generate_ham(subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, = convert_to_dressed(
            H0, subsystem_dims
        )
        self.dressed_tester(dressed_states, subsystem_dims)

        dressed_states_manual = {'0': [1.,0.,0.], '1': [0.,1.,0.], '2':[0.,0.,1.]}
        dressed_freqs_manual = [5.104999999498378]
        dressed_evals_manual = {'0': 0., '1': 32.07566099, '2': 62.04431863}

        self.assertTrue(dressed_states_manual.keys() == dressed_states.keys())
        self.assertAllClose(list(dressed_states_manual.values()), list(dressed_states.values()))
        self.assertTrue(dressed_evals_manual.keys() == dressed_evals.keys())
        self.assertAllClose(list(dressed_evals_manual.values()), list(dressed_evals.values()))
        self.assertAllClose(dressed_freqs_manual, dressed_freqs)

    def test_convert_to_dressed_two_q_states(self):
        """Test convert_to_dressed with a 2 qubit system with 3 levels per qubit."""
        """also test state and energy using alternative method"""
        subsystem_dims = [3, 3]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, = convert_to_dressed(
            H0, subsystem_dims
        )

        dim = subsystem_dims[0]
        evals, evectors = np.linalg.eigh(H0)


        E00, dressed00 = get_dressed_state_and_energy_3x3(evals, 0, 0, dim, evectors.transpose())
        E01, dressed01 = get_dressed_state_and_energy_3x3(evals, 0, 1, dim, evectors.transpose())
        E02, dressed02 = get_dressed_state_and_energy_3x3(evals, 0, 2, dim, evectors.transpose())
        E10, dressed10 = get_dressed_state_and_energy_3x3(evals, 1, 0, dim, evectors.transpose())
        E11, dressed11 = get_dressed_state_and_energy_3x3(evals, 1, 1, dim, evectors.transpose())
        E12, dressed12 = get_dressed_state_and_energy_3x3(evals, 1, 2, dim, evectors.transpose())
        E20, dressed20 = get_dressed_state_and_energy_3x3(evals, 2, 0, dim, evectors.transpose())
        E21, dressed21 = get_dressed_state_and_energy_3x3(evals, 2, 1, dim, evectors.transpose())
        E22, dressed22 = get_dressed_state_and_energy_3x3(evals, 2, 2, dim, evectors.transpose())

        if dressed00[np.argmax(np.abs(dressed00))] < 0: dressed00 = -1 * dressed00
        if dressed01[np.argmax(np.abs(dressed01))] < 0: dressed01 = -1 * dressed01
        if dressed02[np.argmax(np.abs(dressed02))] < 0: dressed02 = -1 * dressed02
        if dressed10[np.argmax(np.abs(dressed10))] < 0: dressed10 = -1 * dressed10
        if dressed11[np.argmax(np.abs(dressed11))] < 0: dressed11 = -1 * dressed11
        if dressed12[np.argmax(np.abs(dressed12))] < 0: dressed12 = -1 * dressed12
        if dressed20[np.argmax(np.abs(dressed20))] < 0: dressed20 = -1 * dressed20
        if dressed21[np.argmax(np.abs(dressed21))] < 0: dressed21 = -1 * dressed21
        if dressed22[np.argmax(np.abs(dressed22))] < 0: dressed22 = -1 * dressed22

        dressed_states = {key: (-1 * eval) if eval[np.argmax(np.abs(eval))] < 0 else eval for key, eval in dressed_states.items()}

        self.assertAllClose(dressed00, dressed_states['00'])
        self.assertAllClose(dressed01, dressed_states['01'])
        self.assertAllClose(dressed02, dressed_states['02'])
        self.assertAllClose(dressed10, dressed_states['10'])
        self.assertAllClose(dressed11, dressed_states['11'])
        self.assertAllClose(dressed12, dressed_states['12'])
        self.assertAllClose(dressed20, dressed_states['20'])
        self.assertAllClose(dressed21, dressed_states['21'])
        self.assertAllClose(dressed22, dressed_states['22'])

        self.assertAllClose(E00, dressed_evals["00"])
        self.assertAllClose(E01, dressed_evals["01"])
        self.assertAllClose(E02, dressed_evals["02"])
        self.assertAllClose(E10, dressed_evals["10"])
        self.assertAllClose(E11, dressed_evals["11"])
        self.assertAllClose(E12, dressed_evals["12"])
        self.assertAllClose(E20, dressed_evals["20"])
        self.assertAllClose(E21, dressed_evals["21"])
        self.assertAllClose(E22, dressed_evals["22"])

        # self.assertTrue(np.max(dressed00 - dressed_states["00"] < 1e-12))
        # self.assertTrue(np.max(dressed01 - dressed_states["01"] < 1e-12))
        # self.assertTrue(np.max(dressed10 - dressed_states["10"] < 1e-12))
        # self.assertTrue(np.max(dressed11 - dressed_states["11"] < 1e-12))

        # self.assertTrue(E00 - dressed_evals["00"] < 1e-12)
        # self.assertTrue(E01 - dressed_evals["01"] < 1e-12)
        # self.assertTrue(E10 - dressed_evals["10"] < 1e-12)
        # self.assertTrue(E11 - dressed_evals["11"] < 1e-12)

        self.dressed_tester(dressed_states, subsystem_dims)

    def test_convert_to_dressed_three_q_states(self):
        """Test convert_to_dressed with a 3 qubit system with different levels per qubit."""
        subsystem_dims = [3, 4, 5]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, = convert_to_dressed(
            H0, subsystem_dims
        )

        self.dressed_tester(dressed_states, subsystem_dims)

    def test_convert_to_dressed_three_q_states_high(self):
        """Test convert_to_dressed with a 3 qubit system with different levels per qubit."""
        subsystem_dims = [6, 8, 4]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, = convert_to_dressed(
            H0, subsystem_dims
        )

        self.dressed_tester(dressed_states, subsystem_dims)


class TestComputeandSampleProbabilities(QiskitDynamicsTestCase):
    """
    How do we test compute probabilities? We can just take our systems
    """

    def test_compute_and_sample_probabilities_1q(self):
        "Test compute_probabilities for a 1q system"
        subsystem_dims = [3]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals = convert_to_dressed(H0, subsystem_dims)
        state = [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
        probs = compute_probabilities(state, basis_states=dressed_states)

        self.assertTrue(sum(list(probs.values())) == 1)

        self.assertTrue(1 - sum(probs.values()) < 1e-12)
        self.assertTrue(0.5 - probs["1"] < 1e-16)
        self.assertTrue(0.5 - probs["2"] < 1e-16)

        samples = sample_counts(probs, 1000, seed=RANDOM_SEED)
        counts = Counter(samples)
        self.assertTrue(counts["1"] == 501)
        self.assertTrue(counts["2"] == 499)

    def test_compute_and_sample_probabilities_2q(self):
        "Test compute_probabilities for a 2q system"
        subsystem_dims = [3, 4]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, _, _ = convert_to_dressed(H0, subsystem_dims)
        state1 = [1 / np.sqrt(2), 0, 1 / np.sqrt(2)]
        state2 = [0, 1, 0, 0]
        state = np.kron(state2, state1)

        probs = compute_probabilities(state, basis_states=dressed_states)

        self.assertTrue(1 - sum(probs.values()) < 1e-12)

        self.assertTrue(0.5 - probs["12"] < 1e-4)
        self.assertTrue(0.5 - probs["10"] < 1e-4)

        samples = sample_counts(probs, 1000, seed=RANDOM_SEED)
        counts = Counter(samples)
        self.assertTrue(counts["12"] > 450)
        self.assertTrue(counts["10"] > 450)

    def test_compute_and_sample_probabilities_3q(self):
        "Test compute_probabilities for a 3q system"
        subsystem_dims = [3, 6, 3]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, _, _ = convert_to_dressed(H0, subsystem_dims)
        state1 = [1 / np.sqrt(2), 0, 1 / np.sqrt(2)]
        state2 = [0, 1 / np.sqrt(4), 0, 1 / np.sqrt(4), 1 / np.sqrt(4), 1 / np.sqrt(4)]
        state3 = [1, 0, 0]
        state = np.kron(state3, state2)
        state = np.kron(state, state1)
        probs = compute_probabilities(state, basis_states=dressed_states)
        self.assertTrue(1 - sum(probs.values()) < 1e-12)

        self.assertTrue(0.125 - probs["010"] < 1e-4)
        self.assertTrue(0.125 - probs["030"] < 1e-4)
        self.assertTrue(0.125 - probs["040"] < 1e-4)
        self.assertTrue(0.125 - probs["050"] < 1e-4)
        self.assertTrue(0.125 - probs["012"] < 1e-4)
        self.assertTrue(0.125 - probs["032"] < 1e-4)
        self.assertTrue(0.125 - probs["042"] < 1e-4)
        self.assertTrue(0.125 - probs["052"] < 1e-4)

        samples = sample_counts(probs, 1000, seed=RANDOM_SEED)
        counts = Counter(samples)
        self.assertTrue(counts["010"] > 100)
        self.assertTrue(counts["030"] > 100)
        self.assertTrue(counts["040"] > 100)
        self.assertTrue(counts["050"] > 100)
        self.assertTrue(counts["012"] > 100)
        self.assertTrue(counts["032"] > 100)
        self.assertTrue(counts["042"] > 100)
        self.assertTrue(counts["052"] > 100)

    # def test_compute_probabilities_density_matrix(self):
    # def test_compute_probabilities_density_matrix_terra(self):
    # def test_compute_probabilities_density_statevector_terra(self):
