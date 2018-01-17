"""
Description:
    This module provides unit tests for the functionality provided in qm.py

Author:
    Johannes Cartus, TU Graz
"""

import unittest

import numpy as np

from utilities import InfoStream
from qm import SpinState, Sector, simulate_heisenberg_model

class TestStates(unittest.TestCase):
    """This will test the basis state's functionality (Magnetisation, etc.)."""

    def setUp(self):

        self._test_state = SpinState(state=np.array([1, 0, 0, 1, 1]))

    def test_bin_to_dec(self):
        #carefull lowest bit at index 0, i.e. at the left ;)
        bin = [
            np.array([0, 1, 0, 0, 1]),
            np.array([1, 0, 1]),
            np.array([1, 0, 0, 1, 1, 0])
        ]

        expected = [18, 5, 25]

        # init with some random value
        state = SpinState(2,4)
        
        for (b,e) in zip(bin, expected):
            self.assertEqual(e, state._binary_to_decimal(b))

    def test_dec_to_bin(self):
        #decimals and msb
        decimal = [(18, 4), (18, 5), (5, 2), (25, 5)]
        expected = [
            np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        ]

        # init with some random value
        state = SpinState(2,4)
        
        for (d,e) in zip(decimal, expected):
            self.assertListEqual(list(e), list(state._decimal_to_binary(*d)))

        # check if out of range assersts
        with self.assertRaises(ValueError):
            state._decimal_to_binary(22, 3)

    def test_magnetisation(self):
        self.assertEqual(
            (0.5 - (-0.5) + (-0.5) - (+0.5) + 0.5) / 5,
            self._test_state.magnetisation()
        )

    def test_magnetisation_two_spins(self):
        state_all_up = SpinState(np.array([1, 1]))
        self.assertEqual(0, state_all_up.magnetisation())

        state_neel_1 =  SpinState(np.array([0, 1]))
        self.assertEqual(-0.5, state_neel_1.magnetisation())

        state_neel_2  = SpinState(np.array([1, 0]))
        self.assertEqual(0.5, state_neel_2.magnetisation())

    def test_energy(self):

        self.assertEqual(
            0,
            self._test_state.energy(jz=0)
        )

        self.assertEqual(
            - 0.25 + 0.25 - 0.25 + 0.25 + 0.25,
            self._test_state.energy(jz=1)
        )

class TestBasisGeneration(unittest.TestCase):
    """This class tests the generation of possible basis states for given 
    lattice size and number of spin-ups"""

    def test_basis_generation(self):
        sector = Sector(
            number_of_sites=4, 
            number_spinups=2, 
            jz=1
        )

        sector.setup_basis()

        expected = [3, 5, 6, 9, 10, 12]
        actual = [x.decimal for x in sector.basis]

        self.assertListEqual(expected, actual)


    def test_hilbertspace_sizes(self):
        """Setup basis for given setup and compare to given results (see 
        instructions)"""

        # N = 8, nUp = 4 (Sz=0), Jz = 0
        expected = 70
        basis = Sector(8, 4, 0).setup_basis()
        self.assertEqual(expected, len(basis))        

        # N = 14, nUp = 7 (Sz=0), Jz = 0
        expected = 3432
        basis = Sector(14, 7, 0).setup_basis()
        self.assertEqual(expected, len(basis))        

        # N = 20, nUp = 10 (Sz=0), Jz = 0
        expected = 184756
        basis = Sector(20, 10, 0).setup_basis()
        self.assertEqual(expected, len(basis))

class TestHamiltonianGeneration(unittest.TestCase):
    """Test the setup of the hamiltonian matrix for given N, nUp. 
    
    Note:
        Implicitly in all tests the basis is generated first. Thus, these tests 
        may fail if there is an error there...
    """

    def test_generate_2SpinSystem_Sz0(self):
        """Setup H for 2 spin-system and compare results to what is given
        in the script for Sz_tot = 0 (center of the matrix)"""
        # Sz = 0
        for Jz in [0, 1, 2]: 
            #expected = np.array([[-Jz/4, 1/2], [1/2, -Jz/4]]) # laut skript
            expected = np.array([[-Jz/2, 1], [1, -Jz/2]]) # was ich mir so denk
            H = Sector(2, 1, Jz).setup_hamiltonian().toarray()
            np.testing.assert_array_almost_equal(expected, H)


    def test_H_generation_N4_nUp2(self):
        
        for J in [0, 1, 2]:
            sector = Sector(
                number_of_sites=4, 
                number_spinups=2, 
                jz=J
            )

            sector.setup_hamiltonian()        

            expected = np.array(
                [
                    [0, 0.5, 0, 0, 0.5, 0 ], 
                    [0.5, -J, 0.5, 0.5, 0, 0.5 ],
                    [0, 0.5, 0, 0, 0.5, 0 ],
                    [0, 0.5, 0, 0, 0.5, 0 ],
                    [0.5, 0, 0.5, 0.5, -J, 0.5 ],
                    [0, 0.5, 0, 0, 0.5, 0 ]
                ]
            )

            np.testing.assert_array_equal(expected, sector.H.toarray())
class TestDiagonalisation(unittest.TestCase):
    """Test the lanczos algorithm"""
    def test_lanczos_small(self):

        A = np.array([[2, 1], [1, 2]])
        a = 1
        v = np.array([1,-1])

        a_act, v_act, _ = Sector.lanczos_diagonalisation(A)
        self._assert_eig_result(a, v, a_act, v_act, 1E-4)

    def test_lanzos_middle(self):

        A = np.array(
            [
                [ 3, 2, 4, 0, -2 ], 
                [ 2, -2, 6, -2, 1 ], 
                [ 4, 6, 2, 4, 4 ], 
                [ 0, -2, 4, 7, 6 ], 
                [ -2, 1, 4, 6, -9 ]
            ]
        )
        a = -12.0509
        v = np.array([0.204647, -0.04609, -0.246984, -0.267927, 1])

        a_act, v_act, _ = Sector.lanczos_diagonalisation(A)
        self._assert_eig_result(a, v, a_act, v_act, 1E-4)
        

    def test_lanczos_random_10x10(self):

        N = 10
        
        # create a hermitian matrix
        A = np.random.rand(N, N)
        A = A + A.T

        # expected:
        energies, vectors = np.linalg.eigh(A)
        E_expected = energies[0]
        v_expected = vectors[:, 0]

        # actual:
        E_actual, v_actual, _ = Sector.lanczos_diagonalisation(A)

        self._assert_eig_result(E_expected, v_expected, E_actual, v_actual)
    
    def _assert_eig_result(self, a, v, a_act, v_act, delta=1E-7):
        
        self.assertAlmostEqual(a, a_act, delta=delta)

        self._assert_vector_match(v, v_act, delta=delta)


    def _assert_vector_match(self, a, b, delta=1E-7):

        self.assertEqual(len(a), len(b))

        # vectors can only differ by a constant factor in all elements
        self.assertAlmostEqual(np.var(a / b), 0, delta=delta)

    def test_highlevel_N2(self):

        # todo

        sector = Sector(
            number_of_sites=2,
            number_spinups=1,
            jz=1
        )

        E, ground_state = sector.calculate_ground_state()

        self.assertAlmostEqual(0, ground_state.magnetisation(), delta=1E-7)
        self.assertAlmostEqual(0.25, ground_state.magnetisation_squared(), delta=1E-7)
        


    def test_highlevel_N6_Jz0_Sz0(self):
        """
        N=6,
        Sz=0 => Nup=3,
        Jz=0

        Lt. Markus Aichhorn diagonal in 6 Schritten
        """

        sector = Sector(
            number_of_sites=6,
            number_spinups=3,
            jz=0
        )

        steps_expected = 6
        E_expected = -2.000
        m_expected = 0.0
        m2_expected = 0.08333333
        correlation_expected = [
            0.25, -0.1111111, -0.0, -0.0277777, 0.0, -0.1111111
        ]

        E, ground_state, steps = \
            sector.calculate_ground_state(give_iterations=True)

        # check iterations
        self.assertTrue(steps <= steps_expected)

        # check energy
        self.assertAlmostEqual(E_expected, E, delta=1E-4)

        # check magnetisations
        self.assertAlmostEqual(m_expected, ground_state.magnetisation(), delta=1E-4)
        self.assertAlmostEqual(m2_expected, ground_state.magnetisation_squared(), delta=1E-2)

        for exp, act in zip(correlation_expected, ground_state.correlation()):
            self.assertAlmostEqual(exp, act, delta=1E-2)

    def test_highlevel_N10_J2_Sz0(self):
        """
        N=10,
        Sz=0 => Nup=5,
        Jz=2

        Lt. Markus Aichhorn diagonal in 21 Schritten
        """

        sector = Sector(
            number_of_sites=10,
            number_spinups=5,
            jz=2
        )

        steps_expected = 21
        E_expected = -6.24458366
        m_expected = 0.0
        m2_expected = 0.15776331
        correlation_expected = [
            0.25, 
            -0.19414343, 
            0.14145689, 
            -0.13587744, 
            0.12795200,
            -0.12877517,
            0.12795171,
            -0.13587788,
            0.14145712,
            -0.19414381
        ]

        E, ground_state, steps = \
            sector.calculate_ground_state(give_iterations=True)

        # check iterations
        self.assertTrue(steps <= steps_expected)

        # check energy
        self.assertAlmostEqual(E_expected, E, delta=1E-4)

        # check magnetisations
        self.assertAlmostEqual(m_expected, ground_state.magnetisation(), delta=1E-4)
        self.assertAlmostEqual(m2_expected, ground_state.magnetisation_squared(), delta=1E-2)

        for exp, act in zip(correlation_expected, ground_state.correlation()):
            self.assertAlmostEqual(exp, act, delta=1E-2)





if __name__ == '__main__':
    InfoStream.suppress_level=1
    unittest.main()
    