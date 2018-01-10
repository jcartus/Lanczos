"""
Todo:
    - tests für correlation in spin und composed states

"""

import unittest

import numpy as np

from qm import SpinState, HeisenbergSector

class TestStates(unittest.TestCase):

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

    def test_energy(self):
        
        pass

        self.assertEqual(
            - 0.25 + 0.25 - 0.25 + 0.25 + 0.25,
            self._test_state.energy()
        )


class TestSector(unittest.TestCase):

    def test_basis_generation(self):
        sector = HeisenbergSector(
            number_of_sites=4, 
            number_spinups=2, 
            jz=1
        )

        sector.setup_basis()

        expected = [3, 5, 6, 9, 10, 12]
        actual = [x.decimal for x in sector.basis]

        self.assertListEqual(expected, actual)

    def test_H_generation(self):
        sector = HeisenbergSector(
            number_of_sites=4, 
            number_spinups=2, 
            jz=1
        )

        sector.setup_hamiltonian()

        J = sector.Jz

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

        np.testing.assert_array_equal(expected, sector.H)

    def test_lanczos_small(self):
        
        sector = HeisenbergSector(
            number_of_sites=4, 
            number_spinups=2, 
            jz=1
        )

        A = np.array([[2, 1], [1, 2]])
        a = 1
        v = np.array([1,-1])

        self._assert_eig_result(a, v, sector.lanczos_diagonalisation(A))

    def test_lanzos_middle(self):
        sector = HeisenbergSector(
            number_of_sites=4, 
            number_spinups=2, 
            jz=1
        )

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

        self._assert_eig_result(a, v, sector.lanczos_diagonalisation(A), 1E-4)
        

    
    def _assert_eig_result(self, a, v, actual, delta=1E-7):
        
        self.assertAlmostEqual(a, actual[0], delta=delta)

        self._assert_vector_match(v, actual[1], delta=delta)


    def _assert_vector_match(self, a, b, delta=1E-7):

        self.assertEqual(len(a), len(b))

        lhs = np.dot(a, b)
        rhs = np.sqrt(np.sum(a**2)) + np.sqrt(np.sum(b**2))

        self.assertTrue((lhs - rhs) < delta)
        









if __name__ == '__main__':
    unittest.main()
        