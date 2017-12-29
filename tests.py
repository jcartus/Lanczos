import unittest

import numpy as np

from qm import SpinState

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

        #self.assertEqual(
        #    - 0.25 + 0.25 - 0.25 + 0.25 + 0.25,
        #    self._test_state.energy
        #)

if __name__ == '__main__':
    unittest.main()
        