import numpy as np

class HeisenbergSector(object):

    def __init__(self, number_of_sites, number_spinups, jz):

        self.number_of_sites = number_of_sites
        self.number_spinups = number_spinups
        self.basis = []

    def setup_basis(self):
        """Generates a list of state object (basis for this sector)"""
        basis = []



class SpinState(object):
    
    def __init__(self, state, msb=None):
        """state .. either array of bits or int (in which case msb should be 
        specifed)"""

        if msb is None:
            self.bit_seq = state
            self.msb = len(state) - 1
        else:
            self.bit_seq = self._decimal_to_binary(state, msb)
        

    def _binary_to_decimal(self, bit_array):
        # get non zero exponents
        exponents = np.arange(len(bit_array))[bit_array > 0 ]
        
        # sum up and return
        return np.sum(2**exponents)

    def _decimal_to_binary(self, decimal, msb):
        if decimal >= 2**(msb+1):
            raise ValueError("Decimal value is out of range!")

        rest, i, bit_seq = decimal, msb, np.zeros(msb+1)
        while(rest > 0):
            if 2**i <= rest:
                bit_seq[i] = 1
                rest -= 2**i
            i -= 1
        
        return bit_seq

    def magnetisation(self):
        # {0,1} -> {-1/2,1/2}
        sz = self.bit_seq - 0.5
        
        # sz_i * (-1)^-1
        # von 1 bis end. end = msb, wenn msb ungerade; end = msb -1, msb gerade
        ind = np.arange(1, self.msb - int(self.msb % 2 == 0) + 1, 2)
        sz[ind] *= -1
        return np.sum(sz) / (self.msb + 1)

    def magnetisation_sqared(self):
        # {0,1} -> {-1/2,1/2}
        sz = self.bit_seq - 0.5

        # matrix for (-1)^(i+j)
        sign = np.ones((self.msb+1, self.msb+1))
        sign[1::2,::2] *= -1
        sign[::2, 1::2] *= -1

        return np.dot(sz, np.dot(sign, sz)) / (self.msb + 1)**2


