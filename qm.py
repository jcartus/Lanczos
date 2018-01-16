import numpy as np
import math
import copy
import utilities


class HeisenbergSector(object):

    def __init__(self, number_of_sites, number_spinups, jz):

        self.number_of_sites = number_of_sites
        self.number_spinups = number_spinups
        self.Jz = jz
        self.basis = []
        self.H = None

    def setup_basis(self):
        """Generates a list of state object (basis for this sector)"""

        #--- some auxillary functions ---
        def basis_recursion(pos, basis):
            bits = copy.deepcopy(basis[-1].bit_seq)

            # shift highest to the right
            bits[pos] = 1
            bits[pos - 1] = 0

            bits = shift_right(pos, bits)

            # lowest state in this recursion
            basis.append(SpinState(bits))

            # continue recursion if limit not reached
            if pos > 1:

                # find next spin up
                diff = 0 #steps till next spin up
                while bits[pos - 1] == 0:
                    pos -= 1
                    diff += 1
                    if pos < 1:
                        return basis
                
                # create next recursion level
                for i in range(diff):
                    basis = basis_recursion(pos + i, basis)
            
            self.basis = basis
            return basis

        def shift_right(pos, bits):
            """Shifts all up bits to the rhs of pos as far the right as possible
            e.g. 110101100, pos=6 -> 110000111"""
            count = int(np.sum(bits[0:pos-1]))
            bits[0:pos-1] = 0
            bits[:count] = 1
            return bits
        
        #---

        basis = []

        # if all spins are up only one state possible
        if self.number_spinups == self.number_of_sites:
            basis.append(SpinState(np.ones(self.number_spinups)))
            self.basis = basis
            return basis

        pos = self.number_spinups

        # 1. state
        basis.append(
            SpinState(2**self.number_spinups -1, self.number_of_sites - 1)
        )

        while pos < self.number_of_sites:       
            basis = basis_recursion(pos, basis)
            pos += 1
        

        return basis

    def setup_hamiltonian(self):
        """Creates the hamiltonian matrix in the spin basis"""
        
        if self.basis == []:
            self.setup_basis()


        #--- prerequisites ---
        basis_in_decimal = [x.decimal for x in self.basis]

        dim = len(self.basis)
        H = np.zeros((dim, dim)) #todo sparse matrix!
        #---

        for i, state in enumerate(self.basis):
            # diagonal
            H[i, i] = state.energy(self.Jz)
            
            #--- off-diagonal (are all 1/2 because J^\top = 1) ---
            flipped = [x.decimal for x in state.generate_flipped_states()]

            # bisection
            j = [basis_in_decimal.index(f) for f in flipped]            

            # again all elements are 1/2, expect for N=2
            H[i, j] = 0.5 + int(self.number_of_sites == 2) * 0.5
            #---

        self.H = H
        return H

    @staticmethod
    def lanczos_diagonalisation(
            H, 
            n_max=None, 
            n_diag=None, 
            delta_E=1E-10, 
            delta_k=1E-25
        ):
        
        L = len(H)

        if n_max is None:
            n_max = 2 * L**2 #todo oder doch nur L?

        if n_diag is None:
            n_diag = np.ceil(L / 10.0)

        #--- init ---
        n = 1
        converged = False
        k, e = [], []

        x = np.random.rand(L)
        x_start = x # needed for coefficient recovery
        x_old = np.zeros(L)
        E_old = 1E10

        def norm(vector):
            return np.sqrt(np.sum(x**2))
        #---

        while not converged:
            k.append(norm(x))

            if k[-1] < delta_k:
                converged = True
                n -= 1
                utilities.InfoStream.message(
                    "Convergence reached after {0} iterations: k < delta_k".format(n)
                )
                break

            x = x / k[-1]
            e.append(np.dot(x, np.dot(H, x)))

            if n % n_diag == 0:
                # diagonalize in krylov space
                H_t = np.diag(e) + np.diag(k[1:], +1) + np.diag(k[1:], -1)
                E = np.linalg.eigvalsh(H_t)[0]

                if np.abs(E - E_old) < delta_E:
                    converged = True
                    utilities.InfoStream.message(
                        "Convergence reached after {0} iterations: dE < delta_E".format(n)
                    )
                    break
                else:
                    E_old = E

            x_new = np.dot(H, x) - e[-1] * x - k[-1] * x_old
            x_old = x
            x = x_new

            n += 1

            if n > n_max:
                n -= 1
                converged = True
                utilities.InfoStream.message(
                    "Convergence reached after {0} iterations: n_max iterations exceeded".format(n))

        #--- calculate coefficients in spin basis ---
        c = np.zeros(L)
        c_krylov = np.linalg.eigh(H_t)[1][:, 0]

        x = x_start
        x_old = 0

        for i in range(len(c_krylov)):
            x  = x / k[i]
            c += c_krylov[i] * x

            x_new = np.dot(H, x) - e[i] * x - k[i] * x_old
            x_old = x
            x = x_new
        #---

        return E, c, n
        
    def calculate_ground_state(self, give_iterations=False):
        self.setup_basis()
        self.setup_hamiltonian()
        E, c, n = self.lanczos_diagonalisation(self.H)

        groundstate = MixedState(c, self.basis)

        if give_iterations:
            return E, groundstate, give_iterations
        else:
            return E, groundstate


class SpinState(object):
    
    def __init__(self, state, msb=None):
        """state .. either array of bits or int (in which case msb should be 
        specifed)"""

        if msb is None:
            self.bit_seq = state
            self.decimal = self._binary_to_decimal(state)
            self.msb = len(state) - 1
        else:
            self.bit_seq = self._decimal_to_binary(state, msb)
            self.decimal = state
            self.msb = msb

    
    def generate_flipped_states(self):
        #--- find flipable sites ---
        # get spins of nearest neighbour
        bit_seq_nearest_neighbour = np.roll(self.bit_seq, -1)

        # todo check this again
        is_flipable = self.bit_seq != bit_seq_nearest_neighbour
        #---

        flip_states = []

        # create new state for every flipable site 
        for ind in np.arange(self.msb + 1)[is_flipable]:
            flipped_seq = copy.deepcopy(self.bit_seq)
            
            # swap spin valies at i and i+1 (equivalent to spin flip)
            flipped_seq[ind], flipped_seq[(ind + 1) % (self.msb + 1)] = \
                flipped_seq[(ind + 1) % (self.msb + 1)], flipped_seq[ind]

            flip_states.append(SpinState(flipped_seq))
        
        return flip_states
        

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

    def energy(self, jz):
        sz = self.bit_seq - 0.5
        return np.sum(sz * np.roll(sz, -1)) * jz

    def magnetisation(self):
        # {0,1} -> {-1/2,1/2}
        sz = self.bit_seq - 0.5
        
        # sz_i * (-1)^-1
        # von 1 bis end. end = msb, wenn msb ungerade; end = msb -1, msb gerade
        ind = np.arange(1, self.msb - int(self.msb % 2 == 0) + 1, 2)
        sz[ind] *= -1
        return np.sum(sz) / (self.msb + 1)

    def magnetisation_squared(self):
        # {0,1} -> {-1/2,1/2}
        sz = self.bit_seq - 0.5

        # matrix for (-1)^(i+j)
        sign = np.ones((self.msb+1, self.msb+1))
        sign[1::2,::2] *= -1
        sign[::2, 1::2] *= -1

        return np.dot(sz, np.dot(sign, sz)) / (self.msb + 1)**2

    def correlation(self):
        sz = self.bit_seq - 0.5
        return sz[0] * sz

class MixedState(object):

    def __init__(self, coeffs, basis):
        self._basis = basis
        self._coefficients = coeffs


    def magnetisation(self):
        """m = sum_i |c_i|^2 * m_i"""
        M = np.array([x.magnetisation() for x in self._basis])
        return np.dot(self._coefficients**2, M) / np.sum(self._coefficients**2)

    def magnetisation_squared(self):
        """m2 = sum_i |c_i|^2 * m2_i"""
        M2 = np.array([x.magnetisation_squared() for x in self._basis])
        return np.dot(self._coefficients**2, M2) / np.sum(self._coefficients**2)

    def correlation(self):
        """corr_i = sum_n (S_0^zS_i^z)_n * |c_n|^2"""
        correlations = np.array([x.correlation() for x in self._basis])
        return np.dot(
            self._coefficients**2, correlations
        ) / np.sum(self._coefficients**2)

def simulate_heisenberg_model(L, jz):
    """Calculates a few properties of the 1-D Heisenberg Modell
    
    Args:
        L: number of lattice sites
        jz: the value of J^z

    Returns:
        Result object (stores energy density, magnetisation, 
        magnetisation squared and the autocorrelation)
    """

    msg = "Simulating System with Jz = {0} and L = {1}".format(jz, L)
    utilities.InfoStream.message(msg, 1)


    #--- determine sectors to search for ground state in ---
    # number of spin ups in sector with lowest S_tot^z
    N_min = math.ceil(L / 2)

    # if jz != lieb-mattis theorem can be used
    if jz == 0:
        N = list(range(N_min, L+1))
    else:
        N = [N_min]
    #---

    #--- scan through sector(s) ---
    E_min = 1E7
    lowest_ground_state = None
    for n in N:
        utilities.InfoStream.message("Analyzing sector n_up = {0}".format(n))
        sector = HeisenbergSector(L, n, jz)
        E, ground_state = sector.calculate_ground_state()
        if E < E_min:
            E_min = E
            lowest_ground_state = ground_state
    #---

    result = utilities.Result(
        L, 
        E_min, 
        lowest_ground_state.magnetisation(),
        lowest_ground_state.magnetisation_squared(),
        lowest_ground_state.correlation()
    )

    return result

