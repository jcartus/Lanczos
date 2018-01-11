import numpy as np
import matplotlib.pyplot as plt

from qm import simulate_heisenberg_model

class Result(object):
    def __init__(self, L, E, m, m2, corr):
        self.L = L
        self.energy_density = E
        self.magnetisation = m
        self.magnetisation_squared = m2
        self.correlation = corr

class Experiment(object):
    def __init__(self, Jz, L_min=2, L_max=16, L_step=1):
        self.Jz = Jz
        self.lattice_sizes = list(range(L_min, L_max + 1, L_step))
        
        self.energy_densities = []
        self.magnetisations = []
        self.magnetisations_squared = []
        self.correlation = None

    def store_results(self, result):
        """Append results of a simulation to previous results """
        self.energy_densities.append(result.energy_density)
        self.magnetisations.append(result.magnetisation)
        self.magnetisations_squared.append(result.magnetisation_squared)

        if result.L == max(self.lattice_sizes):
            self.correlation = result.correlation

    def conduct(self):
        """Simulate Heisenberg model for given lattice sizes and jz"""
        msg = "Start experiment with Jz = " + str(self.Jz) + \
            " and L = [ " + ", ".join(map(str, self.lattice_sizes)) + " ]"
        InfoStream.message(msg, 2)
        for L in self.lattice_sizes:
            self.store_results(simulate_heisenberg_model(L, self.Jz))

    def display(self, wait=False):
        """Display results of experiment"""
        Printer.plot_all(self)
        if wait:
            plt.show()

class InfoStream(object):
    """Used to print properly formatted user messages """
    
    _prefixes = ["[-] ", "[+] ", "[#] "]

    suppress_level = -1    

    @classmethod
    def message(cls, text, level=0):
        """Prints a user message in practical format. Level 0 is an nice to have
        , 1 is a practical info, and 2 is an important info"""

        if level > cls.suppress_level:
            print(cls._prefixes[level] + text)



class Printer(object):

    @staticmethod
    def generate_subplot():
        return plt.subplots()

    @classmethod
    def plot_energy(cls, experiments, ax=None):
        if not isinstance(experiments, list):
            experiments = [experiments]

        if ax is None:
            _, ax = cls.generate_subplot()

        for ex in experiments:
            ax.plot(
                ex.lattice_sizes, 
                ex.energy_densities, 
                label="$J^z={0}$".format(ex.Jz)
            )
        
        ax.set_title("Energy densities")
        ax.set_xlabel("L")
        ax.set_ylabel("E / L")
        ax.legend()

    @classmethod
    def plot_magnetisations(cls, experiments, ax=None):
        if not isinstance(experiments, list):
            experiments = [experiments]

        if ax is None:
            _, ax = cls.generate_subplot()
        
        for ex in experiments:
            ax.plot(
                ex.lattice_sizes, 
                ex.magnetisations, 
                label="$J^z={0}$".format(ex.Jz)
            )
        
        ax.set_title("Magnetisations")
        ax.set_xlabel("L")
        ax.set_ylabel("$< \hat{M}_z>$")
        ax.legend()

    @classmethod
    def plot_magnetisations_squared(cls, experiments, ax=None):
        if not isinstance(experiments, list):
            experiments = [experiments]

        if ax is None:
            _, ax = cls.generate_subplot()
        
        for ex in experiments:
            ax.plot(
                ex.lattice_sizes, 
                ex.magnetisations_squared, 
                label="$J^z={0}$".format(ex.Jz)
            )
        
        ax.set_title("Squared Magnetisations")
        ax.set_xlabel("L")
        ax.set_ylabel("$<\hat{M}_z^2>$")
        ax.legend()
    

    @classmethod
    def plot_correlation(cls, experiments, ax=None):
        if not isinstance(experiments, list):
            experiments = [experiments]

        if ax is None:
            _, ax = cls.generate_subplot()
        
        for ex in experiments:
            ax.plot(
                list(range(len(ex.correlation))),
                ex.correlation, 
                label="$J^z={0}$".format(ex.Jz)
            )
        
        ax.set_title("Spin correlation")
        ax.set_xlabel("i")
        ax.set_ylabel("$<\hat{S}_0^z\hat{S}_0^i>$")
        ax.legend()   

    @classmethod
    def plot_all(cls, experiments):

        fig, axes = plt.subplots(2, 2)

        cls.plot_energy(experiments, axes[0, 0])
        cls.plot_magnetisations(experiments, axes[0, 1])
        cls.plot_magnetisations_squared(experiments, axes[1, 1])
        cls.plot_correlation(experiments, axes[1, 0])

        return fig

