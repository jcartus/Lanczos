from utilities import Experiment, Printer
from utilities import InfoStream as info
import matplotlib.pyplot as plt


def main():

    info.message("Welcome!", 2)

    Jz = [0, 1, 2]
    L_min = 2
    L_max = 16
    L_step = 1

    experiments = []
    for jz in Jz:
        experiment = Experiment(jz, L_min=L_min, L_max=L_max, L_step=L_step)
        experiment.conduct()
        experiments.append(experiment)
        #experiment.display()

    info.message("Calculation finished. Start plotting ...", 2)

    Printer.plot_all(experiments)

    plt.show()

    info.message("All done. Goodbye ...", 2)

    

if __name__ == '__main__':
    main()

