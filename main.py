from utilities import Experiment, Printer
from utilities import InfoStream as info
import matplotlib.pyplot as plt
import numpy as np

def main():

    info.message("Welcome!", 2)

    Jz = [0, 1, 2]
    L_min = 2
    L_max = 10#16
    L_step = 1

    export = False

    file_name = "results"

    seed = 0
    if seed:
        info.message("Setting seed to " + str(seed), 1)
        np.random.seed(13)

    experiments = []
    for jz in Jz:
        experiment = Experiment(jz, L_min=L_min, L_max=L_max, L_step=L_step)
        experiment.conduct()
        experiments.append(experiment)
        #experiment.display()

    info.message("Calculation finished. Start plotting ...", 2)

    if export:
        Printer.export_all(experiments, file_name + ".txt", file_name + ".png")
    else:
        Printer.plot_all(experiments)

    plt.show()

    info.message("All done. Goodbye ...", 2)

    

if __name__ == '__main__':
    main()

