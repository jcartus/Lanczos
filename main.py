from utils import Experiment, Printer

def main():
    Jz = [0, 1, 2]
    L_min = 2
    L_max = 16
    L_step = 1

    experiments = []
    for jz in Jz:
        experiment = Experiment(jz, L_min=2, L_max=16, L_step=1)
        experiment.conduct()
        experiments.append(experiment)
        #experiment.display()

    Printer.plot_all(experiments)

if __name__ == '__main__':
    main()

