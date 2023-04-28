import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", type=str, default=None, help="A directory that contains results."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    epochs, mpjpe_baseline = [], []
    with open(args.result_dir + "/baseline/train.log", "r") as log_file:
        for line in log_file.readlines():
            log_line = [col.strip() for col in line[20:].split(", ") if col]
            epochs.append(float(log_line[0].split(": ")[1]))
            mpjpe_baseline.append(float(log_line[3].split(": ")[1]))
    
    our_dir = next(os.walk(args.result_dir + '/ours/.'))[1]
    mpjpes = []
    if len(our_dir) > 0:
        for dir in our_dir:
            mpjpe = []
            with open(args.result_dir + '/ours/' + dir + "/train.log", "r") as log_file:
                for line in log_file.readlines():
                    log_line = [col.strip() for col in line[20:].split(", ") if col]
                    mpjpe.append(float(log_line[3].split(": ")[1]))
            mpjpes.append(mpjpe)

    plt.figure()
    TITLE_SIZE = 25
    FONT_SIZE = 15
    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=TITLE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title
    plt.title("Mean Per Joint Position Error (MPJPE)")
    plt.xlabel("n epochs")
    plt.ylabel("MPJPE")
    plt.xticks(np.arange(0, 15, step=1))
    plt.yticks(np.arange(45, 55, step=0.5))


    plt.plot(epochs, mpjpe_baseline, 's-', color = 'r', label="baseline")
    for mpjpe, model in zip(mpjpes, our_dir):
        plt.plot(epochs, mpjpe, 'o-', label=model)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()