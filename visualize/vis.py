import json
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

domains = ['attraction', 'hotel', 'laptop', 'restaurant', 'taxi', 'train', 'tv']
curriculums = ["NC", "one_pass", "baby_step"]
base_output_dir = "saved_models/t5-small"
figure_path = "visualize/results"


def read_history_json(domain, curriculum):
    output_dir = f"{base_output_dir}/{curriculum}/{domain}"
    with open(f"{output_dir}/history.json", 'r') as fp:
        history = json.load(fp)

    batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen = \
        history["batch_losses"], history["batch_ex_seen"], history["epoch_losses"], history["epoch_ex_seen"]
    return batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen


def plot_epoch_losses(domain, ax=None):
    fig, ax = plt.subplots()

    for curriculum in curriculums:
        batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen = read_history_json(domain, curriculum)
        if curriculum == 'NC':
            ex_seen = epoch_ex_seen
            losses = epoch_losses
        elif curriculum in ['one_pass', 'baby_step']:  # flatten history for bucket curriculums
            ex_seen = [item for sublist in epoch_ex_seen for item in sublist]
            losses = [item for sublist in epoch_losses for item in sublist]
        else:
            raise ValueError("Invalid curriculum name.")
        ax.plot(np.cumsum(ex_seen), losses, label=curriculum)

    plt.legend()
    ax.set_xlabel("num examples")
    ax.set_ylabel("per epoch loss")
    ax.set_title(f"{domain.title()} Domain Training Losses")

    fig.savefig(f"{figure_path}/{domain}.BucketCL.epoch.png")
    return fig


def epoch_per_curriculum(domain, ax=None):
    fig, ax = plt.subplots()

    for curriculum in curriculums:
        batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen = read_history_json(domain, curriculum)
        if curriculum == 'NC':
            n_epochs = [sum(epoch_ex_seen)]
            cl_rank = [1]
        elif curriculum in ['one_pass', 'baby_step']:  # flatten history for bucket curriculums
            n_epochs = [sum(curr) for curr in epoch_ex_seen]
            cl_rank = range(1, len(epoch_ex_seen) + 1)
        else:
            raise ValueError("Invalid curriculum name.")
        ax.plot(cl_rank, n_epochs, 'o-', label=curriculum, markersize=7)

    plt.legend()
    ax.set_xlabel("curriculum")
    ax.set_ylabel("num_examples")
    ax.set_title(f"{domain.title()} Domain Number of Seen Examples per Curriculum")

    fig.savefig(f"{figure_path}/{domain}.BucketCL.CLsize.png")
    return fig


if __name__ == "__main__":
    fig = plot_epoch_losses("restaurant")
    # plt.show()
    epoch_per_curriculum("restaurant", ax=None)
    plt.show()
