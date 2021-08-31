import json
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_output_dir = "saved_models"
dataset_name = "sgd"
domains = ['naive_5_shot']
# domains = ['attraction', 'hotel', 'laptop', 'restaurant', 'taxi', 'train', 'tv']
curriculums = ["NC", "one_pass", "baby_step", "dcl", "dcl.accu"]
figure_path = "visualize/results"


def read_history_json(path):
    """ keys = {batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen, eval_losses}"""
    with open(path, 'r') as fp:
        history = json.load(fp)
    return history


def get_hist_path(domain, curriculum):
    return f"{base_output_dir}/{dataset_name}/{domain}/{curriculum}/history.json"


def flatten_list(nested_list):
    flattened = [item for sublist in nested_list for item in sublist]
    return flattened


def plot_loss(ex_seen, losses, ax, label):
    ax.plot(np.cumsum(ex_seen), losses, label=label)


########
# Plot #
########
def plot_epoch_bleu(domain, saveto, curriculums):
    fig, ax = plt.subplots()

    for curriculum in curriculums:
        path = get_hist_path(domain, curriculum)
        history = read_history_json(path)
        ex_seen, losses = history['epoch_ex_seen'], history['neg_bleu']
        ex_seen, losses = flatten_list(ex_seen), flatten_list(losses)
        bleus = [-loss for loss in losses]
        plot_loss(ex_seen, bleus, ax, label=curriculum)

    plt.legend()
    ax.set_xlabel("num train examples seen")
    ax.set_ylabel("BLEU")
    ax.set_title(f"{domain.title()} Dev Set BLEU")
    ax.set_ylim([0, 70])
    plt.show()
    fig.savefig(f"{figure_path}/{saveto}")
    return fig


# def epoch_per_curriculum(domain, ax=None):
#     fig, ax = plt.subplots()
#
#     for curriculum in curriculums:
#         batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen = read_history_json(domain, curriculum)
#         if curriculum == 'NC':
#             n_epochs = [sum(epoch_ex_seen)]
#             cl_rank = [1]
#         elif curriculum in ['one_pass', 'baby_step']:  # flatten history for bucket curriculums
#             n_epochs = [sum(curr) for curr in epoch_ex_seen]
#             cl_rank = range(1, len(epoch_ex_seen) + 1)
#         else:
#             raise ValueError("Invalid curriculum name.")
#         ax.plot(cl_rank, n_epochs, 'o-', label=curriculum, markersize=7)
#
#     plt.legend()
#     ax.set_xlabel("curriculum")
#     ax.set_ylabel("num_examples")
#     ax.set_title(f"{domain.title()} Domain Number of Seen Examples per Curriculum")
#
#     fig.savefig(f"{figure_path}/{domain}.BucketCL.CLsize.png")
#     return fig


if __name__ == "__main__":
    plot_epoch_bleu("naive_10_shot", saveto="sgd.naive_10_shot_manual.png",
                    curriculums=["NC", "one_pass", "baby_step", "dcl"])
    plot_epoch_bleu("naive_10_shot", saveto="sgd.naive_10_shot_dcl.png",
                    curriculums=["dcl", "dcl.accu_0.25", "dcl.accu_0.5", "dcl.accu_0.75"])

    # epoch_per_curriculum("restaurant", ax=None)
