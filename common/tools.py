import os
import random


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def text_sampler(n, file_list, save_to):
    """ randomly sample n samples and save as new txt file.
    file_list (List(Str)): list of file names to read from.
    save_to (List(Str): list of file names to save to.
    """
    f_len = file_len(file_list[0])
    indices = sorted(random.sample(range(f_len), n))  # randomly sample n lines

    # delete existing output files
    for file in save_to:
        if os.path.isfile(file):
            os.remove(file)

    for f_idx, file in enumerate(file_list):
        j = 0
        with open(file, 'r') as in_fp, open(save_to[f_idx], 'a') as out_fp:
            for i, line in enumerate(in_fp):
                try:
                    line_idx = indices[j]
                except IndexError:
                    break
                if i < line_idx:
                    pass
                elif i == line_idx:
                    out_fp.write(line)
                    j += 1


if __name__ == "__main__":
    folder = "data/sgd"
    file_names = ["dev.src", "dev.trg", "dev.vals"]
    out_names = ["dev2.src", "dev2.trg", "dev2.vals"]

    file_list = [f"{folder}/{f}" for f in file_names]
    save_to = [f"{folder}/{f}" for f in out_names]
    text_sampler(1000, file_list, save_to)






##### Train
# 1. shrink dataset while retaining the most data as possible.

##### Visualization
# How curriculum helps:
# 3. SPL
#     - self defined difficult measure (e.x. # epochs not learning from a specific example)

##### Decode

#### Eval
# - slot accuracy: need the mapping of slot <--> whether binary

# Other metrics:
# TODO: SER https://github.com/google-research/schema-guided-dialogue/tree/main/generation


"""
Technical Notes:
1. For Bucket Curriculum, the last bucket is incomplete to make sure all data are used.
   The other buckets are of exactly the same size.
"""