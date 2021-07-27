"""Curriculum Learning Sampler"""
import logging

from torch.utils.data import DataLoader, RandomSampler


class BucketCurriculum(object):
    def __init__(self, dataset, scoring_fn):
        """
        dataset (FewShotWozDataset): Dataset that returns self.intents and self.utterances
        scoring_fn (Callable):
            input: intents(Str), utterance(Str)
            output: scores (Float/Int) that measure how hard the example is
        """
        self.dataset = dataset
        self.intents = dataset.intents
        self.utterances = dataset.utterances
        self.difficulty = [scoring_fn(i, u) for i, u in zip(self.intents, self.utterances)]

        # sort by difficulty score
        sort_by_diff = sorted(zip(self.intents, self.utterances, self.difficulty), key=lambda pair: pair[2])
        self.intents, self.utterances, self.difficulty = zip(*sort_by_diff)

    def get_curriculum(self, bucket_size, batch_size, name, collate_fn):
        """
        num_bucket (Int): number of buckets / curriculum to split the dataset into
        batch_size (Int): train batch_size for each bucket
        name (Str): [one_pass, baby_step]
        Yield:
             same dataset class as inputted to the constructor function
        """
        dataset_class = self.dataset.__class__
        dataset_len = len(self.dataset)
        dataset_separator = self.dataset.separator

        num_buckets = dataset_len // bucket_size + 1
        logging.info(f"Number of instances={len(self.dataset)}")
        logging.info(f"Total Number of curriculums={num_buckets}")
        logging.info(f"**************************************")

        for b in range(num_buckets):
            if name == "one_pass":
                start, end = b * bucket_size, min(b * bucket_size + bucket_size, dataset_len)
            elif name == "baby_step":
                start, end = 0, min(b * bucket_size + bucket_size, dataset_len)
            else:
                raise ValueError("Invalid BucketCurriculum name. Must be in [one_pass, baby_step].")
            curriculum_dataset = dataset_class(self.intents[start: end], self.utterances[start: end], dataset_separator)
            curriculum_dataloader = DataLoader(curriculum_dataset, batch_size=batch_size,
                                               sampler=RandomSampler(curriculum_dataset),
                                               collate_fn=collate_fn, drop_last=False)
            yield curriculum_dataloader, len(curriculum_dataset)


###########################
#    Scoring Function     #
###########################
def length_score_fn(intents, utterances):
    return len(intents.split()) + len(utterances.split())


###########################
#   Main
if __name__ == "__main__":
    from transformers import T5Tokenizer
    from common.data import FewShotWozDataset

    train_file = "../data/restaurant/new.txt"
    cache_path = "../data_cached/restaurant/new.bin"

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    dataset = FewShotWozDataset.from_txt_file(train_file, cache_path, '&')
    curriculum = BucketCurriculum(dataset, length_score_fn)

    for bucket in curriculum.get_curriculum(bucket_size=4, name="one_pass"):
        pass
