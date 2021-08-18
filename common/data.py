import os
import torch
import pickle
import logging

from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from functools import partial


class AbstractDataset(Dataset):
    """
    self.intents (List[str]): Dialogue Act string (the part before '&').
    self.utterances (List[str]): Natural language utterances string (the part after '&').
    """
    def __init__(self, intents=[], utterances=[], separator=""):
        self.separator = separator
        self.intents = intents
        self.utterances = utterances

    @staticmethod
    def from_bin_file(path):
        """ load from cached .bin"""
        with open(path, 'rb') as handle:
            abstract_dataset = pickle.load(handle)
        return abstract_dataset

    @staticmethod
    def from_txt_file(input_path, cache_path, **kwargs):
        raise NotImplementedError("This method needs to be implemented.")

    @staticmethod
    def cache_file(dataset, input_path, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(input_path))[0]
        cache_path = os.path.join(cache_dir, filename + ".bin")
        with open(cache_path, 'wb') as handle:
            pickle.dump(dataset, handle)

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        return self.intents[idx], self.utterances[idx]


class FewShotWozDataset(AbstractDataset):
    """ FewShotWoz Dataset """
    @staticmethod
    def from_txt_file(input_path, cache_dir, separator='&', **kwargs):
        new_dataset = FewShotWozDataset()
        new_dataset.separator = separator

        with open(input_path, encoding="utf-8") as f:
            for line in f:
                str_split = line.lower().split(separator)
                code_str = str_split[0]
                utter_str = str_split[1]
                new_dataset.intents.append(code_str)
                new_dataset.utterances.append(utter_str)

        # cache
        FewShotWozDataset.cache_file(new_dataset, input_path, cache_dir)
        return new_dataset


class MultiwozSgdDataset(AbstractDataset):
    """ Multiwoz or SGD dataset """
    @staticmethod
    def from_txt_file(input_path, cache_dir, **kwargs):
        new_dataset = MultiwozSgdDataset()

        # process src file
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                new_dataset.intents.append(line.lower())

        # process tgt file
        with open(kwargs['tgt_file'], encoding="utf-8") as f:
            for line in f:
                new_dataset.utterances.append(line.lower())

        # cache
        MultiwozSgdDataset.cache_file(new_dataset, input_path, cache_dir)
        return new_dataset


# evaluation of predicted text outputs
class ComparisonDataset(Dataset):
    """ For evaluating outputs and targets
    self.intents: predicted utterances
    self.utterances:
    """
    def __init__(self, output_file, tgt_file):
        self.outputs = []
        self.tgts = []

        # process output file
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                self.outputs.append(line.lower())

        # process tgt file
        with open(tgt_file, encoding="utf-8") as f:
            for line in f:
                self.tgts.append(line.lower())

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return self.outputs[idx], self.tgts[idx]


###########################
#    Collate Functions    #
###########################
def enc_dec_collate_fn(data, tokenizer, max_intent_len, max_utter_len):
    """ Encoder-Decode model collate function
    Arg:
        data (List[Tuple[intent_str, utter_str]): One batch/list of data tuples. Each tuple contains un-tokenized
            intention string and utterance string.
    Return:
        token ids padded to the length of the longest sequence in the batch; truncated to max_xxx_len
        inputs: padded_intents_ids
        labels: padded_utterances_ids
    """
    intents, utterances = zip(*data)  # unzip data
    padded_intents_ids = tokenizer(intents, padding='longest', truncation=True,
                                   max_length=max_intent_len)['input_ids']
    padded_utterances_ids = tokenizer(utterances, padding='longest', truncation=True,
                                      max_length=max_utter_len)['input_ids']
    return torch.LongTensor(padded_intents_ids), torch.LongTensor(padded_utterances_ids)


###########################
#  Construct DataLoader   #
###########################
def get_dataset(data_file, tgt_file, data_cache_dir, overwrite_cache):
    filename = os.path.splitext(os.path.basename(data_file))[0]
    data_cache_path = os.path.join(data_cache_dir, filename + ".bin")
    if not overwrite_cache and os.path.exists(data_cache_path):
        logging.info("Loading processed data from cached dir %s", data_cache_path)
        dataset = MultiwozSgdDataset.from_bin_file(data_cache_path)
    else:
        logging.info("Creating features from dataset file at %s. Caching to %s.",
                     data_file, data_cache_dir)
        dataset = MultiwozSgdDataset.from_txt_file(data_file, data_cache_dir, tgt_file=tgt_file)
    return dataset


def get_collate_fn(args, tokenizer):
    fn = partial(enc_dec_collate_fn, tokenizer=tokenizer,
                 max_intent_len=args.max_intent_len, max_utter_len=args.max_utter_len)
    return fn


def get_data_loader(args, tokenizer, data_file, tgt_file, batch_size, sampler_class):
    dataset = get_dataset(data_file, tgt_file, args.data_cache_dir, args.overwrite_cache)
    sampler = sampler_class(dataset)
    collate_fn = get_collate_fn(args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, drop_last=False)
    return dataloader, len(dataset)


def get_data_loader_from_dataset(args, dataset, tokenizer, batch_size, sampler_class):
    sampler = sampler_class(dataset)
    collate_fn = get_collate_fn(args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, drop_last=False)
    return dataloader, len(dataset)


def get_comp_dataloader(output_file, tgt_file, batch_size):
    dataset = ComparisonDataset(output_file, tgt_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, len(dataset)


###########################
#   Main
if __name__ == "__main__":
    from transformers import T5Tokenizer

    train_file = "../data/restaurant/new.txt"
    cache_path = "../data_cached/restaurant/new.bin"

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    dataset = FewShotWozDataset.from_txt_file(train_file, cache_path, '&', )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, sampler=None,
                            collate_fn=partial(enc_dec_collate_fn, tokenizer=t5_tokenizer,
                                               max_intent_len=40, max_utter_len=60))

    for batch in dataloader:
        print(len(batch))  # 2
        break
