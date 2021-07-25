import os
import torch
import pickle
import logging

from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from functools import partial


class FewShotWozDataset(Dataset):
    """
    self.intents (List[str]): Dialogue Act string (the part before '&').
    self.utterances (List[str]): Natural language utterances string (the part after '&').
    """
    def __init__(self):
        self.separator = ""
        self.intents = []
        self.utterances = []

    @staticmethod
    def from_bin_file(path):
        """ load from cached .bin"""
        with open(path, 'rb') as handle:
            few_shot_woz_dataset = pickle.load(handle)
        return few_shot_woz_dataset

    @staticmethod
    def from_txt_file(input_path, cache_path, separator='&'):
        """ load from .txt file """
        self = FewShotWozDataset()
        self.separator = separator

        with open(input_path, encoding="utf-8") as f:
            for line in f:
                str_split = line.lower().split(separator)
                code_str = str_split[0]
                utter_str = str_split[1]
                self.intents.append(code_str)
                self.utterances.append(utter_str)

        # save
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as handle:
            pickle.dump(self, handle)

        # return
        return self

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        return self.intents[idx], self.utterances[idx]


###########################
#    Collate Functions    #
###########################
def enc_dec_collate_fn(data, tokenizer, max_intent_len=40, max_utter_len=60):
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


def lm_collate_fn(data, tokenizer, max_len=80, separator='&'):
    """ Language Modeling model collate function
    Concatenate intents and utterances by separator.
    Return:
        inputs: padded_raw_ids
        labels: padded_raw_ids
    """
    raw_str = [f"{d[0]} {separator} {d[1]}" for d in data]
    padded_raw_ids = tokenizer(raw_str, padding='longest', truncation=True, max_length=max_len)['input_ids']
    return torch.LongTensor(padded_raw_ids), torch.LongTensor(padded_raw_ids)


###########################
#  Construct DataLoader   #
###########################
def get_dataset(data_file, data_cache_path=None, overwrite_cache=False):
    if not overwrite_cache and os.path.exists(data_cache_path):
        logging.info("Loading processed data from cached file %s", data_cache_path)
        dataset = FewShotWozDataset.from_bin_file(data_cache_path)
    else:
        logging.info("Creating features from dataset file at %s. Caching to %s.",
                     data_file, data_cache_path)
        dataset = FewShotWozDataset.from_txt_file(data_file, data_cache_path, '&')
    return dataset


def get_collate_fn(args, tokenizer):
    if args.enc_dec:
        fn = partial(enc_dec_collate_fn, tokenizer=tokenizer, max_intent_len=args.max_intent_len,
                     max_utter_len=args.max_utter_len)
    else:
        fn = partial(lm_collate_fn, tokenizer=tokenizer, max_len=args.max_len, separator='&')
    return fn


def get_sampler(dataset, mode):
    if mode == 'eval' or 'decode':
        return SequentialSampler(dataset)
    else:
        return RandomSampler(dataset)


def get_data_loader(args, tokenizer):
    if args.mode == "train":
        data_file = args.train_data_file
        batch_size = args.train_batch_size
        sampler_class = RandomSampler
    elif args.mode == "eval":
        data_file = args.eval_data_file
        batch_size = args.eval_batch_size
        sampler_class = SequentialSampler
    elif args.mode == "decode":
        data_file = args.decode_input_file
        batch_size = args.decode_batch_size
        sampler_class = SequentialSampler
    else:
        raise ValueError("Invalid args.mode")

    dataset = get_dataset(data_file, args.data_cache_path, args.overwrite_cache)
    sampler = sampler_class(dataset)
    collate_fn = get_collate_fn(args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, drop_last=False)
    return dataloader, len(dataset)


###########################
#   Main
if __name__ == "__main__":
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader

    train_file = "../data/restaurant/new.txt"
    cache_path = "../data_cached/restaurant/new.bin"

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    dataset = FewShotWozDataset.from_txt_file(train_file, cache_path, '&')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, sampler=None,
                            collate_fn=partial(enc_dec_collate_fn, tokenizer=t5_tokenizer,
                                               max_intent_len=40, max_utter_len=60))

    for batch in dataloader:
        print(len(batch))  # 2
        break
