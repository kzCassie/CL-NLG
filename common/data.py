import os
import pickle
from torch.utils.data import Dataset


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
    return padded_intents_ids, padded_utterances_ids


def lm_collate_fn(data, tokenizer, max_len=80, separator='&'):
    """ Language Modeling model collate function
    Concatenate intents and utterances by separator.
    Return:
        inputs: padded_raw_ids
        labels: padded_raw_ids
    """
    raw_str = [f"{d[0]} {separator} {d[1]}" for d in data]
    padded_raw_ids = tokenizer(raw_str, padding='longest', truncation=True, max_length=max_len)['input_ids']
    return padded_raw_ids, padded_raw_ids


###########################
#   Main
if __name__ == "__main__":
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader
    from functools import partial

    train_file = "../data/restaurant/new.txt"
    cache_file = "../data_cached/restaurant/new.bin"

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    dataset = FewShotWozDataset().from_txt_file(train_file, cache_file, '&')
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, sampler=None,
                            collate_fn=partial(enc_dec_collate_fn, tokenizer=t5_tokenizer,
                                               max_intent_len=40, max_utter_len=60))

    for batch in dataloader:
        print(len(batch))  # 2
        break
