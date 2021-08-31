import os
import re
import json
import torch
import logging
from tqdm import tqdm
from collections import defaultdict

from torch.nn import CrossEntropyLoss
from nltk.translate.meteor_score import meteor_score
import sacrebleu
import bleurt.score
from common.utils import load_checkpoint
from common.data import get_data_loader, get_comp_dataloader
BINARY_ANS = ['none', 'yes', 'no', 'false', 'true']


#################
# slot accuracy #
#################
def parse_intent(src, sv_only=False):
    """ parse sgd src string
    args:
        src (str): Source intention string (sgd dataset format)
        sv_only: only return list of slot-value pairs
    return:
        sv_only=False: (dict)
            domain: str
            intent: defaultdict
                intent_name: dict. key=slot, val=value
        sv_only=True: (list(tuple))
            [(slot, val), (slot, val), ...]
    """
    p_sv = re.compile(".*\((.*)\)")
    intents = src.split('|')
    res = [] if sv_only else {"domain": intents[0].strip(), "intent": defaultdict(dict)}

    for intent_str in intents[1:]:
        tmp1 = intent_str.split('(')
        intent = tmp1[0].strip()

        try:
            matched_sv = re.match(p_sv, intent_str).group(1).split('=')
            matched_sv = [token.strip() for token in matched_sv]
        except AttributeError:
            matched_sv = []

        if len(matched_sv) == 0:
            if not sv_only:
                res["intent"][intent] = {}
            continue

        if len(matched_sv) == 1:
            slot, value = matched_sv[0], "none"
        else:
            slot, value = matched_sv

        if sv_only:
            res.append((slot, value))
        else:
            res["intent"][intent][slot] = value

    return res


def get_non_bin_sv(slot_val_list):
    """ get none-binary slots """
    non_bin_sv = []
    for s, v in slot_val_list:
        if v not in BINARY_ANS:
            non_bin_sv.append((s, v))
    return non_bin_sv


def calc_slot_accu(src, tgt):
    slot_val_list = parse_intent(src, sv_only=True)
    non_bin_sv = get_non_bin_sv(slot_val_list)
    correct = 0
    for s, v in non_bin_sv:
        if v in tgt:
            correct += 1
    accu = 1 if len(non_bin_sv) == 0 else correct/len(non_bin_sv)
    return accu


# def evaluate_loss(eval_dataloader, len_eval_dataset, model, args, sentence_loss=False):
#     model.to(args.device)
#     model.eval()
#     eval_losses = []
#
#     loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
#     for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         inputs, labels = batch
#         inputs = inputs.to(args.device)
#         labels = labels.to(args.device)
#
#         with torch.no_grad():
#             outputs = model(inputs, labels=labels)
#             tmp_loss = loss_fct(outputs.logits.permute(1, 2, 0), labels.permute(1, 0))
#             sample_losses = torch.mean(tmp_loss, dim=0)
#             eval_losses.extend(sample_losses.tolist())
#             # loss = outputs.loss
#             # eval_losses.append(loss.item() * len(inputs))
#
#     if sentence_loss:
#         return eval_losses
#     else:
#         return sum(eval_losses) / len_eval_dataset


# def evaluate_bleu(eval_dataloader, len_eval_dataset, model, tokenizer, args, sentence_bleu=False):
#     model.to(args.device)
#     model.eval()
#     bleu_scores = []
#
#     for batch in tqdm(eval_dataloader, desc="Evaluating BLEU"):
#         inputs, labels = batch
#         inputs = inputs.to(args.device)
#         labels = labels.to(args.device)
#
#         example_ids = model.generate(inputs, max_length=args.max_utter_len)
#         examples = tokenizer.batch_decode(example_ids, skip_special_tokens=True)
#         targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#         for example, target in zip(examples, targets):
#             bleu_scores.append(sacrebleu.sentence_bleu(example, [target]).score)
#
#     if sentence_bleu:
#         return bleu_scores
#     else:
#         return sum(bleu_scores) / len_eval_dataset


def evaluate_data_set(eval_dataloader, model, tokenizer, metrics, args, sentence_level=False):
    """
    metrics = ['loss', 'bleu', 'accu']
    """
    model.to(args.device)
    model.eval()
    eval_losses, bleu_scores, bleurt_scores, slot_accuracies = [], [], [], []

    if 'loss' in metrics:
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    if 'bleurt' in metrics:
        bleurt_scorer = bleurt.score.BleurtScorer()

    for batch in tqdm(eval_dataloader, desc="Evaluating Metrics"):
        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            if 'loss' in metrics:
                outputs = model(inputs, labels=labels)
                tmp_loss = loss_fct(outputs.logits.permute(1, 2, 0), labels.permute(1, 0))
                sample_losses = torch.mean(tmp_loss, dim=0)
                eval_losses.extend(sample_losses.tolist())

            if 'bleu' in metrics or 'accu' in metrics or 'bleurt' in metrics:
                example_ids = model.generate(inputs, max_length=args.max_utter_len)
                examples = tokenizer.batch_decode(example_ids, skip_special_tokens=True)

                if 'bleu' in metrics or 'bleurt' in metrics:
                    targets = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    for example, target in zip(examples, targets):
                        if 'bleu' in metrics:
                            bleu_scores.append(sacrebleu.sentence_bleu(example, [target]).score)
                        if 'bleurt' in metrics:
                            bleurt_scores.append(bleurt_scorer.score(references=[target], candidates=[example])[0])
                if 'accu' in metrics:
                    sources = tokenizer.batch_decode(inputs, skip_special_tokens=True)
                    for source, example in zip(sources, examples):
                        slot_accuracies.append(calc_slot_accu(source, example))

    if sentence_level:
        res = {'loss': eval_losses, "bleu": bleu_scores, "bleurt": bleurt_scores, 'accu': slot_accuracies}
    else:
        res = {}
        if "loss" in metrics:
            res['loss'] = sum(eval_losses) / len(eval_losses)
        if "bleu" in metrics:
            res['bleu'] = sum(bleu_scores) / len(bleu_scores)
        if "bleurt" in metrics:
            res['bleurt'] = sum(bleurt_scores) / len(bleurt_scores)
        if "accu" in metrics:
            res["accu"] = sum(slot_accuracies) / len(slot_accuracies)
    return res


def save_result(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(result, open(path, 'w'), indent=2)


##############################
# Main func called in exp.py #
##############################
def evaluate_output(args, output_file, tgt_file, batch_size):
    """ Compare predicted output and tgt output """
    dataloader, len_dataset = get_comp_dataloader(output_file, tgt_file, batch_size)
    bleu_scores, meteor_scores, bleurt_scores = [], [], []

    bleurt_scorer = bleurt.score.BleurtScorer()
    for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
        output, target = batch
        output = output[0]
        target = target[0]

        bleu_scores.append(sacrebleu.sentence_bleu(output, [target]).score)
        meteor_scores.append(meteor_score([output], target))
        bleurt_scores.append(bleurt_scorer.score(references=[target], candidates=[output])[0])

    # Avg Evaluation
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)

    # result
    results = {
        "Num examples": len_dataset,
        "BLEU": avg_bleu,
        "METEOR": avg_meteor,
        "BLEURT": avg_bleurt
    }

    # save
    path = f"{args.output_dir}/eval_results.json"
    json.dump(results, open(path, 'w'), indent=2)
    logging.info("Evaluation results saved to {}".format(path))
    return results


if __name__ == "__main__":
    src_path = "data/sgd/naive_5_shot/dev2.src"
    tgt_path = "data/sgd/naive_5_shot/dev2.trg"

    with open(src_path, 'r') as fp, open(tgt_path, 'r') as fp2:
        i = 0
        for src, tgt in zip(fp, fp2):
            print(calc_slot_accu(src, tgt))
            i += 1
            if i>50: break


