import os
import json
import torch
import logging
from tqdm import tqdm

from nltk.translate.meteor_score import meteor_score
import sacrebleu
import bleurt.score
from common.utils import load_checkpoint
from common.data import get_data_loader, get_comp_dataloader


def evaluate_loss(eval_dataloader, len_eval_dataset, model, args, verbose=False):
    """ Get model loss """
    # Eval!
    if verbose:
        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len_eval_dataset)
        logging.info("  Batch size = %d", args.eval_batch_size)

    model.to(args.device)
    model.eval()
    eval_loss = 0.0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item() * len(inputs)

    eval_loss = eval_loss / len_eval_dataset
    return eval_loss


def evaluate_bleu(eval_dataloader, len_eval_dataset, model, tokenizer, args):
    model.to(args.device)
    model.eval()
    bleu_scores = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating BLEU"):
        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        example_ids = model.generate(inputs, max_length=args.max_utter_len)
        examples = tokenizer.batch_decode(example_ids, skip_special_tokens=True)
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for example, target in zip(examples, targets):
            bleu_scores += sacrebleu.sentence_bleu(example, [target]).score

    return bleu_scores / len_eval_dataset


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
