import os
import glob
import torch
import logging

from tqdm import tqdm
from transformers import WEIGHTS_NAME

from common.utils import load_checkpoint
from common.data import get_data_loader


def get_checkpoint_list(args):
    """ get list of checkpoints """
    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in sorted(
            glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    else:
        checkpoints = [args.output_dir]
    return checkpoints


def eval_checkpoint(args, model, tokenizer, prefix):
    # data
    eval_dataloader, len_eval_dataset = get_data_loader(args, tokenizer)

    # Eval!
    logging.info("***** Running evaluation {} *****".format(prefix))
    logging.info("  Num examples = %d", len_eval_dataset)
    logging.info("  Batch size = %d", args.eval_batch_size)

    model.to(args.device)
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    # save
    output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
    os.makedirs(os.path.dirname(output_eval_file), exist_ok=True)
    with open(output_eval_file, "w") as writer:
        logging.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def evaluate(args, model_class, tokenizer_class):
    checkpoints = get_checkpoint_list(args)
    logging.info("Evaluate the following checkpoints: %s", checkpoints)

    results = {}
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if args.eval_all_checkpoints else ""
        prefix = checkpoint.split('/')[-1] if 'checkpoint' in checkpoint else ""

        logging.info("Evaluate checkpoint {}".format(checkpoint))
        model, tokenizer = load_checkpoint(checkpoint, model_class, tokenizer_class)
        result = eval_checkpoint(args, model, tokenizer, prefix)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

    return results
