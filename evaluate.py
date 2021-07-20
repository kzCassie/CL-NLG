import os
import glob
import torch
import logging

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME

from common.utils import load_checkpoint


def get_checkpoint_list(args):
    """ get list of checkpoints """
    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in sorted(
            glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        # logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    else:
        checkpoints = [args.output_dir]
    return checkpoints


def eval_checkpoint(args, model, tokenizer):
    model.to(args.device)


    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, enc_dec=args.enc_dec)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)

        inputs, masks, labels = batch
        # import pdb
        # pdb.set_trace()
        inputs = inputs.to(args.device)
        # masks = masks.to(args.device)
        labels = labels.to(args.device)
        # inputs = inputs.to(args.device)
        # labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result



def evaluate(args, model_class, tokenizer_class):
    checkpoints = get_checkpoint_list(args)
    logging.info("Evaluate the following checkpoints: %s", checkpoints)

    results = {}
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if args.eval_all_checkpoints else ""
        # prefix = checkpoint.split('/')[-1] if 'checkpoint' in checkpoint else ""

        model, tokenizer = load_checkpoint(checkpoint, model_class, tokenizer_class)
        result = eval_checkpoint(args, model, tokenizer)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

    return results



