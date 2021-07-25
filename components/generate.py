import torch
import logging
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from common.data import get_data_loader


def decode(args, model, tokenizer):
    dataloader, len_dataset = get_data_loader(args, tokenizer)

    # Decode!
    logging.info("***** Decoding *****")
    logging.info("  Num examples = %d", len_dataset)
    logging.info("  Batch size = %d", args.decode_batch_size)

    model.to(args.device)
    model.eval()

    eval_loss = 0.0
    bleu_scores = []
    outputs = []  # predicted utterance
    for batch in tqdm(dataloader, desc="Decoding", total=len(dataloader)):
        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        example_ids = model.generate(inputs)
        examples = tokenizer.batch_decode(example_ids, skip_special_tokens=True)
        label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)  # get tgt utter
        outputs.extend(examples)

        # Evaluate
        with torch.no_grad():
            loss = model(inputs, labels=labels).loss
            eval_loss += loss.item() * len(labels)
        bleu = [sentence_bleu([label_strs[i]], examples[i]) for i in range(len(labels))]
        bleu_scores.extend(bleu)

    # Evaluation results
    avg_loss = eval_loss / len_dataset
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    results = {
        "Num examples": len_dataset,
        "Batch size": args.decode_batch_size,
        "loss": avg_loss,
        'perplexity': perplexity,
        "BLEU": avg_bleu
    }

    # save
    json.dump(outputs, open(args.decode_output_file, 'w'), indent=2)
    logging.info("Decoded file saved to {}".format(args.decode_output_file))

    json.dump(results, open(args.decode_result_file, 'w'), indent=2)
    logging.info("Evaluation results saved to {}".format(args.decode_result_file))
    return outputs, results
