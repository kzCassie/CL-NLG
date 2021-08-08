import torch
import logging
import json
from tqdm import tqdm
from common.data import get_data_loader


def decode(args, model, tokenizer):
    dataloader, len_dataset = get_data_loader(args, tokenizer, mode='decode')

    # Decode!
    logging.info("***** Decoding *****")
    logging.info("  Num examples = %d", len_dataset)
    logging.info("  Batch size = %d", args.decode_batch_size)

    model.to(args.device)
    model.eval()

    eval_loss = 0.0
    outputs = []  # predicted utterance

    for batch in tqdm(dataloader, desc="Decoding", total=len(dataloader)):
        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        example_ids = model.generate(inputs)
        examples = tokenizer.batch_decode(example_ids, skip_special_tokens=True)
        outputs.extend(examples)

        # Evaluate
        with torch.no_grad():
            loss = model(inputs, labels=labels).loss
            eval_loss += loss.item() * len(labels)

    # Avg Evaluation
    avg_loss = eval_loss / len_dataset
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    results = {
        'avg loss': avg_loss,
        'perplexity': perplexity,
    }

    # save
    path1 = f"{args.output_dir}/decode_loss.json"
    json.dump(results, open(path1, 'w'), indent=2)
    logging.info("Decoding loss saved to {}".format(path1))

    path2 = f"{args.output_dir}/results.txt"
    with open(path2, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(outputs))
    logging.info("Decoded file saved to {}".format(path2))
    return outputs, results
