import os
import logging
import torch

from tqdm import trange
from functools import partial
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from common.data import FewShotWozDataset, enc_dec_collate_fn, lm_collate_fn
from common.utils import rotate_checkpoints, set_seed, save_checkpoint


def get_data_loader(args, tokenizer):
    def get_dataset():
        if not args.overwrite_cache and os.path.exists(args.data_cache_path):
            logging.info("Loading processed data from cached file %s", args.data_cache_path)
            dataset = FewShotWozDataset.from_bin_file(args.data_cache_path)
        else:
            logging.info("Creating features from dataset file at %s. Caching to %s.",
                         args.train_data_file, args.data_cache_path)
            dataset = FewShotWozDataset.from_txt_file(args.train_data_file, args.data_cache_path, '&')
        return dataset

    def get_sampler(dataset):
        return RandomSampler(dataset)

    def get_collate_fn():
        if args.enc_dec:
            fn = partial(enc_dec_collate_fn, tokenizer=tokenizer, max_intent_len=args.max_intent_len,
                         max_utter_len=args.max_utter_len)
        else:
            fn = partial(lm_collate_fn, tokenizer=tokenizer, max_len=args.max_len, separator='&')
        return fn

    train_dataset = get_dataset()
    train_sampler = get_sampler(train_dataset)
    collate_fn = get_collate_fn()
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  sampler=train_sampler, collate_fn=collate_fn)
    return train_dataloader, len(train_dataset)


def train(args, model, tokenizer):
    set_seed(args)
    model.to(args.device)
    tb_writer = SummaryWriter()

    # data
    train_dataloader, len_train_dataset = get_data_loader(args, tokenizer)

    # steps
    t_total = len(train_dataloader) * args.num_train_epochs

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len_train_dataset)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Training batch size = %d", args.train_batch_size)

    global_step = 0
    logging_loss, tr_loss = 0.0, 0.0
    model.train()

    for e in trange(int(args.num_train_epochs), desc="Epoch"):

        for step, batch in enumerate(train_dataloader):
            logging.info(f"  PROGRESS: {float(global_step) / t_total * 100}%")

            inputs, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.zero_grad()
            outputs = model(inputs, labels=labels)

            loss = outputs.loss
            loss.backward()
            tr_loss += loss.item()

            # clip gradient
            if args.max_grad_norm > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            global_step += 1

            # logging
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging.info(f"  EVALERR:  {(tr_loss - logging_loss) / float(args.logging_steps)}")
                logging_loss = tr_loss

            # Save checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, '{}-{}'.format('checkpoint', global_step))
                logging.info("Saving model checkpoint to %s", output_dir)
                save_checkpoint(output_dir, model, tokenizer, args)
                rotate_checkpoints(args, 'checkpoint')

    tb_writer.close()
    avg_loss = tr_loss / global_step  # avg training loss so far

    ## Saving ##
    logging.info("Saving model checkpoint to %s", args.output_dir)
    save_checkpoint(args.output_dir, model, tokenizer, args)
    return global_step, avg_loss
