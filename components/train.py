import os
import logging
import torch

from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from common.utils import rotate_checkpoints, set_seed, save_checkpoint
from common.data import get_data_loader


def train(args, model, tokenizer):
    set_seed(args.seed)
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
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            logging.info(f"  PROGRESS: {float(global_step) / t_total * 100:.2f}%")

            inputs, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.zero_grad()
            outputs = model(inputs, labels=labels)

            loss = outputs.loss
            loss.backward()
            tr_loss += loss.item()
            running_loss += loss.item()

            # clip gradient
            if args.max_grad_norm > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            global_step += 1

            # logging
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_last_lr(), global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging.info(f"  EVALERR:  {(tr_loss - logging_loss) / float(args.logging_steps)}")
                logging_loss = tr_loss

            # Save checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, '{}-{}'.format('checkpoint', global_step))
                logging.info("Saving model checkpoint to %s", output_dir)
                save_checkpoint(output_dir, model, tokenizer, args)
                rotate_checkpoints(args, 'checkpoint')

        logging.info("[Epoch %d] Running loss = %.4f", e+1, running_loss)

    tb_writer.close()
    avg_loss = tr_loss / global_step  # avg training loss so far

    ## Saving ##
    logging.info("Saving model to %s", args.output_dir)
    save_checkpoint(args.output_dir, model, tokenizer, args)
    return global_step, avg_loss
