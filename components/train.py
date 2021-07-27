import os
import logging
import torch

from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from common.utils import rotate_checkpoints, set_seed, save_checkpoint
from common.data import get_dataset, get_data_loader, get_collate_fn
from common.curriculum import BucketCurriculum, length_score_fn


def train_with_dataloader(args, train_dataloader, model, tokenizer):
    # steps
    t_total = len(train_dataloader) * args.num_train_epochs

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    global_step = 0
    logging_loss, tr_loss, patience = 0.0, 0.0,0
    history_losses = []
    eval_scores = []
    model.train()

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # logging.info(f"  PROGRESS: {float(global_step) / t_total * 100:.2f}%")

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

            # # logging
            # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            #     tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
            #     tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
            #     logging.info(f"  EVALERR:  {(tr_loss - logging_loss) / float(args.logging_steps)}")
            #     logging_loss = tr_loss
            #
            # # Save checkpoint
            # if args.save_steps > 0 and global_step % args.save_steps == 0:
            #     output_dir = os.path.join(args.output_dir, '{}-{}'.format('checkpoint', global_step))
            #     logging.info("Saving model checkpoint to %s", output_dir)
            #     save_checkpoint(output_dir, model, tokenizer, args)
            #     rotate_checkpoints(args, 'checkpoint')

        # logging.info("[Epoch %d] Running loss = %.4f", e + 1, running_loss)
        history_losses.append(running_loss)

        # early stopping
        if e % args.valid_every_epoch == 0:  # TODO: validation
            is_better = eval_scores == [] or running_loss < min(eval_scores)
            eval_scores.append(running_loss)

            if is_better:
                patience = 0
                # logging.info("Saving model to %s", args.output_dir)
                save_checkpoint(args.output_dir, model, tokenizer, args)
            else:
                patience += 1
                # logging.info("Hit patience %d", patience)

            if args.train_patience > 0 and patience >= args.train_patience:
                logging.info(f"Max patience {args.train_patience} hit. Early stopping.")
                break

    return model, history_losses


def train_without_curriculum(args, model, tokenizer):
    train_dataloader, len_train_dataset = get_data_loader(args, tokenizer)

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len_train_dataset)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Training batch size = %d", args.train_batch_size)

    model, history_losses = train_with_dataloader(args, train_dataloader, model, tokenizer)
    return model, history_losses


def train_bucket_curriculum(args, model, tokenizer, score_fn):
    # data
    dataset = get_dataset(args.train_data_file, args.data_cache_path, args.overwrite_cache)
    curriculums = BucketCurriculum(dataset, score_fn)\
        .get_curriculum(bucket_size=args.curriculum_bucket_size, batch_size=args.train_batch_size,
                        name=args.curriculum_name, collate_fn=get_collate_fn(args, tokenizer))

    # Train!
    logging.info("***** Running training *****")
    for idx, curriculum in enumerate(curriculums):
        train_dataloader, len_train_dataset = curriculum

        logging.info("  Curriculum = %d", idx+1)
        logging.info("  Num examples = %d", len_train_dataset)
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Training batch size = %d", args.train_batch_size)

        model, history_losses = train_with_dataloader(args, train_dataloader, model, tokenizer)

        # evaluate on entire training set
        # TODO: validate

    return model, [-1]


def train(args, model, tokenizer):
    set_seed(args.seed)
    model.to(args.device)
    # tb_writer = SummaryWriter()

    # Train!
    if args.curriculum_name == "NC":
        model, history_losses = train_without_curriculum(args, model, tokenizer)
    elif args.curriculum_name in ["one_pass", "baby_step"]:
        model, history_losses = train_bucket_curriculum(args, model, tokenizer, length_score_fn)
    else:
        raise ValueError("Invalid args.curriculum_name.")

    # tb_writer.close()
    logging.info("[Epoch %d] Running loss = %.4f", len(history_losses), history_losses[-1])
    return model, history_losses
