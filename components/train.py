import logging
import torch
import json
from tqdm import trange
from transformers import AdamW, get_linear_schedule_with_warmup

from common.utils import set_seed, save_checkpoint
from common.data import get_dataset, get_data_loader, get_collate_fn
from common.curriculum import BucketCurriculum, intent_slot_score_fn
from components.evaluate import eval_checkpoint


def train_with_dataloader(args, train_dataloader, model, tokenizer):
    # eval during training
    assert(args.dev_data_file is not None)
    eval_dataloader, len_eval_dataset = get_data_loader(args, tokenizer, mode='dev')

    # steps
    t_total = len(train_dataloader) * args.num_train_epochs

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)

    # scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    global_step, logging_loss, patience, best_epoch_loss = 0, 0, 0, -1
    batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen = [], [], [], []
    eval_losses = []
    model.train()

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        running_loss, running_ex = 0.0, 0  # accumulated loss of each epoch
        for step, batch in enumerate(train_dataloader):
            # logging.info(f"  PROGRESS: {float(global_step) / t_total * 100:.2f}%")

            inputs, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.zero_grad()
            outputs = model(inputs, labels=labels)

            loss = outputs.loss
            loss.backward()

            running_loss += loss.item() * len(labels)
            running_ex += len(labels)
            batch_losses.append(loss.item() / len(labels))
            batch_ex_seen.append(len(labels))

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

        # logging.info("[Epoch %d] Running loss = %.4f", e + 1, epoch_loss)
        epoch_loss = running_loss / running_ex
        epoch_losses.append(epoch_loss)
        epoch_ex_seen.append(running_ex)

        # validating & early stopping
        dev_loss = eval_checkpoint(eval_dataloader, len_eval_dataset, model, args, verbose=False)
        is_better = eval_losses == [] or dev_loss < min(eval_losses)
        eval_losses.append(dev_loss)

        if is_better:
            patience = 0
            # logging.info("Saving model to %s", args.output_dir)
            save_checkpoint(args.output_dir, model, tokenizer, args)
            best_epoch_loss = eval_losses[-1]
        else:
            patience += 1
            # logging.info("Hit patience %d", patience)

        if 0 < args.train_patience <= patience:
            logging.info(f"Max patience {args.train_patience} hit. Early stopping at epoch {e}.")
            break

    return model, best_epoch_loss, batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen, eval_losses


def train_without_curriculum(args, model, tokenizer):
    """ View entire training data as one curriculum """
    train_dataloader, len_train_dataset = get_data_loader(args, tokenizer, "train")
    logging.info("***** Running vanilla training *****")
    logging.info("  Num examples = %d", len_train_dataset)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Training batch size = %d", args.train_batch_size)
    model, best_epoch_loss, batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen, eval_losses \
        = train_with_dataloader(args, train_dataloader, model, tokenizer)
    logging.info("  Loss = %.4f", best_epoch_loss)

    # save history
    history = {
        "batch_losses": batch_losses,
        "batch_ex_seen": batch_ex_seen,
        "epoch_losses": epoch_losses,
        "epoch_ex_seen": epoch_ex_seen,
        "eval_losses": eval_losses
    }
    with open(f"{args.output_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=4)
    return model


def train_bucket_curriculum(args, model, tokenizer, score_fn):
    # data
    dataset = get_dataset(args.train_data_file, args.data_cache_path, args.overwrite_cache)
    curriculums = BucketCurriculum(dataset, score_fn)\
        .get_curriculum(num_bucket=args.curriculum_num_bucket, batch_size=args.train_batch_size,
                        name=args.curriculum_name, collate_fn=get_collate_fn(args, tokenizer))

    # Train!
    logging.info("***** Running bucket curriculum training *****")
    logging.info("  Training batch size = %d", args.train_batch_size)
    logging.info("  Max Epochs Per Curriculum = %d", args.num_train_epochs)

    batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen, eval_losses = [], [], [], [], []
    best_epoch_loss = float('inf')
    for idx, curriculum in enumerate(curriculums):
        curriculum_dataloader, len_curriculum_dataset = curriculum
        logging.info("  ***** Curriculum = %d *****", idx+1)
        logging.info("  Num examples = %d", len_curriculum_dataset)

        model, curr_best_epoch_loss, curr_batch_losses, curr_batch_ex_seen, curr_epoch_losses, curr_epoch_ex_seen, \
            curr_eval_losses = train_with_dataloader(args, curriculum_dataloader, model, tokenizer)
        best_epoch_loss = min(best_epoch_loss, curr_best_epoch_loss)
        logging.info("  Loss = %.4f", best_epoch_loss)

        batch_losses.append(curr_batch_losses)
        batch_ex_seen.append(curr_batch_ex_seen)  # List[List[float]], historical losses of each curriculum
        epoch_losses.append(curr_epoch_losses)  # List[int], number of examples seen of each curriculum
        epoch_ex_seen.append(curr_epoch_ex_seen)  # List[int], end epoch of each curriculum
        eval_losses.append(curr_eval_losses)

    history = {
        "batch_losses": batch_losses,
        "batch_ex_seen": batch_ex_seen,
        "epoch_losses": epoch_losses,
        "epoch_ex_seen": epoch_ex_seen,
        "eval_losses": eval_losses,
    }
    with open(f"{args.output_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=4)
    return model


# choose appropriate training mode given args
def train(args, model, tokenizer):
    set_seed(args.seed)
    model.to(args.device)
    # tb_writer = SummaryWriter()

    # Train!
    if args.curriculum_name == "NC":  # no curriculum
        model = train_without_curriculum(args, model, tokenizer)
    elif args.curriculum_name in ["one_pass", "baby_step"]:
        model = train_bucket_curriculum(args, model, tokenizer, intent_slot_score_fn)
    else:
        raise ValueError("Invalid args.curriculum_name.")

    # tb_writer.close()
    return model
