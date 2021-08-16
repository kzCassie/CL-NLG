import logging
import torch
import json
from tqdm import trange
from transformers import Adafactor, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, RandomSampler


from common.utils import set_seed, save_checkpoint
from common.data import get_dataset, get_data_loader, get_collate_fn
from common.curriculum import BucketCurriculum, intent_slot_score_fn, SplRegularizer
from components.evaluate import evaluate_loss, evaluate_bleu


def train_with_dataloader(args, train_dataloader, model, tokenizer, eval_dataloader, len_eval_dataset,
                          spl_regularizer=None):
    # steps
    # t_total = len(train_dataloader) * args.num_train_epochs

    # optimizer
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate,
                          eps=(1e-30, 1e-3),
                          clip_threshold=1.0,
                          decay_rate=-0.8,
                          beta1=None,
                          weight_decay=0.0,
                          relative_step=False,
                          scale_parameter=False,
                          warmup_init=False)

    global_step, logging_loss, patience, best_epoch_loss = 0, 0, 0, float('inf')
    batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen = [], [], [], []
    eval_losses = []
    model.train()

    if spl_regularizer:  # to get loss for each individual sample
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        running_loss, running_ex = 0.0, 0  # accumulated loss of each epoch

        # sample weights for each batch
        if spl_regularizer:
            prev_v_s = [1] * len(train_dataloader)  # v from prev epoch
            v_s = []

        for step, batch in enumerate(train_dataloader):
            # logging.info(f"  PROGRESS: {float(global_step) / t_total * 100:.2f}%")

            inputs, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.zero_grad()
            outputs = model(inputs, labels=labels)

            if spl_regularizer:
                tmp_loss = loss_fct(outputs.logits.permute(1, 2, 0), labels.permute(1, 0))
                sample_loss = torch.mean(tmp_loss, dim=0)  # token avg loss for each sample
                # assert(torch.mean(sample_loss) == outputs.loss)  # TODO: minor mean vs major mean?

                loss = torch.mean(prev_v_s[step] * sample_loss)
                v = spl_regularizer.v(sample_loss)  # new sample weight
                v_s.append(v)
            else:
                loss = outputs.loss
            loss.backward()

            running_loss += loss.item() * len(labels)
            running_ex += len(labels)
            batch_losses.append(loss.item())
            batch_ex_seen.append(len(labels))

            # # clip gradient
            # if args.max_grad_norm > 0.:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
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

        if spl_regularizer:
            spl_regularizer.update_hyper()
            prev_v_s = v_s

        epoch_loss = running_loss / running_ex
        epoch_losses.append(epoch_loss)
        epoch_ex_seen.append(running_ex)
        best_epoch_loss = min(best_epoch_loss, epoch_loss)
        logging.info("[Epoch %d] Running loss = %.4f", e + 1, epoch_loss)

        # validating & early stopping
        if eval_dataloader:
            dev_loss = - evaluate_bleu(eval_dataloader, len_eval_dataset, model, tokenizer, args)  # TODO: dev BLEU
            # dev_loss = evaluate_loss(eval_dataloader, len_eval_dataset, model, args, verbose=False)
            is_better = eval_losses == [] or dev_loss < min(eval_losses)
            eval_losses.append(dev_loss)
            logging.info("[Epoch %d] Running loss = %.4f  Dev loss = %.4f", e + 1, epoch_loss, dev_loss)

            if e <= 50 or is_better:
                patience = 0
                save_checkpoint(args.output_dir, model, tokenizer, args)
            else:
                patience += 1

            if 0 < args.train_patience <= patience:
                print('early stop')
                logging.info(f"Max patience {args.train_patience} hit. Early stopping at epoch {e}.")
                break
        else:
            save_checkpoint(args.output_dir, model, tokenizer, args)

    return model, best_epoch_loss, batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen, eval_losses  # TODO: eval_losses is actually neg BLEU on dev set


def train_with_one_bucket(args, model, tokenizer, eval_dataloader, len_eval_dataset, spl_regularizer=None):
    """ View entire training data as one curriculum """
    # train dataloader
    train_dataloader, len_train_dataset = get_data_loader(args, tokenizer, args.train_data_file, args.train_tgt_file,
                                                          args.train_batch_size, RandomSampler)

    # logging
    if spl_regularizer is None:
        logging.info("***** Running vanilla training *****")
    else:
        logging.info(f"***** Running training with {spl_regularizer.name} SPL Regularizer *****")
    logging.info("  Num examples = %d", len_train_dataset)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Training batch size = %d", args.train_batch_size)

    # train
    model, best_epoch_loss, batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen, eval_losses \
        = train_with_dataloader(args, train_dataloader, model, tokenizer,
                                eval_dataloader, len_eval_dataset, spl_regularizer)
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


def train_bucket_curriculum(args, model, tokenizer, eval_dataloader, len_eval_dataset, score_fn):
    # data
    dataset = get_dataset(args.train_data_file, args.train_tgt_file, args.data_cache_dir, args.overwrite_cache)

    # bucket curriculum
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
            curr_eval_losses = train_with_dataloader(args, curriculum_dataloader, model, tokenizer,
                                                     eval_dataloader, len_eval_dataset)
        best_epoch_loss = min(best_epoch_loss, curr_best_epoch_loss)
        logging.info("  Loss = %.4f", best_epoch_loss)

        batch_losses.append(curr_batch_losses)
        batch_ex_seen.append(curr_batch_ex_seen)
        epoch_losses.append(curr_epoch_losses)
        epoch_ex_seen.append(curr_epoch_ex_seen)
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

    # eval dataloader
    if args.dev_data_file:
        eval_dataloader, len_eval_dataset = get_data_loader(args, tokenizer, args.dev_data_file, args.dev_tgt_file,
                                                            args.dev_batch_size, SequentialSampler)
    else:
        eval_dataloader, len_eval_dataset = None, -1

    # spl
    if "spl" in args.curriculum_name:
        spl_reg_name = args.curriculum_name.split('.')[-1]  # {soft, linear, mixture}
        spl_regularizer = SplRegularizer(spl_reg_name, lam=0.3)  # TODO: init lambda
    else:
        spl_regularizer = None

    # Train!
    if args.curriculum_name == "NC" or "spl" in args.curriculum_name:
        model = train_with_one_bucket(args, model, tokenizer, eval_dataloader, len_eval_dataset, spl_regularizer)
    elif args.curriculum_name in ["one_pass", "baby_step"]:
        model = train_bucket_curriculum(args, model, tokenizer, eval_dataloader, len_eval_dataset, intent_slot_score_fn)
    else:
        raise ValueError("Invalid args.curriculum_name.")

    return model
