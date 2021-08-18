import logging
import torch
import json
import numpy as np
from tqdm import trange
from transformers import Adafactor
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.dataloader import DataLoader

from common.utils import set_seed, save_checkpoint, load_checkpoint
from common.data import get_dataset, get_data_loader, get_data_loader_from_dataset, get_collate_fn
from common.curriculum import BucketCurriculum, DynamicCurriculum, intent_slot_score_fn, SplRegularizer
from components.evaluate import evaluate_data_set


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
            metrics = evaluate_data_set(eval_dataloader, model, tokenizer, ["loss", "bleu", "accu"], args, False)
            dev_loss = - metrics['bleu']
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

    result = {
        "batch_losses": batch_losses,
        "batch_ex_seen": batch_ex_seen,
        "epoch_losses": epoch_losses,
        "epoch_ex_seen": epoch_ex_seen,
        "neg_bleu": eval_losses,
    }
    return model, best_epoch_loss, result


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
    model, best_epoch_loss, result = train_with_dataloader(args, train_dataloader, model, tokenizer,
                                          eval_dataloader, len_eval_dataset, spl_regularizer)
    logging.info("  Loss = %.4f", best_epoch_loss)

    # save history
    history = {
        "batch_losses": [result["batch_losses"]],
        "batch_ex_seen": [result["batch_ex_seen"]],
        "epoch_losses": [result["epoch_losses"]],
        "epoch_ex_seen": [result["epoch_ex_seen"]],
        "neg_bleu": [result["neg_bleu"]]
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

        model, best_curr_loss, result = train_with_dataloader(args, curriculum_dataloader, model, tokenizer,
                                                     eval_dataloader, len_eval_dataset)
        best_epoch_loss = min(best_epoch_loss, best_curr_loss)
        logging.info("  Loss = %.4f", best_epoch_loss)

        batch_losses.append(result["batch_losses"])
        batch_ex_seen.append(result["batch_ex_seen"])
        epoch_losses.append(result["epoch_losses"])
        epoch_ex_seen.append(result["epoch_ex_seen"])
        eval_losses.append(result["neg_bleu"])

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


def train_with_dynamic_curriculum(args, model, tokenizer, eval_dataloader, len_eval_dataset):
    """
    model, tokenizer: load from t5-small
    baseline_model, baseline_tokenizer: trained baseline model to init BLEU_T
    """
    # train_dataset
    train_dataset = get_dataset(args.train_data_file, args.train_tgt_file, args.data_cache_dir, args.overwrite_cache)

    # baseline vanilla model without CL
    model_class, tokenizer_class = type(model), type(tokenizer)
    baseline_model, baseline_tokenizer = load_checkpoint(args.dcl_baseline, model_class, tokenizer_class)

    # baseline model BLEU_T
    seq_train_dataloader, len_train_dataset = get_data_loader_from_dataset(args, train_dataset, baseline_tokenizer,
                                                                           args.dev_batch_size, SequentialSampler)

    metrics = evaluate_data_set(seq_train_dataloader, baseline_model, baseline_tokenizer, ["bleu", "accu"], args)
    bleu_T, accu_T = metrics["bleu"], metrics["accu"]

    # train by phases
    batch_losses, batch_ex_seen, epoch_losses, epoch_ex_seen, eval_losses = [], [], [], [], []
    best_epoch_loss = float('inf')

    dcl = DynamicCurriculum(train_dataset)
    phase_losses = np.empty((0, len_train_dataset))  # historical losses for past 'a' phases
    # phase_accu = np.empty((0, len_train_dataset))  # historical slot accuracies for past 'a' phases
    c_s = [args.dcl_c0]  # model competences

    for t in trange(int(args.dcl_phase), desc="Phase"):
        logging.info(f"Dynamic CL - [Phase {t+1}]")

        # get training sample losses
        seq_train_dataloader, len_train_dataset = get_data_loader_from_dataset(args, train_dataset, tokenizer,
                                                                               args.dev_batch_size, SequentialSampler)
        metrics = evaluate_data_set(seq_train_dataloader, model, tokenizer, ["loss"], args, True)
        phase_losses = np.append(phase_losses, [metrics["loss"]], axis=0)
        # phase_accu = np.append(phase_accu, [metrics["accu"]], axis=0)

        # get sample difficulties
        if t < args.dcl_a:
            difficulties = phase_losses[-1]
        else:
            difficulties = (phase_losses[-1] - phase_losses[0]) / phase_losses[0]
            phase_losses = np.delete(phase_losses, 0, axis=0)
            # phase_accu = np.delete(phase_accu, 0, axis=0)

        # dev set BLEU_t
        metrics = evaluate_data_set(eval_dataloader, model, tokenizer, ["bleu", "accu"], args)
        bleu_t, accu_t = metrics["bleu"], metrics["accu"]

        # estimate model competence
        alpha = float(args.dcl_alpha)
        beta = float(args.dcl_beta)
        c_t = min(1, ((1-alpha)*bleu_t/bleu_T + alpha*accu_t/accu_T) * (1-c_s[0])/beta + c_s[0])
        c_s.append(c_t)

        # Sort samples by difficulties
        # Use the easier subset
        curriculum_dataloader = dcl.get_curriculum(difficulties, c_t, args.train_batch_size,
                                                   get_collate_fn(args, tokenizer))

        # train
        model, best_curr_loss, result = train_with_dataloader(args, curriculum_dataloader, model, tokenizer,
                                                               eval_dataloader, len_eval_dataset)

        # record training results
        best_epoch_loss = min(best_epoch_loss, best_curr_loss)
        logging.info("  Best Epoch Loss = %.4f", best_epoch_loss)

        batch_losses.append(result["batch_losses"])
        batch_ex_seen.append(result["batch_ex_seen"])
        epoch_losses.append(result["epoch_losses"])
        epoch_ex_seen.append(result["epoch_ex_seen"])
        eval_losses.append(result["neg_bleu"])

    history = {
        "batch_losses": batch_losses,
        "batch_ex_seen": batch_ex_seen,
        "epoch_losses": epoch_losses,
        "epoch_ex_seen": epoch_ex_seen,
        "eval_losses": eval_losses,
        "competence": c_s,
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
    elif args.curriculum_name == "dynamic":
        model = train_with_dynamic_curriculum(args, model, tokenizer, eval_dataloader, len_eval_dataset)
    else:
        raise ValueError("Invalid args.curriculum_name.")

    return model
