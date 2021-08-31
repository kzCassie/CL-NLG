"""Parse command line arguments and validate their eligibility."""
import argparse
import os
import re
import shutil
import glob
import random
import torch
import logging
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_arg_parser():
    parser = argparse.ArgumentParser()

    ### General configuration ###
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")
    parser.add_argument("--mode", default="", type=str, choices=["train", "decode", "eval"],
                        required=True, help="Choose the running mode for the script.")

    parser.add_argument("--model_type", choices=['gpt2', 'openai-gpt', 'bert', 'roberta', 'distilbert', 't5'],
                        default="bert", type=str, help="The model architecture to be initialized.")
    parser.add_argument("--model_name", default="", type=str,
                        help="The shortcut name of the pre-trained model weights for initialization.")
    parser.add_argument("--model_path", default="", type=str,
                        help="The path of model weights checkpoint for initialization.")

    ### Cuda & Distributed Training ###
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Do not use gpu")

    ### Training ###
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--train_tgt_file", default=None, type=str,
                        help="The target training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_while_train", action='store_true', help="Validate every epoch during training")

    # Training schedule details
    parser.add_argument("--train_batch_size", default=1, type=int, help="Training batch size of DataLoader")
    parser.add_argument("--train_patience", default=-1, type=int, help="Max epoch without improvements")
    parser.add_argument('--overwrite_output_dir', default=False, action='store_true',
                        help="Overwrite the content of the output directory")
    # parser.add_argument("--valid_every_epoch", default=10, type=int,
    #                     help="Perform validation. Only save model that is deemed better after each validation")

    parser.add_argument("--block_size", default=80, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "When block_size<=0, use the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--max_steps", default=-1, type=int,
    #                     help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    ### Curriculum ###
    parser.add_argument("--curriculum_name", default="NC", type=str, help="[NC, one_pass, baby_step]")
    parser.add_argument("--curriculum_num_bucket", default=5, type=int, help="Num of curriculum buckets.")

    # Dynamic CL
    parser.add_argument("--dcl_baseline", default="", type=str, help="Trained baseline model for dynamic CL")
    parser.add_argument("--dcl_phase", default=5, type=int, help="Num of phases for dynamic CL.")
    parser.add_argument("--dcl_a", default=1, type=int, help="Num of warming-up phases for dynamic CL.")
    parser.add_argument("--dcl_c0", default=0.2, type=float, help="Percentage of the training set to be included in the first phase")
    parser.add_argument("--dcl_alpha", default=0.3, type=float, help="Weight of slot accuracy against BLEU when calculating model competence")
    parser.add_argument("--dcl_beta", default=0.9, type=float, help="Model competence measure hyper-param.")

    ### Dev ###
    parser.add_argument("--dev_data_file", default="", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a dev set).")
    parser.add_argument("--dev_tgt_file", default="", type=str)
    parser.add_argument("--dev_batch_size", default=20, type=int, help="Batch size for validating.")

    ### Decoding ###
    parser.add_argument('--decode_input_file', type=str, default=None, help="File to be decoded")
    parser.add_argument('--decode_tgt_file', type=str, default=None)
    parser.add_argument('--decode_output_file', type=str, default=None, help="Decoded utterance strings")
    parser.add_argument("--decode_batch_size", default=20, type=int, help="Batch size for decoding.")

    ### Evaluating ###
    parser.add_argument('--eval_output_file', type=str, default=None, help="Decoded text file.")
    parser.add_argument('--eval_tgt_file', type=str, default=None, help="Targeted utterances.")
    parser.add_argument("--eval_batch_size", default=20, type=int, help="Batch size for decoding.")

    # parser.add_argument("--top_k", type=int, default=0)
    # parser.add_argument("--top_p", type=float, default=0.9)
    # parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")

    # parser.add_argument("--length", type=int, default=40)
    # parser.add_argument("--num_samples", type=int, default=1)
    #
    # parser.add_argument('--stop_token', type=str, default=None, help="Token at which text generation is stopped")
    # parser.add_argument('--nc', type=int, default=1, help="number of sentence")
    # parser.add_argument("--use_token", action='store_true', help="")

    ### Model configuration ###
    # parser.add_argument("--do_lower_case", default=False, action='store_true',
    #                     help="Set this flag if you are using an uncased model.")

    ### Data Processing ###
    # TODO: these options all related to data loading?
    parser.add_argument('--overwrite_cache', default=False, action='store_true',
                        help="Overwrite the cached processed dataset for training and evaluation")
    parser.add_argument('--data_cache_dir', default="", type=str, help="Dir to cache preprocessed data bin file")
    parser.add_argument('--max_intent_len', default=40, type=int, help="Max intention length (for EncDec model)")
    parser.add_argument('--max_utter_len', default=60, type=int, help="Max utterance length (for EncDec model)")

    # parser.add_argument('--max_len', default=80, type=int, help="Max raw sentence length (for LM model)")
    # parser.add_argument('--text_chunk', default=False, action='store_true',
    #                     help="Read the text data file in one chunk instead of reading it line by line")
    # parser.add_argument('--use_reverse', default=False, action='store_true', help="")
    # parser.add_argument('--with_code_loss', type=bool, default=True, help="")
    # parser.add_argument('--use_tokenizer', default=False, action='store_true',
    #                     help="Use pretrained tokenizer. If false, do a simple split by space")
    # parser.add_argument("--max_seq", default=80, type=int,
    #                     help="Max num tokens when loading text (including tokens for both code and utterance)")

    # ### Logging and save ###
    # parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=5000, help="Save checkpoint every X updates steps.")
    # parser.add_argument('--save_total_limit', type=int, default=None,
    #                     help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, '
    #                          'does not delete by default')
    # parser.add_argument("--eval_all_checkpoints", default=False, action='store_true',
    #                     help="Evaluate all checkpoints starting with the same prefix as model_name or model_path "
    #                          "and ending with step number. When false, only evaluate the model in model_path")
    return parser


def check_config(parser):
    """ Perform sanity checks on command line parsed arguments"""
    args = parser.parse_args()

    ### generic config ###
    set_seed(args.seed)
    args.enc_dec = True if args.model_type in ['t5'] else False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init logging
    os.makedirs(f"{args.output_dir}/log", exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', level=logging.INFO,
                        filename=f"{args.output_dir}/log/{args.mode}.log")
    logging.info("Processing with device: %s", args.device)

    # base model path
    if args.mode == "train":
        args.model_loc = args.model_name
    elif args.mode == "decode":
        args.model_loc = args.model_path
    else:
        args.model_loc = ""

    ### mode ###
    if args.mode == 'train':
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty. Use "
                             "--overwrite_output_dir to overcome.".format(args.output_dir))
    return args


def rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    # do not delete older checkpoints
    if not args.save_total_limit or args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logging.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def load_checkpoint(model_loc, model_class, tokenizer_class):
    model = model_class.from_pretrained(model_loc)
    tokenizer = tokenizer_class.from_pretrained(model_loc)
    return model, tokenizer


def save_checkpoint(output_dir, model, tokenizer, args):
    # if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
