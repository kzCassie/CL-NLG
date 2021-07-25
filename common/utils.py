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
    parser.add_argument("--mode", default="", type=str, choices=["train", "eval", "train_with_eval", "decode"],
                        required=True, help="Choose the running mode for the script.")
    # parser.add_argument("--evaluate_during_training", default=False, action="store_true",
    #                     help="Evaluating during training")

    parser.add_argument("--model_type", choices=['gpt2', 'openai-gpt', 'bert', 'roberta', 'distilbert', 't5'],
                        default="bert", type=str, required=True, help="The model architecture to be initialized.")
    parser.add_argument("--model_name", default="", type=str,
                        help="The shortcut name of the pre-trained model weights for initialization.")
    parser.add_argument("--model_path", default="", type=str,
                        help="The path of model weights checkpoint for initialization.")

    ### Cuda & Distributed Training ###
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Do not use gpu")
    # parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
    #                     help="Batch size per GPU/CPU for training.")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
    #                     help="Batch size per GPU/CPU for evaluation.")

    ### Training ###
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Training schedule details
    parser.add_argument("--train_batch_size", default=1, type=int, help="Training batch size of DataLoader")



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

    # TODO: after modification, must keep all things together in model_path
    # Load pre-train filepath
    # parser.add_argument("--config_name", default="", type=str,
    #                     help="Optional pretrained config name or path if not the same as model_name_or_path")
    # parser.add_argument("--tokenizer_name", default="", type=str,
    #                     help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    # parser.add_argument("--cache_dir", default="", type=str,
    #                     help="Optional directory to store the pre-trained models downloaded from s3 "
    #                          "(instead of the default one)")

    ### Evaluating ###
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Batch size for evaluation.")


    ### Decoding ###
    # TODO: write help strings
    parser.add_argument('--decode_input_file', type=str, default=None, help="File to be decoded")
    parser.add_argument('--decode_output_file', type=str, default=None, help="Decoded utterance strings")
    parser.add_argument('--decode_result_file', type=str, default=None, help="Decoding result file for metric values")
    parser.add_argument("--decode_batch_size", default=1, type=int, help="Batch size for decoding.")
    parser.add_argument("--top_k", type=int, default=0)


    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")

    parser.add_argument('--stop_token', type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument('--nc', type=int, default=1, help="number of sentence")
    parser.add_argument("--use_token", action='store_true', help="")

    ### Model configuration ###
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mlm", default=False, action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    ### Data Processing ###
    # TODO: these options all related to data loading?
    parser.add_argument('--overwrite_cache', default=False, action='store_true',
                        help="Overwrite the cached processed dataset for training and evaluation")
    parser.add_argument('--data_cache_path', default="", type=str, help="Path to cache preprocessed data bin file")
    parser.add_argument('--max_intent_len', default=40, type=int, help="Max intention length (for EncDec model)")
    parser.add_argument('--max_utter_len', default=60, type=int, help="Max utterance length (for EncDec model)")
    parser.add_argument('--max_len', default=80, type=int, help="Max raw sentence length (for LM model)")




    parser.add_argument('--text_chunk', default=False, action='store_true',
                        help="Read the text data file in one chunk instead of reading it line by line")
    parser.add_argument('--use_reverse', default=False, action='store_true', help="")
    parser.add_argument('--with_code_loss', type=bool, default=True, help="")
    parser.add_argument('--use_tokenizer', default=False, action='store_true',
                        help="Use pretrained tokenizer. If false, do a simple split by space")
    parser.add_argument("--max_seq", default=80, type=int,
                        help="Max num tokens when loading text (including tokens for both code and utterance)")

    ### Logging and save ###
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, '
                             'does not delete by default')
    parser.add_argument("--eval_all_checkpoints", default=False, action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name or model_path "
                             "and ending with step number. When false, only evaluate the model in model_path")
    parser.add_argument('--overwrite_output_dir', default=False, action='store_true',
                        help="Overwrite the content of the output directory")

    ### Precision ###
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    ### Distant debugging ###
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    return parser


def check_config(parser):
    """ Perform sanity checks on command line parsed arguments"""
    command_line = """--output_dir=saved_models/t5
        --no_cuda
        --mode train
        --model_type=t5
        --model_name=t5-base
        --train_data_file=data/restaurant/train.txt 
        --eval_data_file=data/restaurant/train.txt 
        --num_train_epochs 5
        --learning_rate 5e-5 
        --use_tokenizer 
        --overwrite_output_dir
        --overwrite_cache
        """

    command_line2 = """
                --no_cuda
                --mode decode
                --model_type=t5 \
                --model_path=saved_models/t5 \
                --num_samples 5 \
                --decode_input_file=data/restaurant/test.txt \
                --top_k 5 \
                --decode_output_file=saved_models/t5/results.json \
                --length 80
                """
    args = parser.parse_args()
    # args = parser.parse_args(command_line.split())
    # args = parser.parse_args(command_line2.split())

    ### generic config ###
    set_seed(args.seed)
    args.enc_dec = True if args.model_type in ['t5'] else False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', level=logging.INFO)
    logging.info("Processing with device: %s, ", args.device)

    # which pre-trained model to load
    if not args.model_name and not args.model_path:
        raise ValueError("Must specify either --model_name or --model_path")
    elif args.model_name and args.model_path:
        raise ValueError("Only specify either model_name or model_path")
    else:
        args.model_loc = args.model_name if args.model_name else args.model_path

    ### mode ###
    if args.mode == 'train':
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
            raise ValueError("Output directory ({}) already exists and is not empty. Use "
                             "--overwrite_output_dir to overcome.".format(args.output_dir))
    if args.mode == 'eval':
        pass
    if args.mode == "decode":
        pass
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
