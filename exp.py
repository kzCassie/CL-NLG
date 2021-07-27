import logging

from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          CTRLConfig, CTRLLMHeadModel, CTRLTokenizer,
                          XLNetConfig, XLNetLMHeadModel, XLNetTokenizer,
                          TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer,
                          XLMConfig, XLMWithLMHeadModel, XLMTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

from common.utils import init_arg_parser, check_config, load_checkpoint
from components.train import train
from components.evaluate import evaluate
from components.generate import decode

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'ctrl': (CTRLConfig, CTRLLMHeadModel, CTRLTokenizer),
    'xlnet': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}


if __name__ == '__main__':
    parser = init_arg_parser()
    args = check_config(parser)

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model, tokenizer = load_checkpoint(args.model_loc, model_class, tokenizer_class)
    # logging.info("Command line parameters: %s", args)

    # run experiment
    if args.mode == "train":
        model, history_losses = train(args, model, tokenizer)
    elif args.mode == "eval":
        evaluate(args, model, tokenizer)
    elif args.mode == "decode":
        decode(args, model, tokenizer)
