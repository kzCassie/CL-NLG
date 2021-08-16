from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from common.utils import init_arg_parser, check_config, load_checkpoint
from components.train import train
from components.generate import decode
from components.evaluate import evaluate_output

MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    # 'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    # 'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    # 'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    # 'ctrl': (CTRLConfig, CTRLLMHeadModel, CTRLTokenizer),
    # 'xlnet': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
    # 'transfo-xl': (TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer),
    # 'xlm': (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}


if __name__ == '__main__':
    parser = init_arg_parser()
    args = check_config(parser)

    # run experiment
    if args.mode in ["train", "decode"]:
        # Load pre-trained model
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        model, tokenizer = load_checkpoint(args.model_loc, model_class, tokenizer_class)

        if args.mode == "train":
            train(args, model, tokenizer)
        elif args.mode == "decode":
            decode(args, model, tokenizer)

    elif args.mode == "eval":
        evaluate_output(args, args.eval_output_file, args.eval_tgt_file, args.eval_batch_size)
    else:
        raise ValueError("Invalid running mode for exp.py")
