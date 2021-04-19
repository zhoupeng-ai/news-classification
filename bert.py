from transformers import BertConfig, BertTokenizer, BertForSequenceClassification


def bert_processor(args):

    config = BertConfig.from_pretrained(args.bert_path)
    config.num_labels = args.num_labels
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertForSequenceClassification.from_pretrained(args.bert_path, config=config)
    return config, tokenizer, model
