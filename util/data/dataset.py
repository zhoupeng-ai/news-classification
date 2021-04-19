import torch
from torch.utils.data import Dataset
from util.common_util import (splice_path)


def load_label_vocab(label_path):
    with open(label_path, encoding="utf-8") as f:
        res = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
    return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)), res))  # list, token2index, index2token


class THUCNewsDataset(Dataset):
    """
        THUCNews 新闻数据集的数据处理里类
    """

    def __init__(self, args, path, tokenizer, logger, max_lengths=2048):
        """
            args: 数据加载的基本参数
            path: 数据的源路径
            tokenizer: 分词器
            logger: 日志处理对象
            max_lengths: 序列的最大长度 默认：2048

        """
        super(THUCNewsDataset, self).__init__()
        self.logger = logger
        self.tokenizer = tokenizer
        self.config = args
        self.max_lengths = max_lengths
        self.label_vocal = load_label_vocab(splice_path(args.root, args.label_path))
        self.path = splice_path(args.root, path)
        self.data = THUCNewsDataset.data_processor(path=self.path,
                                                   logger=logger,
                                                   tokenizer=tokenizer,
                                                   label_vocal=self.label_vocal,
                                                   max_length=self.max_lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label_vocab, sent_token, attention_mask = self.data[item]
        return {'label': label_vocab, 'sent_token': sent_token, 'attention_mask': attention_mask}

    @staticmethod
    def data_processor(path, logger, tokenizer, label_vocal, max_length):
        dataset = []
        logger.info('reading data from {}'.format(path))
        with open(path, 'r', encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines() if len(line.strip()) != 0]
            lines = [line.split(",") for line in lines]
            for line in lines:
                label, text = line[0], line[1]
                sent_token = tokenizer(text[:max_length])
                dataset.append([int(label),
                                sent_token['input_ids'],
                                sent_token['attention_mask']])

        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset


class PadTHUCNewsSeqFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        res = dict()
        res['label'] = torch.tensor([i['label'] for i in batch]).long()
        max_length = max([len(i['sent_token']) for i in batch])
        res['sent_token'] = torch.tensor([i['sent_token'] +
                                          [self.pad_idx] * (max_length - len(i['sent_token']))
                                          for i in batch]).long()
        res['attention_mask'] = torch.tensor([i['attention_mask'] +
                                              [self.pad_idx] * (max_length - len(i['attention_mask']))
                                              for i in batch]).long()

        return res



