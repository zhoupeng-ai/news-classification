from util.data.dataset import THUCNewsDataset
from util.data.confg import DataConfig
from util.common_util import (
    is_use_cuda,
    init_seed,
    splice_path
)
from util.config import args
from bert import bert_processor
from util.log_util import get_logger
from trainer import BertTrainer
from executor import Executor
import time
import traceback
import os

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

logger = get_logger(filename=splice_path(args.save_path, 'train.log'))

train_path = splice_path(args.save_path, 'train')
log_path = splice_path(args.save_path, 'log')

try:

    for path in [train_path, log_path]:
        if not os.path.isdir(path):
            logger.info('cannot find {}, mkdiring'.format(path))
            os.makedirs(path)

    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    init_seed(seed=args.seed)
    data_config = DataConfig()
    config, tokenizer, model = bert_processor(args, logger)
    logger.info("Loading Training Data")
    train_dataset = THUCNewsDataset(args=args, path=args.train_file, tokenizer=tokenizer, logger=logger)
    logger.info("Loading Validation Data")
    val_dataset = THUCNewsDataset(args=args, path=args.validation_file, tokenizer=tokenizer, logger=logger)

    trainer = BertTrainer(args=args,
                          model=model,
                          logger=logger,
                          tokenizer=tokenizer,
                          train_dataset=train_dataset,
                          validation_dataset=val_dataset)

    executor = Executor(args=args,
                        trainer=trainer,
                        start_epoch=1,
                        logger=logger)

    executor.execute()

except:

    logger.error(traceback.format_exc())
