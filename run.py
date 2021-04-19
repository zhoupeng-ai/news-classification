from util.data.dataset import THUCNewsDataset
from util.data.confg import DataConfig
from util.common_util import (
    is_use_cuda,
    init_seed
)
from util.config import TrainConfig
from bert import bert_processor
from util.log_util import Logger
from trainer import BertTrainer
from executor import Executor
import time


today = time.strftime("%Y-%m-%d", time.localtime())
logger = Logger(log_path="./log/model_train_"+today+".log")
init_seed(seed=2021)

data_config = DataConfig()
train_config = TrainConfig()
train_config.reset_config(num_labels=14)
config, tokenizer, model = bert_processor(train_config)

train_dataset = THUCNewsDataset(args=data_config, path=data_config.train_path, tokenizer=tokenizer)

val_dataset = THUCNewsDataset(args=data_config, path=data_config.validation_path, tokenizer=tokenizer)

trainer = BertTrainer(args=train_config,
                      model=model,
                      tokenizer=tokenizer,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset)

executor = Executor(args=train_config,
                    trainer=trainer,
                    start_epoch=0)

executor.execute()
