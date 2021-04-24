import torch
from util.common_util import splice_path


class Executor:
    def __init__(self, args, trainer, start_epoch, logger):
        self.config = args
        self.trainer = trainer
        self.start_epoch = start_epoch
        self.logger = logger

    def execute(self):
        self.logger.info("Start Training")
        self.trainer.train(self.start_epoch,
                           self.config.epoch_size,
                           after_epoch_funcs=[self.save_func],
                           after_step_funcs=[])

    def save_func(self, epoch):
        filename = self.get_ckpt_filename('model', epoch)
        self.logger.info("Save Model as ", filename)
        torch.save(self.trainer.state_dict(), splice_path(self.config.model_root_path, filename))

    def get_epoch_from_ckpt(self, ckpt):
        return int(ckpt.split('-')[-1].split('.')[0])

    def get_ckpt_filename(self, name, epoch):
        return '{}-{}.ckpt'.format(name, epoch)
