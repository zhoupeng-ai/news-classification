import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.data.dataset import PadTHUCNewsSeqFn
from util.common_util import (get_device, splice_path)
from optim import Adam, NoamOpt
from torch.utils.tensorboard import SummaryWriter
import logging as logger
from tqdm import tqdm


class BertTrainer:
    def __init__(self, args, model, tokenizer, train_dataset, validation_dataset, valid_writer=None):
        self.config = args
        self.device = get_device()
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        basic_optim = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optim = NoamOpt(self.config.embed_size, 0.1, self.config.lr_warmup, basic_optim)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           collate_fn=PadTHUCNewsSeqFn(tokenizer.pad_token_id),
                                           shuffle=True,
                                           pin_memory=True)
        self.val_dataloader = DataLoader(validation_dataset,
                                         batch_size=args.batch_size,
                                         collate_fn=PadTHUCNewsSeqFn(tokenizer.pad_token_id),
                                         shuffle=True,
                                         pin_memory=True)

        self.train_writer = SummaryWriter(splice_path(args.log_dir, 'train_cls'))
        self.valid_writer = SummaryWriter(splice_path(args.log_dir, 'valid_cls'))

    def _train(self, epoch):
        self.model.train()
        logger.info("Epoch: {}th".format(epoch))
        loss, acc, step_count = 0., 0., 0
        total = len(self.train_dataloader)

        tqdm_processor = tqdm(enumerate(self.train_dataloader),
                              desc='Train (epoch #{})'.format(epoch),
                              dynamic_ncols=True,
                              total=total)

        for i, batch_data in tqdm_processor:
            label, text = batch_data['label'].to(self.device), batch_data['sent_token'].to(self.device)
            attention_mask = batch_data['attention_mask'].to(self.device)
            output = self.model(text, attention_mask=attention_mask, return_dict=True)

            batch_loss = self.criterion(output.logits, label)
            batch_acc = (torch.argmax(output.logits, dim=1) == label).float().mean()
            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()

            loss += batch_loss.item()
            acc += batch_acc.item()
            step_count += 1

            curr_step = self.optim.curr_step()
            lr = self.optim.param_groups[0]["lr"]
            if (i + 1) % self.config.batch_split == 0:
                self.optim.step()
                self.optim.zero_grad()

                loss /= step_count
                acc /= step_count

                self.train_writer.add_scalar('ind/loss', loss, curr_step)
                self.train_writer.add_scalar('ind/acc', acc, curr_step)
                self.train_writer.add_scalar('ind/lr', lr, curr_step)
                tqdm_processor.set_postfix({'loss': loss, 'acc': acc})

                loss, acc, step_count = 0., 0., 0

                if curr_step % self.config.eval_steps == 0:
                    self._eval_train(epoch, curr_step)

    def _eval_train(self, epoch, curr_step):
        self.model.eval()
        with torch.no_grad():
            all_logits = []
            all_label = []
            for batch_data in self.val_dataloader:
                text, label = batch_data['sent_token'].to(self.device), batch_data['label'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)

                output = self.model(text, attention_mask=attention_mask, return_dict=True)
                all_label.append(label)
                all_logits.append(output.logits)

            all_logits = torch.cat(all_logits, dim=0)
            all_label = torch.cat(all_label, dim=0)

            val_loss = self.criterion(all_logits, all_label).float()
            val_acc = (torch.argmax(all_logits, dim=1) == all_label).float().mean()

            self.valid_writer.add_scalar('ind/loss', val_loss, curr_step)
            self.valid_writer.add_scalar('ind/acc', val_acc, curr_step)
            log_str = 'epoch {:>3}, step {}'.format(epoch, curr_step)
            log_str += ', loss {:>4.4f}'.format(val_loss)
            log_str += ', acc {:>4.4f}'.format(val_acc)
            logger.info(log_str)
        self.model.train()

    def train(self, start_epoch, epoch_size, after_epoch_funcs=[], after_step_funcs=[]):
        logger.info('Training Start')
        for epoch in range(start_epoch+1, epoch_size):
            logger.info('Training on epoch'.format(epoch))
            self._train(epoch)
            for after_fn in after_epoch_funcs:
                after_fn(epoch, self.device)
        logger.info('Training Stop')

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)