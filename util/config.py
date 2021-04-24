import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--bert_path', help='config file', default='bert-base-chinese')
parser.add_argument('--save_path', help='training log', default='train')
parser.add_argument('--data_root', help='data root', default='data')
parser.add_argument('--train_file', help='training file', default='train.txt')
parser.add_argument('--validation_file', help='validation file', default='dev.txt')
parser.add_argument('--label_path', help='label file', default='classes.txt')
parser.add_argument('--model_root_path', help='model root path', default='pth')
parser.add_argument('--lr', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--batch_split', type=int, default=3)
parser.add_argument('--eval_steps', type=int, default=100)
parser.add_argument('--epoch_size', type=int, default=30)
parser.add_argument('--max_length', type=int, default=90)
parser.add_argument('--num_labels', type=int, default=14)
parser.add_argument('--weight_decay', type=int, default=0.01)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--n_jobs', type=int, default=0, help='num of workers to process data')

parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')

args = parser.parse_args()
