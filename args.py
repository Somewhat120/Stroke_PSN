import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device_id', default='0', type=str,
                    help='Set the device (GPU ids)')
parser.add_argument('-da', '--dataset', default='sepsis', type=str,
                    help='Set the data set for training')
parser.add_argument('-sp', '--saved_path', default='saved', type=str,
                    help='Path to save training results')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed to be used')
parser.add_argument('-mo', '--mode', default='gnn', type=str,
                    choices=['gnn', 'automl', 'ml'],
                    help='Whether to use GNN or AutoML model')
# Training Arguments
parser.add_argument('-st', '--split_type', default='train_val_test', type=str,
                    choices=['train_val_test', 'train_val_test_ordered', 'kfold'],
                    help='The type of data splitting')
parser.add_argument('-sa', '--sampling', default='no_sp', type=str,
                    choices=['no_sp', 'sp'],
                    help='Whether to sample the input graph to save working memory')
parser.add_argument('-pr', '--print_every', default=1, type=int,
                    help='The number of epochs to print a training record')
parser.add_argument('-fo', '--nfold', default=5, type=int,
                    help='The number of k in k-fold cross validation')
parser.add_argument('-ep', '--epoch', default=1000, type=int,
                    help='The number of epochs for model training')
parser.add_argument('-bs', '--batch_size', default=1024, type=int,
                    help='The size of a batch to be used')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
                    help='Learning rate to be used in optimizer')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to be used')
parser.add_argument('-ck', '--check_metric', default='loss', type=str,
                    choices=['loss', 'auc', 'aupr'],
                    help='weight decay to be used')
# Model Argument
parser.add_argument('-k', '--k', default=10, type=int,
                    help='The number of topk similarities to be binarized')
parser.add_argument('-nh', '--num_heads', default=5, type=int,
                    help='The number of attention heads in GAT')
parser.add_argument('-hf', '--hidden_feats', default=32, type=int,
                    help='The dimension of hidden tensor in the model')
parser.add_argument('-dp', '--dropout', default=0., type=float,
                    help='The rate of dropout layer')

args = parser.parse_args()
args.datapath = 'dataset/' + args.dataset + '.csv'
# if args.mode == 'gnn':
args.saved_path = 'result/' + args.dataset + '/' \
                  + args.saved_path + '_' + str(args.seed)
# elif args.mode == 'automl':
#     args.saved_path = 'result/' + args.dataset + '/' \
#                       + args.saved_path + '_' + str(args.seed) + args.mode
