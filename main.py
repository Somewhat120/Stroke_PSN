import os
import dgl
import torch
import pandas as pd
import numpy as np
from warnings import simplefilter
from sklearn.model_selection import StratifiedKFold
from model import Model
from utils import get_metrics, get_metrics_auc, set_seed, \
    plot_curve, checkpoint, train_resample
from args import args
from torch.utils.data import DataLoader
from ML_model import rf_classifier, lr_classifier, \
    xgb_classifier, lgb_classifier, ctb_classifier
import autogluon
import statsmodels
from autogluon.tabular import TabularDataset, TabularPredictor


def generate_sn(feature, f_type):
    sims = np.zeros((feature.shape[0], feature.shape[0]))
    count = 0
    for i in range(len(f_type)):
        # sim = np.zeros((feature.shape[0], feature.shape[0]))
        if f_type[i] == 'category':
            sim = (feature[:, i] == feature[:, i].reshape(-1, 1)).astype('float16')
        elif f_type[i] == 'continous':
            sim = abs(feature[:, i] - feature[:, i].reshape(-1, 1)).astype('float16')
            sim = 1 - (sim - sim.min()) / (sim.max() - sim.min())
        sims += sim / len(f_type)
        print('feature {} processing finished'.format(count))
        count += 1
        del sim
    return sims


def train():
    simplefilter(action='ignore', category=UserWarning)
    print('Arguments: {}'.format(args))
    set_seed(args.seed)
    try:
        os.mkdir('result')
    except:
        try:
            os.mkdir('result/' + args.dataset)
        except:
            try:
                os.mkdir(args.saved_path)
            except:
                pass
        pass
    pass

    try:
        os.mkdir(args.saved_path)
    except:
        pass

    argsDict = args.__dict__
    with open(os.path.join(args.saved_path, 'setting.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    if args.device_id != 'cpu':
        print('Training on GPU')
        device = torch.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = torch.device('cpu')

    # if args.split_type == 'train_val_test_ordered':
    #     train = pd.read_csv('dataset/train_std.csv', sep='\t')
    #     val = pd.read_csv('dataset/valid_std.csv', sep='\t')
    #     test = pd.read_csv('dataset/test_std.csv', sep='\t')
    #     d_type = pd.read_csv('dataset/uti_type.csv')
    #     print('features: {}'.format(train.columns[1:]))
    #     train_feature, train_label = train.iloc[:, 1:].values.astype(float), train.iloc[:, 0].values
    #     val_feature, val_label = val.iloc[:, 1:].values.astype(float), val.iloc[:, 0].values
    #     test_feature, test_label = test.iloc[:, 1:].values.astype(float), test.iloc[:, 0].values
    # else:
    data = pd.read_csv(args.datapath)
    print('features: {}'.format(data.columns[1:]))
    d_type = pd.read_csv(args.datapath.split('.csv')[0] + '_type.csv')
    feature = data.iloc[:, 1:].values.astype(float)
    label = data.iloc[:, 0].values

    if args.mode == 'gnn':
        if args.dataset + '_sn.npy' not in os.listdir(path='./dataset'):
            sims = generate_sn(feature, d_type['Type'].values[:-1])
            np.save('./dataset/{}_sn.npy'.format(args.dataset), sims)
        else:
            sims = np.load('./dataset/{}_sn.npy'.format(args.dataset))
        sims = torch.tensor(sims).float()
        print(sims)

        _, idx = sims.topk(args.k, dim=1)
        dst_ids = idx.flatten()
        src_ids = torch.tensor([i // args.k for i in range(dst_ids.shape[0])])
        g = dgl.graph((src_ids, dst_ids))
        g = dgl.add_self_loop(g)

    kf = StratifiedKFold(args.nfold, shuffle=True, random_state=args.seed)
    fold = 1

    metric_list = []
    pred_result = np.zeros(len(label))
    if args.mode == 'ml':
        metric_list = {'LR': [], 'RF': [], 'XGB': [], 'LGB': [], 'CTB': []}
        pred_result = {'LR': np.zeros(len(label)),
                       'RF': np.zeros(len(label)),
                       'XGB': np.zeros(len(label)),
                       'LGB': np.zeros(len(label)),
                       'CTB': np.zeros(len(label))}
        model_list = [lr_classifier, rf_classifier, xgb_classifier,
                      lgb_classifier, ctb_classifier]
        model_name = ['LR', 'RF', 'XGB', 'LGB', 'CTB']

    for (train_idx, val_idx) in kf.split(feature, label):
        print('{}-Fold Cross Validation: Fold {}'.format(args.nfold, fold))
        if args.mode == 'gnn':
            feature = torch.tensor(feature).float().to(device)
            label = torch.tensor(label).float().to(device)
            train_label, val_label = label[train_idx], label[val_idx]
            model = Model(feature.shape[1],
                          hidden_feats=args.hidden_feats,
                          num_heads=args.num_heads,
                          dp=args.dropout,
                          sample=args.sampling)
            model.to(device)
            print(model)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay)

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(torch.where(train_label == 0)[0]) /
                                                                           len(torch.where(train_label == 1)[0])))
            print('BCE loss pos weight: {:.3f}'.format(
                len(torch.where(train_label == 0)[0]) / len(torch.where(train_label == 1)[0])))

            print_list = []

            if args.sampling == 'no_sp':
                g = g.to(device)
                for epoch in range(1, args.epoch + 1):
                    model.train()
                    pred, attn_gat = model(feature, g)
                    pred = pred.squeeze(dim=1)
                    score = torch.sigmoid(pred)
                    optimizer.zero_grad()
                    loss = criterion(pred[train_idx], train_label)
                    loss.backward()
                    optimizer.step()
                    AUC_train, AUPR_train = get_metrics_auc(train_label.cpu().detach().numpy(),
                                                            score[train_idx].cpu().detach().numpy())

                    if epoch % args.print_every == 0:
                        # model.eval()
                        # pred_, attn_, attn_gat_, g_ = model(train_data, sims, args.k)
                        # score_ = torch.sigmoid(pred_)
                        AUC_val, AUPR_val = get_metrics_auc(val_label.cpu().detach().numpy(),
                                                            score[val_idx].cpu().detach().numpy())
                        print('Epoch {} Loss: {:.5f}; Train: AUC {:.3f}, AUPR {:.3f};'
                              ' Val: AUC {:.3f}, AUPR {:.3f}'.format(epoch, loss.item(), AUC_train,
                                                                     AUPR_train, AUC_val, AUPR_val))
                    print_list.append([epoch, loss.item(), AUC_train, AUPR_train, AUC_val, AUPR_val])
                    m = checkpoint(args, model, print_list, [loss, AUC_train, AUPR_train], fold)
                    if m:
                        best_model = m

                best_model.eval()
                pred_, attn_gat_ = best_model(feature, g)
                pred_ = pred_.squeeze(dim=1)
                score_ = torch.sigmoid(pred_).cpu().detach().numpy()
                AUC_val, AUPR_val = get_metrics_auc(val_label.cpu().detach().numpy(),
                                                    score_[val_idx])
                pred_result[val_idx] = score_[val_idx]

            else:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
                train_loader = dgl.dataloading.NodeDataLoader(g, train_idx, sampler,
                                                              batch_size=args.batch_size,
                                                              shuffle=True, drop_last=False)
                val_loader = dgl.dataloading.NodeDataLoader(g, val_idx, sampler,
                                                            batch_size=args.batch_size,
                                                            shuffle=True, drop_last=False)
                for epoch in range(1, args.epoch + 1):
                    model.train()
                    labels, preds, losses = [], [], []
                    for input_nodes, output_nodes, blocks in train_loader:
                        # print(f'input node:{input_nodes.shape}')
                        # print(f'output node:{output_nodes.shape}')
                        blocks = [b.to(device) for b in blocks]
                        train_feat, train_label = feature[input_nodes], label[output_nodes]
                        # print(f'label:{train_label.shape}')
                        pred, attn_gat = model(train_feat, blocks)
                        pred = pred.squeeze(dim=1)
                        score = torch.sigmoid(pred)
                        # print(pred.shape)
                        optimizer.zero_grad()
                        loss = criterion(pred, train_label)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                        labels.extend(train_label.detach().cpu().numpy().tolist())
                        preds.extend(score.detach().cpu().numpy().tolist())
                    AUC_train, AUPR_train = get_metrics_auc(np.array(labels),
                                                            np.array(preds))
                    if epoch % args.print_every == 0:
                        val_labels, val_preds = [], []
                        model.eval()
                        for input_nodes, output_nodes, blocks in val_loader:
                            blocks = [b.to(device) for b in blocks]
                            val_feat, val_label = feature[input_nodes], label[output_nodes]
                            pred, attn_gat = model(val_feat, blocks)
                            pred = pred.squeeze(dim=1)
                            score = torch.sigmoid(pred)
                            val_labels.extend(val_label.detach().cpu().numpy().tolist())
                            val_preds.extend(score.detach().cpu().numpy().tolist())
                        AUC_val, AUPR_val = get_metrics_auc(np.array(val_labels),
                                                            np.array(val_preds))
                        print('Epoch {} Loss: {:.5f}; Train: AUC {:.3f}, AUPR {:.3f};'
                              ' Val: AUC {:.3f}, AUPR {:.3f}'.format(epoch, sum(losses) / len(losses), AUC_train,
                                                                     AUPR_train, AUC_val, AUPR_val))
                    print_list.append([epoch, sum(losses) / len(losses), AUC_train, AUPR_train, AUC_val, AUPR_val])
                    m = checkpoint(args, model, print_list, [sum(losses) / len(losses), AUC_train, AUPR_train], fold)
                    if m:
                        best_model = m
                best_model.eval()
                idx_list, pred_list, label_list, attn_list = [], [], [], []
                for input_nodes, output_nodes, blocks in val_loader:
                    blocks = [b.to(device) for b in blocks]
                    val_feat, val_label = feature[input_nodes], label[output_nodes]
                    pred, attn_gat = model(val_feat, blocks)
                    pred = pred.squeeze(dim=1)
                    score = torch.sigmoid(pred)
                    idx_list.extend(output_nodes.detach().cpu().numpy().tolist())
                    pred_list.extend(score.detach().cpu().numpy().tolist())
                    label_list.extend(val_label.detach().cpu().numpy().tolist())
                    attn_list.extend(attn_gat.detach().cpu().numpy().tolist())
                AUC, AUPR, Acc, F1, Pre, Rec, Spec = get_metrics(np.array(label_list),
                                                                 np.array(pred_list))
                pred_result[idx_list] = np.array(pred_list)
            plot_curve(args, print_list, fold)

        elif args.mode == 'automl':
            label_name = data.columns[0]
            train_data, val_data = data.iloc[train_idx, :], data.iloc[val_idx, 1:]
            model = TabularPredictor(label=label_name, path=args.saved_path).fit(train_data)
            val_pred = np.array(model.predict_proba(val_data))[:, 1].flatten()
            AUC, AUPR, Acc, F1, Pre, Rec, Spec = get_metrics(label[val_idx], val_pred)
            print('Fold {}: AUC {:.3f}, AUPR: {:.3f}, Accuracy: {:.3f},'
                  ' F1 {:.3f}, Precision {:.3f}, Recall {:.3f}, Specificity {:.3f}'.format(
                fold, AUC, AUPR, Acc, F1, Pre, Rec, Spec))
            pred_result[val_idx] = val_pred

        elif args.mode == 'ml':
            train_data, val_data = feature[train_idx], feature[val_idx]
            train_label, val_label = label[train_idx], label[val_idx]
            train_data, train_label = train_resample(train_data, train_label, args.seed)

            for model_func, name in zip(model_list, model_name):
                model = model_func(train_data, train_label, args.seed)
                val_pred = model.predict_proba(val_data)[:, 1]
                AUC, AUPR, Acc, F1, Pre, Rec, Spec = get_metrics(val_label, val_pred)
                print('Model {} in Fold {}: AUC {:.3f}, AUPR: {:.3f}, Accuracy: {:.3f},'
                      ' F1 {:.3f}, Precision {:.3f}, Recall {:.3f}, Specificity {:.3f}'.format(
                    name, fold, AUC, AUPR, Acc, F1, Pre, Rec, Spec))
                pred_result[name][val_idx] = val_pred
                metric_list[name].append([AUC, AUPR, Acc, F1, Pre, Rec, Spec])

        if args.mode in ['gnn', 'automl']:
            metric_list.append([AUC, AUPR, Acc, F1, Pre, Rec, Spec])
            print('Overall: AUC {:.3f}, AUPR: {:.3f}, Accuracy: {:.3f},'
                  ' F1 {:.3f}, Precision {:.3f}, Recall {:.3f}, Specificity {:.3f}'.format(
                AUC, AUPR, Acc, F1, Pre, Rec, Spec))
            pd.DataFrame(np.array(metric_list),
                         columns=['AUC', 'AUPR', 'Acc', 'F1', 'Pre', 'Rec', 'Spec']). \
                to_csv(os.path.join(args.saved_path, 'results.csv'), index=False)
            pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, 'predictions.csv'),
                                             index=False, header=False)
        elif args.mode in ['ml']:
            for model in model_name:
                pd.DataFrame(np.array(metric_list[model]),
                             columns=['AUC', 'AUPR', 'Acc', 'F1', 'Pre', 'Rec', 'Spec']). \
                    to_csv(os.path.join(args.saved_path, f'{model}_results.csv'), index=False)
                pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, f'{model}_predictions.csv'),
                                                 index=False, header=False)
        fold += 1


if __name__ == '__main__':
    train()
