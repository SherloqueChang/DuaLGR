import argparse
import os.path

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from utils import load_data, normalize_weight, cal_homo_ratio, get_n_clusters, pca_weights
from models import EnDecoder, DuaLGR, GNN
from evaluation import eva, eva_real
from settings import get_settings
import matplotlib.pyplot as plt
from visulization import plot_loss, plot_tsne
import pandas as pd



    
# python DuaLGR.py --dataset dblp --train True --use_cuda True --cuda_device 0

files_syn = [
    
    'PolarDB-for-PostgreSQL-2_src_include_utils_merge_two_h',
    'PolarDB-for-PostgreSQL-4_src_include_utils_merge_four_h',
    'PolarDB-for-PostgreSQL-6_src_include_utils_merge_six_h',
    'PolarDB-for-PostgreSQL-8_src_include_utils_merge_eight_h',
    
    'FreeRDP-2_include_freerdp_merge_h',
    'FreeRDP-4_include_freerdp_merge_h',
    'FreeRDP-6_include_freerdp_merge_h',
    'FreeRDP-8_include_freerdp_merge_h',
    'fontforge-2_fontforge_merge_h',
    'fontforge-4_fontforge_merge_h',
    'fontforge-6_fontforge_merge_h',
    'fontforge-8_fontforge_merge_h'
]

files_real = [
    'FreeRDP_include_freerdp_settings_h',
    'PolarDB-for-PostgreSQL_src_include_utils_guc_h',
    'SDL_src_dynapi_SDL_dynapi_overrides_h',
    'SoftEtherVPN_src_Mayaqua_Network_h',
    'stress-ng_stress-ng_h',
    'wiredtiger_src_include_extern_h'
]

for file in files_syn:

    is_real = False
    # for n_clusters in [4,5,6,7,8,9,10]:
    n_clusters = get_n_clusters(file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='acm', help='datasets: acm, dblp, texas, chameleon, acm00, acm01, acm02, acm03, acm04, acm05')
    parser.add_argument('--train', type=bool, default=False, help='training mode')
    parser.add_argument('--cuda_device', type=int, default=0, help='')
    parser.add_argument('--use_cuda', type=bool, default=True, help='')
    parser.add_argument('--file', type=str, default='fontforge-6_fontforge_merge_h', help='')
    parser.add_argument('--model_name', type=str, default='DuaLGR_dblp', help='model_name')
    parser.add_argument('--path', type=str, default='./data/', help='The path of datasets')
    parser.add_argument('--weight_soft', type=int, default=3, help='smooth-sharp paramter')
    parser.add_argument('--alpha', type=int, default=5, help='alpha')
    parser.add_argument('--quantize', type=float, default=0.8, help='quantize Omega')
    parser.add_argument('--varepsilon', type=float, default=0.5, help='varepsilon')
    parser.add_argument('--endecoder_hidden_dim', type=int, default=512, help='endecoder_hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=512, help='latent_dim')
    parser.add_argument('--pretrain', type=int, default=500, help='pretrain epochs')
    parser.add_argument('--epoch', type=int, default=1000, help='')
    parser.add_argument('--patience', type=int, default=500, help='')
    parser.add_argument('--endecoder_lr', type=float, default=1e-4, help='learning rate for autoencoder')
    parser.add_argument('--endecoder_weight_decay', type=float, default=5e-6, help='weight decay for autoencoder')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for DuaLGR')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for DuaLGR')
    parser.add_argument('--update_interval', type=int, default=10, help='')
    parser.add_argument('--random_seed', type=int, default=2023, help='')
    parser.add_argument('--n_clusters', type=int, default=n_clusters, help='number of clusters')
    args = parser.parse_args()

    dataset = args.dataset
    train = args.train
    cuda_device = args.cuda_device
    use_cuda = args.use_cuda

    # settings = get_settings(dataset)

    path = args.path
    weight_soft = args.weight_soft
    alpha = args.alpha
    quantize = args.quantize
    varepsilon = args.varepsilon
    endecoder_hidden_dim = args.endecoder_hidden_dim
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    pretrain = args.pretrain
    epoch = args.epoch
    patience = args.patience
    endecoder_lr = args.endecoder_lr
    endecoder_weight_decay = args.endecoder_weight_decay
    lr = args.lr
    weight_decay = args.weight_decay
    update_interval = args.update_interval
    random_seed = args.random_seed
    torch.manual_seed(random_seed)


    # 修改这里的数据读取（参数选择 dblp），对应函数 load_multi()
    labels, adjs_labels, shared_feature, shared_feature_label, graph_num = load_data(dataset, path, file, is_real)

    # for v in range(graph_num):
    #     r = cal_homo_ratio(adjs_labels[v].cpu().numpy(), labels.cpu().numpy(), self_loop=True)
    #     print(r)
    print('nodes: {}'.format(shared_feature_label.shape[0]))
    print('features: {}'.format(shared_feature_label.shape[1]))
    print('class: {}'.format(labels.max() + 1))

    feat_dim = shared_feature.shape[1]
    if is_real == False:
        class_num = labels.max() + 1
    else:
        class_num = args.n_clusters
    y = labels.cpu().numpy()

    endecoder = EnDecoder(feat_dim, endecoder_hidden_dim, class_num)
    model = DuaLGR(feat_dim, hidden_dim, latent_dim, endecoder, class_num=class_num, num_view=graph_num)

    if use_cuda:
        torch.cuda.set_device(cuda_device)
        torch.cuda.manual_seed(random_seed)
        endecoder = endecoder.cuda()
        model = model.cuda()
        adjs_labels = [adj_labels.cuda() for adj_labels in adjs_labels]
        shared_feature = shared_feature.cuda()
        shared_feature_label = shared_feature_label.cuda()
    device = shared_feature.device

    if train:
        # =============================================== pretrain endecoder ============================
        print('shared_feature_label for clustering...')
        kmeans = KMeans(n_clusters=class_num, n_init=5)
        y_pred = kmeans.fit_predict(shared_feature_label.data.cpu().numpy())
        if is_real == False:
            eva(y, y_pred, 'Kz')
        else:
            eva_real(y_pred, file)
        print()

        optimizer_endecoder = Adam(endecoder.parameters(), lr=endecoder_lr, weight_decay=endecoder_weight_decay)

        for epoch_num in range(pretrain):
            endecoder.train()
            loss_re = 0.
            loss_a = 0.

            a_pred, x_pred, z_norm = endecoder(shared_feature)
            for v in range(graph_num):
                loss_a += F.binary_cross_entropy(a_pred, adjs_labels[v])
            loss_re += F.binary_cross_entropy(x_pred, shared_feature_label)

            loss = loss_re + loss_a
            optimizer_endecoder.zero_grad()
            loss.backward()
            optimizer_endecoder.step()
            # print('epoch: {}, loss:{}, loss_re:{}, loss_a: {}'.format(epoch_num, loss, loss_re, loss_a))

            if epoch_num == pretrain - 1:
                print('Pretrain complete...')
                kmeans = KMeans(n_clusters=class_num, n_init=5)
                y_pred = kmeans.fit_predict(z_norm.data.cpu().numpy())
                if is_real == False:
                    eva(y, y_pred, 'Kz')
                else:
                    eva_real(y_pred, file)
                break


        # =========================================Train=============================================================
        print('Begin trains...')
        param_all = []
        for v in range(graph_num+1):
            param_all.append({'params': model.cluster_layer[v]})
        param_all.append({'params': model.gnn.parameters()})
        optimizer_model = Adam(param_all, lr=lr, weight_decay=weight_decay)

        best_a = [1e-12 for i in range(graph_num)]
        weights = normalize_weight(best_a)
        weights = np.load('../data/weights/' + file + '.npy', allow_pickle=True)

        with torch.no_grad():
            model.eval()
            pseudo_label = y_pred
            a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha, quantize=quantize, varepsilon=varepsilon)
            kmeans = KMeans(n_clusters=class_num, n_init=5)
            for v in range(graph_num+1):
                y_pred = kmeans.fit_predict(z_all[v].data.cpu().numpy())
                model.cluster_layer[v].data = torch.tensor(kmeans.cluster_centers_).to(device)
                # eva(y, y_pred, 'K{}'.format(v))
            pseudo_label = y_pred

        bad_count = 0
        best_acc = 1e-12
        best_nmi = 1e-12
        best_ari = 1e-12
        best_f1 = 1e-12
        best_epoch = 0
        best_modularity = 0
        besrt_result = []

        nmi_list = []
        acc_list = []
        loss_list = []
        modularity_list = []
        for epoch_num in range(epoch):
            model.train()

            loss_re = 0.
            loss_kl = 0.
            loss_re_a = 0.
            loss_re_ax = 0.

            a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha, quantize=quantize, varepsilon=varepsilon)
            for v in range(graph_num):
                loss_re_a += F.binary_cross_entropy(a_pred, adjs_labels[v])
            loss_re_x = F.binary_cross_entropy(x_pred, shared_feature_label)
            loss_re += loss_re_a + loss_re_x

            kmeans = KMeans(n_clusters=class_num, n_init=5)
            y_prim = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
            pseudo_label = y_prim
            views = []

            for v in range(graph_num):
                y_pred = kmeans.fit_predict(z_all[v].detach().cpu().numpy())
                views.append(z_all[v].detach().cpu().numpy())
                a = eva(y_prim, y_pred, visible=False, metrics='nmi')
                best_a[v] = a

            # weights = normalize_weight(best_a, p=weight_soft)
            weights = pca_weights(views)
            # print(weights)


            p = model.target_distribution(q_all[-1])
            for v in range(graph_num):
                loss_kl += F.kl_div(q_all[v].log(), p, reduction='batchmean')
            loss_kl += F.kl_div(q_all[-1].log(), p, reduction='batchmean')

            loss = loss_re + loss_kl
            loss_list.append(loss.item())
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

            # print('epoch: {}, loss: {}, loss_re: {}, loss_kl:{}, badcount: {}, loss_re_a: {}, loss_re_x: {}'. format(epoch_num, loss, loss_re, loss_kl, bad_count, loss_re_a, loss_re_x))

        # =========================================evaluation=============================================================
            if epoch_num % update_interval == 0:
                model.eval()
                with torch.no_grad():
                    a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha, quantize=quantize, varepsilon=varepsilon)
                    kmeans = KMeans(n_clusters=class_num, n_init=5)
                    y_eval = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
                    if is_real == False:
                        nmi, acc, ari, f1 = eva(y, y_eval, str(epoch_num) + 'Kz')
                        nmi_list.append(nmi.item())
                        acc_list.append(acc.item())
                    else:
                        nmi, acc, ari, f1 = 0, 0, 0, 0
                        modularity = eva_real(y_pred, file)
                        modularity_list.append(modularity.item())
            if is_real == False:
                if acc > best_acc:
                    if os.path.exists('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc)):
                        os.remove('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
                    best_epoch = epoch_num
                    bad_count = 0
                    best_result = y_eval
                    torch.save({'state_dict': model.state_dict(),
                                'state_dict_endecoder': endecoder.state_dict(),
                                'weights': weights,
                                'pseudo_label': pseudo_label},
                            './pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
                    print('best acc:{}, best nmi:{}, best ari:{}, best f1:{}, bestepoch:{}'.format(
                                                best_acc, best_nmi, best_ari, best_f1, best_epoch))
                else:
                    bad_count += 1
            else:
                if modularity > best_modularity:
                    if os.path.exists('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, modularity)):
                        os.remove('./pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, modularity))
                    best_modularity = modularity
                    best_epoch = epoch_num
                    bad_count = 0
                    best_result = y_eval
                    torch.save({'state_dict': model.state_dict(),
                                'state_dict_endecoder': endecoder.state_dict(),
                                'weights': weights,
                                'pseudo_label': pseudo_label},
                            './pkl/dualgr_{}_acc{:.4f}.pkl'.format(dataset, best_modularity))
                    print('best modularity:{}'.format(
                                                best_modularity))
                else:
                    bad_count += 1

            if bad_count >= patience:
                if is_real == False:
                    print('complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{}, bestepoch:{}'.format(
                    best_acc, best_nmi, best_ari, best_f1, best_epoch))
                else:
                    print('complete training, best modularity:{}, bestepoch:{}'.format(best_modularity, best_epoch))
                print()
                break

    if is_real == False:
        model_name = 'dualgr_{}_acc{:.4f}'.format(dataset, best_acc)
    else:
        model_name = 'dualgr_{}_acc{:.4f}'.format(dataset, best_modularity)

    best_model = torch.load('./pkl/'+model_name+'.pkl', map_location=shared_feature.device)
    state_dic = best_model['state_dict']
    state_dic_encoder = best_model['state_dict_endecoder']
    weights = best_model['weights']
    pseudo_label = best_model['pseudo_label']

    endecoder.load_state_dict(state_dic_encoder)
    model.load_state_dict(state_dic)

    model.eval()
    with torch.no_grad():
        model.endecoder = endecoder
        a_pred, x_pred, z_all, q_all, a_pred_x, x_pred_x = model(shared_feature, adjs_labels, weights, pseudo_label, alpha,quantize=quantize, varepsilon=varepsilon)
        kmeans = KMeans(n_clusters=class_num, n_init=5)
        y_eval = kmeans.fit_predict(z_all[-1].detach().cpu().numpy())
        if is_real == False:
            nmi, acc, ari, f1 = eva(y, y_eval, 'Final Kz')
            with open('./result/results_test.txt', 'a') as f:
                f.write(str(file) + ', {:.4f}'.format(acc) + ', {:.4f}'.format(f1) + \
                    ', {:.4f}'.format(ari) + ', {:.4f}'.format(nmi) + '\n')
            with open('./result/test/' + file + '.txt', 'w') as f:
                for item in best_result:
                    f.write(str(item) + ' ')
        else:
            modularity = eva_real(y_eval, file)
            with open('./result/results_real.txt', 'a') as f:
                f.write(str(file) + ', ' + str(args.n_clusters) +', {:.4f}'.format(modularity) + '\n')
            with open('./result/' + file + '-' + str(args.n_clusters) + '.txt', 'w') as f:
                for item in best_result:
                    f.write(str(item) + ' ')
                    

    print('Test complete...')
