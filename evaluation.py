import numpy as np
import networkx as nx
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from utils import calculate_modularity

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro





def eva(y_true, y_pred, epoch=0, visible=True, metrics='all'):
    # print(len(y_true), len(y_pred))
    if metrics == 'all':
        acc, f1 = cluster_acc(y_true, y_pred)
        nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
        ari = ari_score(y_true, y_pred)
        result = (nmi, acc, ari, f1)
    elif metrics == 'nmi':
        nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
        result = nmi
    elif metrics == 'acc':
        acc, f1 = cluster_acc(y_true,y_pred)
        result = acc
    # if visible:
        # print(epoch, ':acc {:.4f}'.format(acc), ', ari {:.4f}'.format(ari), ', nmi {:.4f}'.format(nmi), 
        #     ', f1 {:.4f}'.format(f1))
        # pass
    return result

def eva_real(y_pred, file):
    adj1 = np.load('D:/Document/研究生/research/graph clustering/data//usage/' + file + '.npy', allow_pickle=True)
    adj2 = np.load('D:/Document/研究生/research/graph clustering/data/semantic/' + file + '.npy', allow_pickle=True)
    adj3 = np.load('D:/Document/研究生/research/graph clustering/data/cdm/' + file + '.npy', allow_pickle=True)
    # adj1 = np.load('D:/Document/研究生/research/graph clustering/data//bundle/usage/' + file + '.npy', allow_pickle=True)
    # adj2 = np.load('D:/Document/研究生/research/graph clustering/data/bundle/semantic/' + file + '.npy', allow_pickle=True)
    
    adj = adj1 + adj2 + adj3
    # print(adj.shape[0], len(y_pred))
    
    graph = nx.Graph()
    for i in range(adj.shape[0]):
        graph.add_node(i)
        for j in range(i + 1, adj.shape[1]):
            if adj[i][j] != 0:
                graph.add_edge(i, j, weight=adj[i][j])
        
    communities = []
    for i in range(max(y_pred) + 1):
        communities.append([])
    for i, j in enumerate(y_pred):
        communities[j].append(i)
    
    modularity = nx.community.modularity(graph, communities)
    # print('modularity: ', modularity)
    return modularity