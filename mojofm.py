def hungarian(graph):
    # 初始化匈牙利树
    tree = {'V': set(), 'E': set()}
    # 寻找增广路径
    while True:
        # 找到一条增广路径
        path = find_augmenting_path(graph, tree)
        if len(path['E']) == 0:
            break
        # 更新匈牙利树
        tree['V'].update(path['V'])
        tree['E'].update(path['E'])
    # 返回最大匹配
    return tree


def find_augmenting_path(graph, tree):
    # 初始化增广路径
    path = {'V': set(), 'E': set()}
    # 寻找增广路径
    for v in graph['V']:
        if v not in tree['V']:
            # 寻找从v出发的增广路径
            path = find_augmenting_path_from(graph, tree, v)
            if len(path['E']) > 0:
                break
    # 返回增广路径
    return path


def find_augmenting_path_from(graph, tree, v):
    # 初始化增广路径
    path = {'V': set(), 'E': set()}
    # 寻找增广路径
    for e in graph['E']:
        if e[0] == v and e[1] not in tree['V']:
            # 找到一条增广路径
            path['V'].add(v)
            path['V'].add(e[1])
            path['E'].add(e)
            break
    # 返回增广路径
    return path

def mno(cluster_result,ground_truth):
    num_of_A=[] # how many nodes in every A[i]
    graph = {
        'V': set(),
        'E': set()
    }
    set_of_A,set_of_B=set(),set()
    A=dict()
    # struct of A: {0: {'max_num': 5, 'max_id': 1, 'sum': 7, 1: 5, 2: 2}, 1: {'max_num': 24, 'max_id': 2, 'sum': 24, 2: 24}}

    for key,value in cluster_result.items():
        a_id=value
        b_id=ground_truth[key]  #找到在B中对应的tag
        str_a_id="A"+str(a_id)
        str_b_id="B"+str(b_id)

        set_of_A.add(a_id)
        set_of_B.add(b_id)
        if a_id not in A:
            A[a_id]=dict()
            A[a_id]["max_num"]=0
            A[a_id]["sum"]=0
            A[a_id]["max_id"]=0

        if b_id not in A[a_id]:
            A[a_id][b_id]=1
        else:
            A[a_id][b_id]+=1

        if A[a_id][b_id]>=A[a_id]["max_num"]:
            # Update the attribution of A[a_id] to G
            A[a_id]["max_num"]=A[a_id][b_id]
            A[a_id]["max_id"] = b_id
        A[a_id]["sum"]+=1

        graph["V"].add(str_a_id)    # add set of nodes
        graph["V"].add(str_b_id)
        graph["E"].add((str_a_id,str_b_id)) # 添加双向便
        graph["E"].add((str_b_id, str_a_id))
    #print(A)

    # 针对max_id不唯一，需要从max_id_list找到最合适的max_id

    #print(graph)
    res = hungarian(graph)  # 用最大二分图匹配找到最大的|G|
    #print(res)
    for item in res["E"]:
        if item[0][0]=='A':
            a_id=int(item[0][1:])
            b_id=int(item[1][1:])
        else:
            a_id=int(item[1][1:])
            b_id=int(item[0][1:])
        A[a_id]["max_id"] = b_id
    #print(A)

    # 开始计算move和join
    move=0
    G={}    # struct of G: {1: [0], 2: [1]}
    join=0
    for key,value in A.items():
        move+=value["sum"]-value["max_num"]
        # The total number of nodes in A[i] minus the largest number of nodes of the same kind

        max_id=value["max_id"]
        num_of_A.append(value["sum"])
        if max_id in G:
            join+=1 # 我认为只需要合并数量最多的标识符号相同的，这里可能需要学姐帮我判断一下是否正确
        else:
            G[max_id]=list()
        G[max_id].append(key)
    #print(G)

    # 第一个参数是M+l-g，第二个参数是A[i]的数量
    return move+join,num_of_A

def max_mno(cluster_result,ground_truth,num_of_A):
    # 将A[i]降序排列
    num_of_A.sort()
    G=0
    for i in range(len(num_of_A)):
        if num_of_A[i]>G:
            G+=1

    # return n - l + l - g = n - g
    return len(cluster_result)-G

def MoJoFM(cluster_result, ground_truth):
    mno_A_B,num_of_A=mno(cluster_result,ground_truth)   #molecule
    max_mno_A_B=max_mno(cluster_result,ground_truth,num_of_A)   #denominator
    return (1.0-1.0*mno_A_B/max_mno_A_B)*100.0  #percentage
