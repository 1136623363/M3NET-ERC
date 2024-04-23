import torch


def jvbu(nodes,windows,dia_len=0):
    index=[]
    if len(nodes) > windows:
        for i in range(0,windows):
            if i+windows+1 > len(nodes):
                continue
            index.append(nodes[:i+windows+1])

        for i in range(windows,len(nodes)-windows):
            index.append(nodes[i-windows:i+windows+1])

        for i in range(len(nodes)-windows,len(nodes)):
            if i < windows:
                continue
            index.append(nodes[i-windows:])
    return index

def jvbu2(nodes,windows,step,dia_len=0):
    index=[]
    i=windows
    if len(nodes) > windows:
        while i+windows<=len(nodes):
            index.append(nodes[i-windows:i+windows+1])
            i+=step
    return index

def single_people(nodes,windows,spk_idx,step,mode,dia_len=0):
    # index0 = []
    # index1 = []
    num_people = len(set(spk_idx))
    index={}
    for i in range(num_people):
        index.update({f"index_{str(i)}":[]})
        exec(f"index{i}=[]")

    for i,j in zip(nodes,spk_idx):
        # if j == 0:
        #     index0.append(i)
        # elif j == 1:
        #     index1.append(i)
        for k in range(num_people):
            if j == k:
                exec(f"index['index_{k}'].append(i)")
    index_all = []

    if mode == "local":
        for i in range(num_people):
            exec(f"index{i}=jvbu2(index['index_{i}'],windows,step)")
            exec(f"index_all+=index{i}")
        # index_0 = jvbu2(index0,windows,step)
        # index_1 = jvbu2(index1,windows,step)
        # index_all=[]
        # for i in index.values():
        #     index_all+=i
    elif mode == "remote":
        for i in index.values():
            index_all.append(i)

    return index_all
# print(single_people(nodes,8,spk_idx))
# print(creat_hyper_index(nodes,30))




def create_hyper_index_label( a, v, l, dia_len, modals, start, label):
    """
    全部说话者局部超边
    """
    self_loop = False
    num_modality = len(modals)

    edge_type = torch.zeros(0).cuda()
    in_index0 = torch.zeros(0).cuda()

    nodes_l_all = []
    nodes_a_all = []
    nodes_v_all = []
    num_modality = 3
    node_count = 0
    index0 = []
    index1 = []
    # spk_idx = spk_idx.cpu().detach().numpy()
    index_1 = []
    index_2 = []
    edge_count=0
    nodes_count_l=0
    nodes_count_a = 0
    nodes_count_v = 0
    # windows = 5
    spk_tmp = 0
    # step = 6
    mode = "remote"
    # num_label = set(label)
    index=[]
    label_count=0
    for i in dia_len:
        nodes = list(range(i * num_modality))
        nodes = [j + node_count for j in nodes]
        nodes_l = nodes[0:i * num_modality // 3]
        nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
        nodes_v = nodes[i * num_modality * 2 // 3:]



        index1.extend(single_people(nodes_l,None,label[spk_tmp:spk_tmp+i],None,mode))
        index1.extend(single_people(nodes_a, None, label[spk_tmp:spk_tmp+i],None,mode))
        index1.extend(single_people(nodes_v, None, label[spk_tmp:spk_tmp+i],None,mode))
        spk_tmp+=i

        node_count += i * num_modality
        label_count+=i

    for i in index1:
        index_2.extend([len(i) * [edge_count + start]])
        edge_count+=1
    index_3 = []
    for i in index_2:
        index_3.extend(i)
    for i in index1:
        index_1.extend(i)
    index_1 = torch.LongTensor(index_1).view(1, -1).cuda()
    index_3 = torch.LongTensor(index_3).view(1, -1).cuda()
    hyperedge_index = torch.cat([index_1, index_3], dim=0)
    return hyperedge_index

