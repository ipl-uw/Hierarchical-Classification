import torch
from IPython import embed
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import csv
import os
import pandas as pd

# hierarchy_dict = {'Skates':['Hard Snout Skates', 'Soft Snout Skates'],
#                   'Sharks':['Blue Shark','Spiny Dogfish Shark','Spotted Ratfish','Soupfin Shark'],
#                   'Roundfish':['Pacific Cod','Grenadier','Sablefish','Lingcod','Sculpin','Walleye Pollock'],
#                   'Flatfishes':['Kamchatka-Arrowtooth','Pacific Halibut'],
#                   'Rockfishes':['Thornyheads','Redbanded Rockfish','Shortraker-Rougheye-BlackSpotted Rockfish','Silvergray Rockfish','Canary Rockfish','Yelloweye Rockfish','Northern Rockfish','Quillback Rockfish'],
#                   'Invertebrates':['Anemones','Octopus','Coral','Starfish','Bivalvia','Snails','Sea Urchins','Sponges','Mollusca']}


#for sleeper shark model
hierarchy_dict = {'Skates':['Hard Snout Skates', 'Soft Snout Skates'],
                  'Sharks':['Blue Shark','Spiny Dogfish Shark','Spotted Ratfish','Soupfin Shark', 'Pacific Sleeper Sharks'],
                  'Roundfish':['Pacific Cod','Grenadier','Sablefish','Lingcod','Sculpin','Walleye Pollock'],
                  'Flatfishes':['Kamchatka-Arrowtooth','Pacific Halibut','Dover Sole','Flathead Sole'],
                  'Rockfishes':['Thornyheads','Redbanded Rockfish','Shortraker-Rougheye-BlackSpotted Rockfish','Silvergray Rockfish','Canary Rockfish','Yelloweye Rockfish','Northern Rockfish','Quillback Rockfish'],
                  'Invertebrates':['Anemones','Octopus','Coral','Starfish','Bivalvia','Snails','Sea Urchins','Sponges','Mollusca'],
                  'Bird':[],
                  'Bait':[],
                  'Fishing Gear':[],
                  'Misc_Other':[]}

level_1_names = list(hierarchy_dict.keys())
level_2_names = []
level_2_num = []

for key in list(hierarchy_dict.keys()):
    level_2_num.append(len(hierarchy_dict[key]))
    for species_name in hierarchy_dict[key]:
        level_2_names.append(species_name)



def find_id_in_31(level_1_id,level_2_id):
    id_in_31=0

    for i in range(level_1_id):
        id_in_31 +=len(hierarchy_dict[level_1_names[i]])

    id_in_31+=level_2_id


    return id_in_31






def get_level_2_name(level_1_target, level_2_target):
    idx = 0
    for j in range(level_1_target):  # level_1_target=3, 则 j = 0,1,2
        idx += level_2_num[j]
    name = level_2_names[idx + level_2_target]

    return name





def put_in_dict(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets):

    _, level_1_predicted_labels = torch.max(probas[0], 1)  # 所有sample的 level-1预测, dim=128

    for i, target in enumerate(targets):  #遍历每个sample
        ### level-1 计算准确率
        level_1_gt = target[0]
        name = level_1_names[level_1_gt]
        num_examples1[name] += 1

        if level_1_predicted_labels[i] == level_1_gt:
            tp1[name] += 1


        ### level-2 计算准确率,
        level_2_gt = target[1]
        if level_2_gt ==-1:  #-1代表没有gt 准确率不考虑在内
            continue
        else:
            _, level_2_predicted_labels = torch.max(probas[level_1_gt+1], 1)  #只是取出gt的那个head看看是不是最大的
            name = get_level_2_name(level_1_gt, level_2_gt)
            num_examples2[name] += 1

            if level_2_predicted_labels[i] == level_2_gt:   #根据gt，直接去看level 2 的head
                tp2[name] += 1

            if level_2_predicted_labels[i] == level_2_gt and level_1_predicted_labels[i] == level_1_gt:  #如果第一个对了，第二个也对了，才算对
                tp2_p1p2[name] += 1






    return tp1, num_examples1,tp2, num_examples2, tp2_p1p2



def put_in_dict_fast(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets):
    num_level_1 = len(tp1)

    targets_level_1 =targets[:,0]
    targets_level_2 =targets[:,1]
    probas_level_1 = probas[0]
    for level_1_id in range(num_level_1):  #遍历每一个level-1的 类别，例如 level_1_id = 0
        targets_level_1_species_idx = torch.where(targets_level_1==level_1_id)  #targets 为skates的所有sample 的Index

        if len(targets_level_1_species_idx[0])==0:
            continue

        targets_level_1_species = targets_level_1[targets_level_1_species_idx]  #targets 为skates的所有sample
        probas_level_1_species = probas_level_1[targets_level_1_species_idx]    #targets 为skates的所有sample 的预测

        _, predicted_labels_level_1_species = torch.max(probas_level_1_species, 1)  #targets 为skates的所有sample 的预测

        name_1 = level_1_names[level_1_id]                                         #skates
        num_examples1[name_1] += targets_level_1_species.size(0)
        tp1[name_1] += (predicted_labels_level_1_species == targets_level_1_species).sum()



        targets_level_2_species = targets_level_2[targets_level_1_species_idx]   #还是这些sample 只不过找到target 和 probas
        probas_level_2 = probas[level_1_id + 1]
        probas_level_2_species = probas_level_2[targets_level_1_species_idx]

        num_current_level_2 = probas_level_2.size(1)    #Skates 下有2个类别

        for level_2_id in range(num_current_level_2):  #遍历每一个 level-2的类别
            targets_level_2_sp_index = torch.where(targets_level_2_species==level_2_id)  #自动把-1的level 2 label筛了

            if len(targets_level_2_sp_index[0]) == 0:
                continue

            targets_level_1_sp = targets_level_1_species[targets_level_2_sp_index]
            probas_level_1_sp = probas_level_1_species[targets_level_2_sp_index]
            targets_level_2_sp = targets_level_2_species[targets_level_2_sp_index]
            probas_level_2_sp = probas_level_2_species[targets_level_2_sp_index]

            _, predicted_labels_level_2_sp = torch.max(probas_level_2_sp, 1)  # 返回的是下标 从0开始

            index_ = find_previous_len(level_1_id, hierarchy_dict)
            name = level_2_names[level_2_id + index_]
            num_examples2[name] += targets_level_2_sp.size(0)
            tp2[name] += (predicted_labels_level_2_sp == targets_level_2_sp).sum()


            _, predicted_labels_level_1_sp = torch.max(probas_level_1_sp, 1)  # targets 为skates的所有sample 的预测
            # embed()
            correct_ind = torch.where(predicted_labels_level_1_sp == targets_level_1_sp)
            tp2_p1p2[name] += (predicted_labels_level_2_sp[correct_ind] == targets_level_2_sp[correct_ind]).sum()


        # index_ += len(hierarchy_dict[name_1])



    return tp1, num_examples1,tp2, num_examples2, tp2_p1p2



def multiply_7_head(probas_level_1, probas):

    probas_level_2_31 = torch.zeros((probas_level_1.shape[0],31))
    idx=0
    for i, level_2_head in enumerate(probas):
        if i==0:
            continue


        probas_level_2_31[:,idx:idx+level_2_head.shape[1]] = probas_level_1[:,i-1].unsqueeze(dim=1) * level_2_head
        idx += level_2_head.shape[1]


    return probas_level_2_31.to('cuda:0')

def put_in_dict_fast_model4(tp1, num_examples1, tp2, num_examples2, tp2_p1p2_maxmax, tp2_p1p2_31,probas, targets):
    num_level_1 = len(tp1)

    targets_level_1 =targets[:,0]
    targets_level_2 =targets[:,1]
    probas_level_1 = probas[0]


    probas_level_2_31 = multiply_7_head(probas_level_1, probas)

    for level_1_id in range(num_level_1):  #遍历每一个level-1的 类别，例如 level_1_id = 0
        targets_level_1_species_idx = torch.where(targets_level_1==level_1_id)  #targets 为skates的所有sample 的Index

        if len(targets_level_1_species_idx[0])==0:
            continue

        targets_level_1_species = targets_level_1[targets_level_1_species_idx]  #targets 为skates的所有sample
        probas_level_1_species = probas_level_1[targets_level_1_species_idx]    #targets 为skates的所有sample 的预测

        _, predicted_labels_level_1_species = torch.max(probas_level_1_species, 1)  #targets 为skates的所有sample 的预测

        name_1 = level_1_names[level_1_id]                                         #skates
        num_examples1[name_1] += targets_level_1_species.size(0)
        tp1[name_1] += (predicted_labels_level_1_species == targets_level_1_species).sum()



        targets_level_2_species = targets_level_2[targets_level_1_species_idx]   #还是这些sample 只不过找到target 和 probas
        probas_level_2 = probas[level_1_id + 1]
        probas_level_2_species = probas_level_2[targets_level_1_species_idx]

        probas_level_2_31_species=probas_level_2_31[targets_level_1_species_idx]

        num_current_level_2 = probas_level_2.size(1)    #Skates 下有2个类别

        for level_2_id in range(num_current_level_2):  #遍历每一个 level-2的类别
            targets_level_2_sp_index = torch.where(targets_level_2_species==level_2_id)  #自动把-1的level 2 label筛了

            if len(targets_level_2_sp_index[0]) == 0:
                continue

            targets_level_1_sp = targets_level_1_species[targets_level_2_sp_index]
            probas_level_1_sp = probas_level_1_species[targets_level_2_sp_index]
            targets_level_2_sp = targets_level_2_species[targets_level_2_sp_index]
            probas_level_2_sp = probas_level_2_species[targets_level_2_sp_index]

            _, predicted_labels_level_2_sp = torch.max(probas_level_2_sp, 1)  # 返回的是下标 从0开始

            index_ = find_previous_len(level_1_id, hierarchy_dict)
            name_2 = level_2_names[level_2_id + index_]
            num_examples2[name_2] += targets_level_2_sp.size(0)
            tp2[name_2] += (predicted_labels_level_2_sp == targets_level_2_sp).sum()


            _, predicted_labels_level_1_sp = torch.max(probas_level_1_sp, 1)  # targets 为skates的所有sample 的预测

            correct_ind = torch.where(predicted_labels_level_1_sp == targets_level_1_sp)
            tp2_p1p2_maxmax[name_2] += (predicted_labels_level_2_sp[correct_ind] == targets_level_2_sp[correct_ind]).sum()


            #开始计算  max out of 31
            id_in_31 = find_id_in_31(level_1_id, level_2_id)
            probas_level_2_31_sp = probas_level_2_31_species[targets_level_2_sp_index]
            _, predicted_probas_level2_31_sp = torch.max(probas_level_2_31_sp, 1)
            # embed()
            tp2_p1p2_31[name_2] += (id_in_31 == predicted_probas_level2_31_sp).sum()








    return tp1, num_examples1,tp2, num_examples2, tp2_p1p2_maxmax, tp2_p1p2_31

def put_in_dict_fast_model7(tp1, num_examples1, tp2, num_examples2, tp2_p1p2_31, tp2_p1p2_maxmax, probas, probas_level2,targets_31, targets_split):
    num_level_1 = len(tp1)

    targets_level_1 =targets_31[:,0]
    targets_level_2_31 =targets_31[:,1]
    targets_level_2_split = targets_split[:, 1]


    probas_level_1 = probas[0]
    for level_1_id in range(num_level_1):  #遍历每一个level-1的 类别，例如 level_1_id = 0
        targets_level_1_species_idx = torch.where(targets_level_1==level_1_id)  #targets 为skates的所有sample 的Index

        if len(targets_level_1_species_idx[0])==0:
            continue

        targets_level_1_species = targets_level_1[targets_level_1_species_idx]  #targets 为skates的所有sample
        probas_level_1_species = probas_level_1[targets_level_1_species_idx]    #targets 为skates的所有sample 的预测

        _, predicted_labels_level_1_species = torch.max(probas_level_1_species, 1)  #targets 为skates的所有sample 的预测

        name_1 = level_1_names[level_1_id]                                         #skates
        num_examples1[name_1] += targets_level_1_species.size(0)
        tp1[name_1] += (predicted_labels_level_1_species == targets_level_1_species).sum()




        #计算单独level-2 Head的准确率，用  targets_level_2_split， probas[level_1_id + 1]
        if level_1_id>5: # last groups such as birds, do ont have level2 species
            continue
        targets_level_2_species = targets_level_2_split[targets_level_1_species_idx]   #还是这些sample 只不过找到target 和 probas
        probas_level_2 = probas[level_1_id + 1]
        probas_level_2_species = probas_level_2[targets_level_1_species_idx]

        targets_level_2_species_31 = targets_level_2_31[targets_level_1_species_idx]
        probas_level2_31 = probas_level2[targets_level_1_species_idx]


        num_current_level_2 = probas_level_2.size(1)    #Skates 下有2个类别

        for level_2_id in range(num_current_level_2):  #遍历每一个 level-2的类别
            targets_level_2_sp_index = torch.where(targets_level_2_species==level_2_id)  #自动把-1的level 2 label筛了

            if len(targets_level_2_sp_index[0]) == 0:
                continue

            targets_level_1_sp = targets_level_1_species[targets_level_2_sp_index]
            probas_level_1_sp = probas_level_1_species[targets_level_2_sp_index]
            targets_level_2_sp = targets_level_2_species[targets_level_2_sp_index]
            probas_level_2_sp = probas_level_2_species[targets_level_2_sp_index]

            _, predicted_labels_level_2_sp = torch.max(probas_level_2_sp, 1)  # 返回的是下标 从0开始

            index_ = find_previous_len(level_1_id, hierarchy_dict)
            name_2 = level_2_names[level_2_id + index_]
            num_examples2[name_2] += targets_level_2_sp.size(0)
            tp2[name_2] += (predicted_labels_level_2_sp == targets_level_2_sp).sum()


            # model 12 用的计算level1 and level2同时对，同时是最大的
            _, predicted_labels_level_1_sp = torch.max(probas_level_1_sp, 1)  # targets 为skates的所有sample 的预测
            # embed()
            correct_ind = torch.where(predicted_labels_level_1_sp == targets_level_1_sp)
            tp2_p1p2_maxmax[name_2] += (predicted_labels_level_2_sp[correct_ind] == targets_level_2_sp[correct_ind]).sum()




            # model 7计算31个output里面对不对，即31里取最大，而不是level-1 max 且level-2max
            id_in_31=find_id_in_31(level_1_id,level_2_id)  #根据当前level-1 id   level—2 id转换成  0-31的数字
            targets_level_2_sp_index_31  = torch.where(targets_level_2_species_31==id_in_31)  #在0-31的targets里面查找
            targets_level_2_sp_31 = targets_level_2_species_31[targets_level_2_sp_index_31]
            probas_level2_31_sp = probas_level2_31[targets_level_2_sp_index_31]                #取出对应的预测

            _, predicted_probas_level2_31_sp = torch.max(probas_level2_31_sp, 1)
            tp2_p1p2_31[name_2] +=(targets_level_2_sp_31 == predicted_probas_level2_31_sp).sum() #计算31个outputs里，最大的那个是不是对的




    return tp1, num_examples1,tp2, num_examples2, tp2_p1p2_31, tp2_p1p2_maxmax




def find_previous_len(level_1_id, hierarchy_dict):

    if level_1_id==0:
        return 0

    index_=0

    keys = list(hierarchy_dict.keys())
    for i in range(level_1_id):  #0,1, level_1_id-1
        index_+=len(hierarchy_dict[keys[i]])


    return index_

def put_in_dict_fast_model3(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas31,probas32, targets):

    num_level_1 = len(tp1)

    targets_level_1 =targets[:,0]
    targets_level_2 =targets[:,1]
    probas_level_1 = probas31


    for level_1_id in range(num_level_1):  #遍历每一个level-1的 类别，例如 level_1_id = 0
        targets_level_1_species_idx = torch.where(targets_level_1==level_1_id)  #targets 为skates的所有sample 的Index

        if len(targets_level_1_species_idx[0])==0:
            continue

        targets_level_1_species = targets_level_1[targets_level_1_species_idx]  #targets 为skates的所有sample
        probas_level_1_species = probas_level_1[targets_level_1_species_idx]    #targets 为skates的所有sample 的预测

        _, predicted_labels_level_1_species = torch.max(probas_level_1_species, 1)  #targets 为skates的所有sample 的预测

        name_1 = level_1_names[level_1_id]                                         #skates
        num_examples1[name_1] += targets_level_1_species.size(0)
        tp1[name_1] += (predicted_labels_level_1_species == targets_level_1_species).sum()



        targets_level_2_species = targets_level_2[targets_level_1_species_idx]   #还是这些sample 只不过找到target 和 probas
        probas_level_2 = probas32[level_1_id]
        probas_level_2_species = probas_level_2[targets_level_1_species_idx]

        num_current_level_2 = probas_level_2.size(1)    #Skates 下有2个类别

        for level_2_id in range(num_current_level_2):  #遍历每一个 level-2的类别
            targets_level_2_sp_index = torch.where(targets_level_2_species==level_2_id)  #自动把-1的level 2 label筛了

            if len(targets_level_2_sp_index[0])==0:
                continue

            targets_level_1_sp = targets_level_1_species[targets_level_2_sp_index]
            probas_level_1_sp = probas_level_1_species[targets_level_2_sp_index]
            targets_level_2_sp = targets_level_2_species[targets_level_2_sp_index]
            probas_level_2_sp = probas_level_2_species[targets_level_2_sp_index]

            _, predicted_labels_level_2_sp = torch.max(probas_level_2_sp, 1)  # 返回的是下标 从0开始

            index_ = find_previous_len(level_1_id, hierarchy_dict)  #level_1_id=2
            name = level_2_names[level_2_id + index_]

            # embed()
            num_examples2[name] += targets_level_2_sp.size(0)
            tp2[name] += (predicted_labels_level_2_sp == targets_level_2_sp).sum()


            _, predicted_labels_level_1_sp = torch.max(probas_level_1_sp, 1)  # targets 为skates的所有sample 的预测
            # embed()
            correct_ind = torch.where(predicted_labels_level_1_sp == targets_level_1_sp)
            tp2_p1p2[name] += (predicted_labels_level_2_sp[correct_ind] == targets_level_2_sp[correct_ind]).sum()

        # embed()  之前会continue跳过这一步




    return tp1, num_examples1,tp2, num_examples2, tp2_p1p2


def avg_acc(num_examples, tp, acc):
    denominator = 0
    nominator = 0
    for name in num_examples:
        denominator += num_examples[name]
        nominator += tp[name]
        acc[name] = (tp[name]*1.0/num_examples[name]).item()

    avg_level_acc = (nominator*1.0 / denominator).item()


    return avg_level_acc, acc

def avg_acc_track_based(num_examples, tp, acc):
    denominator = 0
    nominator = 0
    for name in num_examples:
        denominator += num_examples[name]
        nominator += tp[name]
        if num_examples[name]==0:
            print(name)
            embed()
        acc[name] = tp[name]/num_examples[name]

    avg_level_acc = nominator / denominator


    return avg_level_acc, acc




#计算level-1的准确率，和level-2的准确率
def compute_accuracy_model12(model, data_loader, device):

    #初始化 统计字典
    tp1 = {}
    num_examples1 = {}
    acc_1={}
    tp2 = {}
    num_examples2 = {}
    acc_2 = {}

    #level-2的准确率用p1*p2判断
    tp2_p1p2 = {}
    acc_2_p1p2 = {}


    for key in list(hierarchy_dict.keys()):
        tp1[key] = 0
        num_examples1[key] =  0
        acc_1[key]=0

        for species in hierarchy_dict[key]:
            tp2[species] = 0
            num_examples2[species] = 0
            acc_2[species] = 0

            tp2_p1p2[species] = 0
            acc_2_p1p2[species] = 0

    for i, (features, targets) in tqdm(enumerate(data_loader)):
        features = features.to(device)
        targets = targets.to(device)


        logits, probas = model(features)

        # tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets)
        tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict_fast(tp1, num_examples1, tp2, num_examples2, tp2_p1p2, probas, targets)

    ###计算 准确率
    avg_level_1_acc, acc_1 = avg_acc(num_examples1, tp1, acc_1)

    avg_level_2_acc, acc_2 = avg_acc(num_examples2, tp2, acc_2)

    avg_level_2_acc_p1p2, acc_2_p1p2 = avg_acc(num_examples2, tp2_p1p2, acc_2_p1p2)
    # embed()

    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_p1p2, acc_1, acc_2, acc_2_p1p2

def find_who_stop_at_level_1(low_conf_indx, targets_31, species_stop_at_level_1):

    stop_at_level_1= targets_31[low_conf_indx]

    for sample in stop_at_level_1:
        name = level_2_names[sample[1]]
        species_stop_at_level_1[name]+=1



    return species_stop_at_level_1




def avg_acc_can_stop_1(probas, probas_level2, targets_31, species_stop_at_level_1, threshold):

    tp_can_stop_1 = 0
    # num_0=0


    # level-1, level-2的单独label
    targets_level_1 = targets_31[:, 0]
    targets_level_2_31 = targets_31[:, 1]

    # level-1的预测
    probas_level_1 = probas[0]

    only_level_1_idx = torch.where(targets_level_2_31==-1)  #这些sample的预测，只能停在level-1
    num_0 = len(only_level_1_idx[0])
    if num_0!=0:
        targets_only_level_1 = targets_level_1[only_level_1_idx]
        probas_only_level_1 = probas_level_1[only_level_1_idx]
        _, predicted_probas_only_level_1 = torch.max(probas_only_level_1, 1)
        tp_can_stop_1 += (predicted_probas_only_level_1 == targets_only_level_1).sum()


    can_level_2_idx = torch.where(targets_level_2_31 != -1)  # 这些sample的预测，可以采用level2的预测，也可以停在level-1
    num_2 = len(can_level_2_idx[0])
    if num_2 != 0:
        probas_level_1_can_level2 = probas_level_1[can_level_2_idx]
        probas_level2_can_level2 = probas_level2[can_level_2_idx]
        targets_level_1_can_level2 = targets_level_1[can_level_2_idx]
        targets_level_2_31_can_level2 = targets_level_2_31[can_level_2_idx]


        #level-2 31个product的预测： probas_level2
        ### 1：判断 probas_level2   31个里max值，是否大于0.95,  若yes，则用这个label，若no，则用Level-1的预测，并统计用了多少个level-1 Level-2
        (probas_level2_max, probas_level2_max_index) = torch.max(probas_level2_can_level2, dim=1)


        high_conf_indx = torch.where(probas_level2_max > threshold)[0]
        low_conf_indx = torch.where(probas_level2_max <= threshold)[0]

        num_level_1 = len(low_conf_indx)
        num_level_2 = len(high_conf_indx)

        if num_level_1 >0 and num_level_2>0:
            probas_level2_high_conf = probas_level2_can_level2[high_conf_indx]
            probas_level1_low_conf = probas_level_1_can_level2[low_conf_indx]

            targets_level_1_low_conf = targets_level_1_can_level2[low_conf_indx]
            targets_level_2_31_high_conf = targets_level_2_31_can_level2[high_conf_indx]

            species_stop_at_level_1 = find_who_stop_at_level_1(low_conf_indx, targets_31[can_level_2_idx], species_stop_at_level_1)

            # embed()
            _, predicted_labels_level_2_species = torch.max(probas_level2_high_conf, 1)
            _, predicted_labels_level_1_species = torch.max(probas_level1_low_conf, 1)

            tp_can_stop_1 += (predicted_labels_level_1_species == targets_level_1_low_conf).sum()
            tp_can_stop_1 += (predicted_labels_level_2_species == targets_level_2_31_high_conf).sum()

        elif num_level_1==0:
            probas_level2_high_conf = probas_level2_can_level2[high_conf_indx]
            # probas_level1_low_conf = probas_level_1_can_level2[low_conf_indx]

            # targets_level_1_low_conf = targets_level_1_can_level2[low_conf_indx]
            targets_level_2_31_high_conf = targets_level_2_31_can_level2[high_conf_indx]

            species_stop_at_level_1 = find_who_stop_at_level_1(low_conf_indx, targets_31[can_level_2_idx], species_stop_at_level_1)

            # embed()
            _, predicted_labels_level_2_species = torch.max(probas_level2_high_conf, 1)
            # _, predicted_labels_level_1_species = torch.max(probas_level1_low_conf, 1)

            # tp_can_stop_1 += (predicted_labels_level_1_species == targets_level_1_low_conf).sum()
            tp_can_stop_1 += (predicted_labels_level_2_species == targets_level_2_31_high_conf).sum()

        elif num_level_2==0:
            # probas_level2_high_conf = probas_level2_can_level2[high_conf_indx]
            probas_level1_low_conf = probas_level_1_can_level2[low_conf_indx]

            targets_level_1_low_conf = targets_level_1_can_level2[low_conf_indx]
            # targets_level_2_31_high_conf = targets_level_2_31_can_level2[high_conf_indx]

            species_stop_at_level_1 = find_who_stop_at_level_1(low_conf_indx, targets_31[can_level_2_idx], species_stop_at_level_1)

            # embed()
            # _, predicted_labels_level_2_species = torch.max(probas_level2_high_conf, 1)
            _, predicted_labels_level_1_species = torch.max(probas_level1_low_conf, 1)

            tp_can_stop_1 += (predicted_labels_level_1_species == targets_level_1_low_conf).sum()
            # tp_can_stop_1 += (predicted_labels_level_2_species == targets_level_2_31_high_conf).sum()

    else:
        num_level_2=0
        num_level_1=num_0



    return tp_can_stop_1, num_0,num_level_1, num_level_2, species_stop_at_level_1


def compute_accuracy_model7(model, data_loader, device):
    # 初始化 统计字典
    tp1 = {}
    num_examples1 = {}
    acc_1 = {}
    tp2 = {}
    num_examples2 = {}
    acc_2 = {}

    # level-2的准确率用p1*p2判断
    tp2_p1p2_31 = {}
    acc_2_p1p2_31 = {}
    tp2_p1p2_maxmax = {}
    acc_2_p1p2_maxmax = {}

    # can stop at level-1
    all_tp_can_stop_1 = 0
    all_num_0=0
    all_num_level_1 =0
    all_num_level_2 =0
    species_stop_at_level_1 = {}



    for key in list(hierarchy_dict.keys()):
        tp1[key] = 0
        num_examples1[key] = 0
        acc_1[key] = 0



        for species in hierarchy_dict[key]:
            tp2[species] = 0
            num_examples2[species] = 0
            acc_2[species] = 0

            tp2_p1p2_31[species] = 0
            acc_2_p1p2_31[species] = 0

            tp2_p1p2_maxmax[species] = 0
            acc_2_p1p2_maxmax[species] = 0
            species_stop_at_level_1[species]=0

    for i, (features, targets_all, targets_split) in tqdm(enumerate(data_loader)):
        features = features.to(device)
        targets_31 = targets_all.to(device)
        targets_split = targets_split.to(device)

        logits, probas, probas_level2= model(features)


        # prediction can stop at level-1
        tp_can_stop_1, num_0, num_level_1, num_level_2, species_stop_at_level_1 = avg_acc_can_stop_1(probas, probas_level2, targets_31, species_stop_at_level_1)
        all_num_0+=num_0
        all_tp_can_stop_1+=tp_can_stop_1
        all_num_level_1+=num_level_1
        all_num_level_2+=num_level_2

        # tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets)
        tp1, num_examples1,tp2, num_examples2, tp2_p1p2_31, tp2_p1p2_maxmax = put_in_dict_fast_model7(tp1, num_examples1, tp2, num_examples2,
                                                                       tp2_p1p2_31, tp2_p1p2_maxmax,probas,probas_level2, targets_31, targets_split)





    ###计算 准确率
    avg_level_1_acc, acc_1 = avg_acc(num_examples1, tp1, acc_1)

    avg_level_2_acc, acc_2 = avg_acc(num_examples2, tp2, acc_2)

    avg_level_2_acc_p1p2_31, acc_2_p1p2_31 = avg_acc(num_examples2, tp2_p1p2_31, acc_2_p1p2_31)

    avg_level_2_acc_p1p2_maxmax, acc_2_p1p2_maxmax = avg_acc(num_examples2, tp2_p1p2_maxmax, acc_2_p1p2_maxmax)

    # embed()
    avg_acc_can_stop_level_1 = all_tp_can_stop_1.item()/(all_num_level_1+all_num_level_2 + all_num_0)

    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_p1p2_31, avg_level_2_acc_p1p2_maxmax, acc_1, acc_2, acc_2_p1p2_31,acc_2_p1p2_maxmax, avg_acc_can_stop_level_1, all_num_level_1, all_num_level_2, species_stop_at_level_1

def get_2_index(header):

    if header in level_1_names:
        head_idx=0
        column_idx = level_1_names.index(header)
    else:
        for j, group_name in enumerate(hierarchy_dict,0):
            if header in hierarchy_dict[group_name]:
                head_idx = j+1
                column_idx = hierarchy_dict[group_name].index(header)
                break

    return head_idx, column_idx

def choose_probs(probas, header):

    head_idx, column_idx = get_2_index(header)
    data = probas[head_idx][:,column_idx]


    return data

def choose_data(header,probas, img_names, targets_31,ids):

    if header=='id':
        data = ids.tolist()
        # embed()

    elif header=='img_name':
        data = list(img_names)
    elif header =='level-1 gt':
        data = targets_31[:,0].tolist()

    elif header == 'level-2 gt':
        data = targets_31[:, 1].tolist()

    else:
        data = choose_probs(probas, header)
        data = data.tolist()


    return data


def do_multiplication(probas):

    probas_mul = []

    for i, probas_i in enumerate(probas,0):
        if i==0:
            probas_mul.append(probas_i)
        elif i <=5 :
            probas_mul.append(probas[0][:,i-1:i] * probas_i)
        elif i==6:
            probas_mul.append(probas[0][:,i-1:] * probas_i)



    return probas_mul

def per_img_predictions_write_to_csv(i, epoch, probas, img_names, save_path,targets_31, ids):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_path = os.path.join(save_path,'epoch_'+str(epoch)+".csv")

    headers = ['id','img_name','level-1 gt', 'level-2 gt'] + level_1_names+level_2_names
    all_data = {}
    for header in headers:
        all_data[header] = choose_data(header,probas, img_names, targets_31,ids)

    dataframe = pd.DataFrame(all_data)
    if i==0:
        dataframe.to_csv(csv_path, mode='w', header=True, index=None)
    else:
        dataframe.to_csv(csv_path, mode='a', header=False, index=None)


def per_img_predictions_write_to_csv_model4(i, epoch, probas, img_names, save_path,targets_31, ids):
    probas = do_multiplication(probas)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_path = os.path.join(save_path,'epoch_'+str(epoch)+".csv")

    headers = ['id','img_name','level-1 gt', 'level-2 gt'] + level_1_names+level_2_names
    all_data = {}
    for header in headers:
        all_data[header] = choose_data(header,probas, img_names, targets_31,ids)

    dataframe = pd.DataFrame(all_data)
    if i==0:
        dataframe.to_csv(csv_path, mode='a', header=True, index=None)
    else:
        dataframe.to_csv(csv_path, mode='a', header=False, index=None)



def compute_accuracy_model7_track_based(model, data_loader, epoch,device, save_path, stop_at_level_1_threshold):
    # 初始化 统计字典
    tp1 = {}
    num_examples1 = {}
    acc_1 = {}
    tp2 = {}
    num_examples2 = {}
    acc_2 = {}

    # level-2的准确率用p1*p2判断
    tp2_p1p2_31 = {}
    acc_2_p1p2_31 = {}
    tp2_p1p2_maxmax = {}
    acc_2_p1p2_maxmax = {}

    # can stop at level-1
    all_tp_can_stop_1 = 0
    all_num_0=0
    all_num_level_1 =0
    all_num_level_2 =0
    species_stop_at_level_1 = {}



    for key in list(hierarchy_dict.keys()):
        tp1[key] = 0
        num_examples1[key] = 0
        acc_1[key] = 0



        for species in hierarchy_dict[key]:
            tp2[species] = 0
            num_examples2[species] = 0
            acc_2[species] = 0

            tp2_p1p2_31[species] = 0
            acc_2_p1p2_31[species] = 0

            tp2_p1p2_maxmax[species] = 0
            acc_2_p1p2_maxmax[species] = 0
            species_stop_at_level_1[species]=0

    for i, (features, targets_all, targets_split, img_names, ids) in tqdm(enumerate(data_loader)):
        features = features.to(device)
        targets_31 = targets_all.to(device)
        targets_split = targets_split.to(device)

        logits, probas, probas_level2= model(features)  ### model 67



        #写入csv文件
        per_img_predictions_write_to_csv(i,epoch, probas, img_names,save_path, targets_31, ids)



        # prediction can stop at level-1
        tp_can_stop_1, num_0, num_level_1, num_level_2, species_stop_at_level_1 = avg_acc_can_stop_1(probas, probas_level2, targets_31, species_stop_at_level_1, stop_at_level_1_threshold)
        all_num_0+=num_0
        all_tp_can_stop_1+=tp_can_stop_1
        all_num_level_1+=num_level_1
        all_num_level_2+=num_level_2

        # tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets)
        tp1, num_examples1,tp2, num_examples2, tp2_p1p2_31, tp2_p1p2_maxmax = put_in_dict_fast_model7(tp1, num_examples1, tp2, num_examples2,
                                                                       tp2_p1p2_31, tp2_p1p2_maxmax,probas,probas_level2, targets_31, targets_split)





    ###计算 准确率
    avg_level_1_acc, acc_1 = avg_acc(num_examples1, tp1, acc_1)

    # embed()
    avg_level_2_acc, acc_2 = avg_acc(num_examples2, tp2, acc_2)

    avg_level_2_acc_p1p2_31, acc_2_p1p2_31 = avg_acc(num_examples2, tp2_p1p2_31, acc_2_p1p2_31)

    avg_level_2_acc_p1p2_maxmax, acc_2_p1p2_maxmax = avg_acc(num_examples2, tp2_p1p2_maxmax, acc_2_p1p2_maxmax)

    # embed()
    avg_acc_can_stop_level_1 = all_tp_can_stop_1.item()/(all_num_level_1+all_num_level_2 + all_num_0)

    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_p1p2_31, avg_level_2_acc_p1p2_maxmax, acc_1, acc_2, acc_2_p1p2_31,acc_2_p1p2_maxmax, avg_acc_can_stop_level_1, all_num_level_1, all_num_level_2, species_stop_at_level_1



def compute_accuracy_model4_track_based(model, data_loader, epoch,device, save_path):
    # 初始化 统计字典
    tp1 = {}
    num_examples1 = {}
    acc_1 = {}
    tp2 = {}
    num_examples2 = {}
    acc_2 = {}

    # level-2的准确率用p1*p2判断
    tp2_p1p2_31 = {}
    acc_2_p1p2_31 = {}
    tp2_p1p2_maxmax = {}
    acc_2_p1p2_maxmax = {}

    # can stop at level-1
    all_tp_can_stop_1 = 0
    all_num_0=0
    all_num_level_1 =0
    all_num_level_2 =0
    species_stop_at_level_1 = {}



    for key in list(hierarchy_dict.keys()):
        tp1[key] = 0
        num_examples1[key] = 0
        acc_1[key] = 0



        for species in hierarchy_dict[key]:
            tp2[species] = 0
            num_examples2[species] = 0
            acc_2[species] = 0

            tp2_p1p2_31[species] = 0
            acc_2_p1p2_31[species] = 0

            tp2_p1p2_maxmax[species] = 0
            acc_2_p1p2_maxmax[species] = 0
            species_stop_at_level_1[species]=0

    for i, (features, targets_all, targets_split, img_names, ids) in tqdm(enumerate(data_loader)):
        features = features.to(device)
        targets_31 = targets_all.to(device)
        targets_split = targets_split.to(device)

        logits, probas, probas_level2= model(features)  ### model 67



        #写入csv文件
        per_img_predictions_write_to_csv_model4(i,epoch, probas, img_names,save_path, targets_31, ids)



        # prediction can stop at level-1
        tp_can_stop_1, num_0, num_level_1, num_level_2, species_stop_at_level_1 = avg_acc_can_stop_1(probas, probas_level2, targets_31, species_stop_at_level_1)
        all_num_0+=num_0
        all_tp_can_stop_1+=tp_can_stop_1
        all_num_level_1+=num_level_1
        all_num_level_2+=num_level_2

        # tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets)
        tp1, num_examples1,tp2, num_examples2, tp2_p1p2_31, tp2_p1p2_maxmax = put_in_dict_fast_model7(tp1, num_examples1, tp2, num_examples2,
                                                                       tp2_p1p2_31, tp2_p1p2_maxmax,probas,probas_level2, targets_31, targets_split)





    ###计算 准确率
    avg_level_1_acc, acc_1 = avg_acc(num_examples1, tp1, acc_1)

    avg_level_2_acc, acc_2 = avg_acc(num_examples2, tp2, acc_2)

    avg_level_2_acc_p1p2_31, acc_2_p1p2_31 = avg_acc(num_examples2, tp2_p1p2_31, acc_2_p1p2_31)

    avg_level_2_acc_p1p2_maxmax, acc_2_p1p2_maxmax = avg_acc(num_examples2, tp2_p1p2_maxmax, acc_2_p1p2_maxmax)

    # embed()
    avg_acc_can_stop_level_1 = all_tp_can_stop_1.item()/(all_num_level_1+all_num_level_2 + all_num_0)

    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_p1p2_31, avg_level_2_acc_p1p2_maxmax, acc_1, acc_2, acc_2_p1p2_31,acc_2_p1p2_maxmax, avg_acc_can_stop_level_1, all_num_level_1, all_num_level_2, species_stop_at_level_1

def compute_accuracy_model0(model, data_loader, device):
    #每个类别的准确率
    tp = {}
    num_examples_species = {}
    indivi_acc = {}
    #初始化字典
    for key in list(hierarchy_dict.keys()):
        for species in hierarchy_dict[key]:
            tp[species] = 0
            num_examples_species[species] = 0
            indivi_acc[species] = 0


    num_cls = len(tp)


    correct_pred, num_examples = 0, 0
    for i, (features, targets) in tqdm(enumerate(data_loader)):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)  #返回的是下标 从0开始
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()


        ### 每个类别的准确率
        for species_id in range(num_cls):
            target_species_index = torch.where(targets==species_id)  # 从当前batch的targets，中获得一个类别的target index


            if len(target_species_index[0])==0: #当前batch里没有这个类别
                continue


            target_species = targets[target_species_index]  # 从当前batch的targets，中获得一个类别的target
            probas_species = probas[target_species_index]   # 从当前batch的probas，中获得相应sample的 预测

            _, predicted_labels_species = torch.max(probas_species, 1)  # 返回的是下标 从0开始

            name = level_2_names[species_id]
            num_examples_species[name]+=target_species.size(0)
            tp[name] += (predicted_labels_species == target_species).sum()

    ### 平均acc，每个类比的 平均acc
    avg_acc = correct_pred.float()/num_examples * 100
    for name in num_examples_species:

        indivi_acc[name] = tp[name].float()/num_examples_species[name]

    # embed()
    return avg_acc, indivi_acc



def compute_accuracy_model3(model31,model32, data_loader, device):

    #初始化 统计字典
    tp1 = {}
    num_examples1 = {}
    acc_1={}
    tp2 = {}
    num_examples2 = {}
    acc_2 = {}

    #level-2的准确率用p1*p2判断
    tp2_p1p2 = {}
    acc_2_p1p2 = {}


    for key in list(hierarchy_dict.keys()):
        tp1[key] = 0
        num_examples1[key] =  0
        acc_1[key]=0

        for species in hierarchy_dict[key]:
            tp2[species] = 0
            num_examples2[species] = 0
            acc_2[species] = 0

            tp2_p1p2[species] = 0
            acc_2_p1p2[species] = 0

    for i, (features, targets) in tqdm(enumerate(data_loader)):
        features = features.to(device)
        targets = targets.to(device)

        logits31, probas31 = model31(features)
        logits32, probas32 = model32(features)

        # tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets)
        tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict_fast_model3(tp1, num_examples1, tp2, num_examples2, tp2_p1p2, probas31,probas32, targets)




    # embed()
    ###计算 准确率

    avg_level_1_acc, acc_1 = avg_acc(num_examples1, tp1, acc_1)

    avg_level_2_acc, acc_2 = avg_acc(num_examples2, tp2, acc_2)

    avg_level_2_acc_p1p2, acc_2_p1p2 = avg_acc(num_examples2, tp2_p1p2, acc_2_p1p2)

    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_p1p2, acc_1, acc_2, acc_2_p1p2

def change_label_to_31(targets_level_1, targets_level_2):
    targets_level_2_31 = torch.zeros_like(targets_level_2)
    # embed()
    for i in range(targets_level_2_31.size(0)):

        level_1_id = targets_level_1[i]
        level_2_id = targets_level_2[i]

        targets_level_2_31 [i] = find_id_in_31(level_1_id, level_2_id)

    return targets_level_2_31

def avg_acc_can_stop_1_model4(probas, targets):
    tp_can_stop_1 = 0


    # level-1, level-2的单独label
    targets_level_1 = targets[:, 0]
    targets_level_2 = targets[:, 1]
    targets_level_2_31 = change_label_to_31(targets_level_1, targets_level_2)

    # level-1, level-2的预测
    probas_level_1 = probas[0]
    probas_level_2_31 = multiply_7_head(probas[0], probas)









    (probas_level2_max, probas_level2_max_index) = torch.max(probas_level_2_31, dim=1)
    high_conf_indx = torch.where(probas_level2_max > 0.95)[0]
    low_conf_indx = torch.where(probas_level2_max <= 0.95)[0]
    num_level_1 = len(low_conf_indx)
    num_level_2 = len(high_conf_indx)

    probas_level2_high_conf = probas_level_2_31[high_conf_indx]
    probas_level1_low_conf = probas_level_1[low_conf_indx]

    targets_level_1_low_conf = targets_level_1[low_conf_indx]
    targets_level_2_31_high_conf = targets_level_2_31[high_conf_indx]

    _, predicted_labels_level_2_species = torch.max(probas_level2_high_conf, 1)
    _, predicted_labels_level_1_species = torch.max(probas_level1_low_conf, 1)

    tp_can_stop_1 += (predicted_labels_level_1_species == targets_level_1_low_conf).sum()
    # embed()
    tp_can_stop_1 += (predicted_labels_level_2_species == targets_level_2_31_high_conf).sum()


    return tp_can_stop_1, num_level_1, num_level_2


def compute_accuracy_model4(model, data_loader, device):

    #初始化 统计字典
    tp1 = {}
    num_examples1 = {}
    acc_1={}
    tp2 = {}
    num_examples2 = {}
    acc_2 = {}

    #level-2的准确率用p1*p2判断
    tp2_p1p2_maxmax = {}
    acc_2_p1p2_maxmax = {}
    tp2_p1p2_31 = {}
    acc_2_p1p2_31 = {}

    # can stop at level-1
    all_tp_can_stop_1 = 0
    all_num_level_1 = 0
    all_num_level_2 =0


    for key in list(hierarchy_dict.keys()):
        tp1[key] = 0
        num_examples1[key] =  0
        acc_1[key]=0

        for species in hierarchy_dict[key]:
            tp2[species] = 0
            num_examples2[species] = 0
            acc_2[species] = 0

            tp2_p1p2_maxmax[species] = 0
            acc_2_p1p2_maxmax[species] = 0

            tp2_p1p2_31 [species] = 0
            acc_2_p1p2_31 [species] = 0

    for i, (features, targets) in tqdm(enumerate(data_loader)):
        features = features.to(device)
        targets = targets.to(device)


        logits, probas = model(features)

        # prediction can stop at level-1
        tp_can_stop_1, num_level_1, num_level_2 = avg_acc_can_stop_1_model4(probas, targets)
        all_tp_can_stop_1 += tp_can_stop_1
        all_num_level_1 += num_level_1
        all_num_level_2 += num_level_2


        # tp1, num_examples1, tp2, num_examples2, tp2_p1p2 = put_in_dict(tp1, num_examples1,tp2, num_examples2,tp2_p1p2,probas, targets)
        tp1, num_examples1, tp2, num_examples2, tp2_p1p2_maxmax,tp2_p1p2_31  = put_in_dict_fast_model4(tp1, num_examples1, tp2, num_examples2, tp2_p1p2_maxmax, tp2_p1p2_31,probas, targets)

    ###计算 准确率
    avg_level_1_acc, acc_1 = avg_acc(num_examples1, tp1, acc_1)

    avg_level_2_acc, acc_2 = avg_acc(num_examples2, tp2, acc_2)

    avg_level_2_acc_maxmax, acc_2_maxmax = avg_acc(num_examples2, tp2_p1p2_maxmax, acc_2_p1p2_maxmax)

    avg_level_2_acc_31, acc_2_31 = avg_acc(num_examples2, tp2_p1p2_31, acc_2_p1p2_31)
    # embed()

    avg_acc_can_stop_level_1 = all_tp_can_stop_1.item() / (all_num_level_1 + all_num_level_2)

    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_maxmax, avg_level_2_acc_31, acc_1, acc_2, acc_2_maxmax, acc_2_31, avg_acc_can_stop_level_1, all_num_level_1, all_num_level_2




import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
def draw_loss(Loss_list,  acc_1_val_list, acc_2_val_list,acc_2_p1p2_val_list):
    host1 = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    # par1 = host.twinx()  # 共享x轴

    # set labels
    host1.set_xlabel("steps")
    host1.set_ylabel("test-loss")
    # par1.set_ylabel("test-accuracy")

    # plot curves
    p1, = host1.plot(range(1,len(Loss_list)+1), Loss_list, label="loss")
    # p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host1.legend(loc=5)

    # set label color
    host1.axis["left"].label.set_color(p1.get_color())
    # par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
    host1.set_ylim([0,10])

    plt.draw()
    plt.savefig('Loss curve.png')
    # if batch_idx%700==0:
    plt.show()
    plt.close()



    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    # par1 = host.twinx()  # 共享x轴

    # set labels
    host.set_xlabel("epochs")
    host.set_ylabel("test-accuracy")

    # plot curves
    p1, = host.plot(range(1,len(acc_1_val_list)+1), acc_1_val_list, label="level-1")
    p2, = host.plot(range(1,len(acc_2_val_list)+1), acc_2_val_list, label="level-2")
    p1p2, = host.plot(range(1,len(acc_2_p1p2_val_list)+1), acc_2_p1p2_val_list, label="level-1&2")
    # p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["left"].label.set_color(p2.get_color())
    host.axis["left"].label.set_color(p1p2.get_color())
    # par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.savefig('Val accuracy curve.png')
    # if batch_idx % 700 == 0:
    plt.show()
    plt.close()



def show_img(features):
    unloader = transforms.ToPILImage()
    # embed()

    for img in features:
        # embed()
        image = img.cpu().clone()  # we clone the tensor to not do changes on it
        image = unloader(image)
        image.show()
        # embed()










def calculate_num_class(hierarchy_dict):
    NUM_CLASSES = 0
    NUM_level_1_CLASSES = 0
    for key in hierarchy_dict:
        NUM_level_1_CLASSES+=1
        NUM_CLASSES += len(hierarchy_dict[key])

    NUM_CLASSES += len(hierarchy_dict)
    NUM_level_2_CLASSES = NUM_CLASSES - NUM_level_1_CLASSES

    return NUM_level_1_CLASSES,  NUM_level_2_CLASSES


def calculate_num_class_model0(hierarchy_dict):
    NUM_CLASSES = 0
    for key in hierarchy_dict:
        NUM_CLASSES += len(hierarchy_dict[key])

    return NUM_CLASSES


def calculate_num_class_for_each_head(hierarchy_dict):


    keys = list(hierarchy_dict.keys())
    num_1 = len(hierarchy_dict)

    num_2 = len(hierarchy_dict[keys[0]])
    num_3 = len(hierarchy_dict[keys[1]])
    num_4 = len(hierarchy_dict[keys[2]])
    num_5 = len(hierarchy_dict[keys[3]])
    num_6 = len(hierarchy_dict[keys[4]])
    num_7 = len(hierarchy_dict[keys[5]])

    NUM_CLASSES = [num_1, num_2, num_3, num_4,num_5,num_6, num_7]

    return NUM_CLASSES



def get_index(hierarchy_dict):
    #这个函数要求level-1只能有6类
    keys = list(hierarchy_dict.keys())
    idx_1 = len(hierarchy_dict)
    idx_2 = idx_1 + len(hierarchy_dict[keys[0]])
    idx_3 = idx_2 + len(hierarchy_dict[keys[1]])
    idx_4 = idx_3 + len(hierarchy_dict[keys[2]])
    idx_5 = idx_4 + len(hierarchy_dict[keys[3]])
    idx_6 = idx_5 + len(hierarchy_dict[keys[4]])

    return idx_1, idx_2, idx_3, idx_4, idx_5, idx_6


def get_index_for_model_2(hierarchy_dict):

    keys = list(hierarchy_dict.keys())
    idx_list = []
    for j, key in enumerate(keys):
        if j==0:
            idx_list.append(len(hierarchy_dict[key]))
        else:
            idx_list.append(len(hierarchy_dict[key]) + idx_list[j-1])


    return idx_list

def find_level2_head_loss_for_model12(targets, logits_list):
    #logits_list 是7个head， 按照顺序排的

    cost_level_2=0
    level_1_target = targets[:, 0]
    level_2_target = targets[:, 1]

    for group_id in range(6):
        # 找到这个Group的 head，在所有sample上的预测 128 x cls
        level_2_logit = logits_list[group_id + 1]

        # 从128个sample 里找到这个Group里的所有sample 和对应的logit
        idx = torch.where(level_1_target==group_id)

        if len(idx[0]) ==0:  #本batch没有当前group的sample
            continue

        level_2_tg = level_2_target[idx]
        level_2_lt = level_2_logit[idx]

        # 只把不为-1的sample 调出来 算loss
        idx2 = torch.where(level_2_tg!=-1)
        if len(idx2[0]) ==0:  #本batch没有当前group的sample
            continue

        selected_level_2_tg = level_2_tg[idx2]
        selected_level_2_lt = level_2_lt[idx2]
        # if np.any(np.isnan(selected_level_2_tg.cpu())) or np.any(np.isnan(selected_level_2_lt.cpu())):
        #     embed()

        # print(selected_level_2_tg)
        # print(selected_level_2_lt)

        cost_level_2 += F.cross_entropy(selected_level_2_lt, selected_level_2_tg)

    return cost_level_2


def find_level2_head_loss_for_model7(probas_level2, targets):

    # embed()
    level_2_targets = targets[:,1]
    idx = torch.where(level_2_targets!=-1)
    cost_level_2 = torch.nn.NLLLoss()(torch.log(probas_level2[idx]),level_2_targets[idx])

    return cost_level_2





def find_level2_head_loss_for_model3(targets, logits_list):
    #logits_list 是6个head， 按照顺序排的

    cost_level_2=0
    level_1_target = targets[:, 0]
    level_2_target = targets[:, 1]

    for group_id in range(6):
        # 找到这个Group的 head，在所有sample上的预测 128 x cls
        level_2_logit = logits_list[group_id]

        # 从128个sample 里找到这个Group里的所有sample 和对应的logit
        idx = torch.where(level_1_target==group_id)

        if len(idx[0]) ==0:  #本batch没有当前group的sample
            continue

        level_2_tg = level_2_target[idx]
        level_2_lt = level_2_logit[idx]

        # 只把不为-1的sample 调出来 算loss
        idx2 = torch.where(level_2_tg!=-1)
        if len(idx2[0]) ==0:  #本batch没有当前group的sample
            continue

        selected_level_2_tg = level_2_tg[idx2]
        selected_level_2_lt = level_2_lt[idx2]

        cost_level_2 += F.cross_entropy(selected_level_2_lt, selected_level_2_tg)

    return cost_level_2

from collections import OrderedDict
def read_csv_as_dict(area_order_file):
    csvFile_all = open(area_order_file, 'r')
    dict_reader_all = csv.DictReader(csvFile_all)

    track_target = OrderedDict()
    for i, row in enumerate(dict_reader_all):
        track_id = row['id']
        if track_id not in track_target:
            track_target[track_id] = []
        track_target[track_id].append(row)
    csvFile_all.close()

    return track_target


def sorted_dict(training_eval_data):
    sorted_tuple = sorted(training_eval_data.items(), key  = lambda item:item[1], reverse=True)
    sorted_dict = {}
    for each_tuple in sorted_tuple:
        sorted_dict[each_tuple[0]] = each_tuple[1]

    return sorted_dict


def get_sub_dict_level_2(track_predict_level_2, name_1_gt):
    sub_track_predict_level_2={}
    for species in hierarchy_dict[name_1_gt]:
        sub_track_predict_level_2[species] = track_predict_level_2[species]

    return sub_track_predict_level_2

def track_based_accuracy(save_path, epoch, threshold_1):


    valid_track = read_csv_as_dict(os.path.join(save_path,'epoch_'+str(epoch)+".csv"))

    track_predict_level_1 = {}  #同一个track的level-1预测，conf相加
    track_predict_level_2 = {}  #同一个track的level-2预测，conf相加
    tp_1={}
    total_num_1 = {}
    tp_2 = {}
    total_num_2 = {}
    tp_2_31 = {}
    total_num_2_31 = {}
    tp_2_maxmax = {}
    total_num_2_maxmax = {}
    tp_can_stop_at_level_1=0
    num_stop_at_level_1=0
    num_stop_at_level_2=0
    species_stop_at_level_1 = {}

    acc_1={}
    acc_2={}
    acc_2_31={}
    acc_2_maxmax={}

    for key in list(hierarchy_dict.keys()):
        track_predict_level_1[key]=0
        tp_1[key]=0
        total_num_1[key]=0
        acc_1[key]=0


        for species in hierarchy_dict[key]:
            track_predict_level_2[species]=0
            tp_2[species] = 0
            total_num_2[species] = 0
            tp_2_31[species] = 0
            total_num_2_31[species] = 0
            tp_2_maxmax[species] = 0
            total_num_2_maxmax[species] = 0
            species_stop_at_level_1[species]=0
            acc_2[species]=0
            acc_2_31[species]=0
            acc_2_maxmax[species]=0


    for j, track_id in enumerate(valid_track,1):
        each_track = valid_track[track_id]  # alist

        #只对当前track进行清零，重新统计 confidence weight，和投票数
        for key in list(hierarchy_dict.keys()):
            track_predict_level_1[key] = 0
            for species in hierarchy_dict[key]:
                track_predict_level_2[species] = 0



        #当前track里所有帧的conf 相加
        for idx, each_frame in enumerate(each_track, 1):
            for key in track_predict_level_1:
                # embed()
                track_predict_level_1[key] += float(each_frame[key])

            # max_conf = 0
            for key in track_predict_level_2:
                track_predict_level_2[key] += float(each_frame[key])




        #计算 level-1的 各个类别的 tp 和总个数
        level_1_gt = each_track[0]['level-1 gt']
        name_1_gt = level_1_names[int(level_1_gt)]
        sorted_tuple = sorted(track_predict_level_1.items(), key  = lambda item:item[1], reverse=True)
        name_1_pred = sorted_tuple[0][0]

        total_num_1[name_1_gt]+=1
        if name_1_gt==name_1_pred:
            tp_1[name_1_gt]+=1


        level_2_gt = each_track[0]['level-2 gt']

        if level_2_gt!='-1': #只有level-2有 gt的，才进行level-2的准确率计算
            name_2_gt = level_2_names[int(level_2_gt)]

            # 1. 在给定level-1 gt的情况下，计算level-2的 各个类别的 tp 和总个数
            sub_track_predict_level_2 = get_sub_dict_level_2(track_predict_level_2, name_1_gt)
            sorted_tuple = sorted(sub_track_predict_level_2.items(), key=lambda item: item[1], reverse=True)
            name_2_pred = sorted_tuple[0][0]

            total_num_2[name_2_gt] += 1
            if name_2_gt == name_2_pred:
                tp_2[name_2_gt] += 1

            # 2. 先找到level-1最大的max，再找相应Head里最大的max，计算level-2的 各个类别的 tp 和总个数
            if name_1_pred not in ['Bird','Bait','Fishing Gear','Misc_Other']:
                sub_track_predict_level_2 = get_sub_dict_level_2(track_predict_level_2, name_1_pred)
                sorted_tuple = sorted(sub_track_predict_level_2.items(), key=lambda item: item[1], reverse=True)
                name_2_pred = sorted_tuple[0][0]

                total_num_2_maxmax[name_2_gt] += 1
                if name_2_gt == name_2_pred:
                    tp_2_maxmax[name_2_gt] += 1

            else:
                total_num_2_maxmax[name_2_gt] += 1


            # 找到31个(已经有product)里面最大的，计算level-2的 各个类别的 tp 和总个数
            sorted_tuple = sorted(track_predict_level_2.items(), key=lambda item: item[1], reverse=True)
            name_2_pred = sorted_tuple[0][0]

            total_num_2_31[name_2_gt] += 1
            if name_2_gt == name_2_pred:
                tp_2_31[name_2_gt] += 1



        #当前track的预测可以stop at level-1，也可以留在level-2，如果31个max 的平均值conf <0.95
        sorted_tuple = sorted(track_predict_level_2.items(), key=lambda item: item[1], reverse=True)
        name_2_pred = sorted_tuple[0][0]
        name_2_avg_conf = sorted_tuple[0][1]/idx


        # threshold_1 = 0.85  # model-7 more
        # threshold_1 = 0.91 #model-7
        # threshold_1 = 0.85  # model-4
        # threshold_1 = 0.9 #model-6

        if name_2_avg_conf>= threshold_1 and level_2_gt!='-1':  #只有level 2的gt的才可以进入level-2计算准确率
            if name_2_pred==name_2_gt:
                tp_can_stop_at_level_1 +=1
            num_stop_at_level_2 +=1
        else:                                          #没有level 2的gt的，直接进入level-1计算准确率
            if name_1_pred==name_1_gt:
                tp_can_stop_at_level_1 += 1
            num_stop_at_level_1 += 1
            species_stop_at_level_1[name_2_gt] +=1


    ###计算 准确率

    avg_level_1_acc, acc_1 = avg_acc_track_based(total_num_1, tp_1, acc_1)
    avg_level_2_acc, acc_2 = avg_acc_track_based(total_num_2, tp_2, acc_2)
    avg_level_2_acc_31, acc_2_31 = avg_acc_track_based(total_num_2_31, tp_2_31, acc_2_31)
    avg_level_2_acc_maxmax, acc_2_maxmax = avg_acc_track_based(total_num_2_maxmax, tp_2_maxmax, acc_2_maxmax)
    avg_acc_can_stop_level_1 = tp_can_stop_at_level_1 / j





    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_31, avg_level_2_acc_maxmax, \
            acc_1, acc_2, acc_2_31, acc_2_maxmax, avg_acc_can_stop_level_1, num_stop_at_level_1, num_stop_at_level_2, species_stop_at_level_1


def track_based_accuracy_majority_vote(save_path, epoch):


    valid_track = read_csv_as_dict(os.path.join(save_path,'epoch_'+str(epoch)+".csv"))

    majority_vote_level_1 = {}
    majority_vote_level_2 = {}
    majority_conf_level_2 = {}

    tp_1={}
    total_num_1 = {}
    tp_2 = {}
    total_num_2 = {}
    tp_2_31 = {}
    total_num_2_31 = {}
    tp_2_maxmax = {}
    total_num_2_maxmax = {}
    tp_can_stop_at_level_1_maj = 0
    num_stop_at_level_1_maj = 0
    num_stop_at_level_2_maj = 0
    species_stop_at_level_1_maj = {}

    acc_1={}
    acc_2={}
    acc_2_31={}
    acc_2_maxmax={}

    for key in list(hierarchy_dict.keys()):
        tp_1[key]=0
        total_num_1[key]=0
        acc_1[key]=0


        for species in hierarchy_dict[key]:
            majority_vote_level_2[species] = 0
            majority_conf_level_2[species]=0
            tp_2[species] = 0
            total_num_2[species] = 0
            tp_2_31[species] = 0
            total_num_2_31[species] = 0
            tp_2_maxmax[species] = 0
            total_num_2_maxmax[species] = 0
            species_stop_at_level_1_maj[species] = 0
            acc_2[species]=0
            acc_2_31[species]=0
            acc_2_maxmax[species]=0


    for j, track_id in enumerate(valid_track,1):
        each_track = valid_track[track_id]  # alist

        #只对当前track进行清零，重新统计 confidence weight，和投票数
        for key in list(hierarchy_dict.keys()):
            majority_vote_level_1[key] = 0
            for species in hierarchy_dict[key]:
                majority_vote_level_2[species] = 0
                majority_conf_level_2[species]=0

        # embed()
        #当前track里所有帧的conf 相加
        for idx, each_frame in enumerate(each_track, 1):

            max_conf_1 = 0
            for key in list(hierarchy_dict.keys()):
                # majority vote
                if max_conf_1 < float(each_frame[key]):
                    max_conf_1 = float(each_frame[key])
                    vote_name_1 = key
                # 记录当前选票
            majority_vote_level_1[vote_name_1] += 1

            max_conf_2 = 0
            for key in level_2_names:
                #majority vote
                if max_conf_2<float(each_frame[key]):
                    max_conf_2 = float(each_frame[key])
                    vote_name_2 = key
            #记录当前选票 和 conf
            majority_conf_level_2[vote_name_2] +=float(each_frame[vote_name_2])
            majority_vote_level_2[vote_name_2] +=1


        #计算 level-1的 各个类别的 tp 和总个数
        level_1_gt = each_track[0]['level-1 gt']
        name_1_gt = level_1_names[int(level_1_gt)]

        sorted_tuple = sorted(majority_vote_level_1.items(), key=lambda item: item[1], reverse=True)
        name_1_pred = sorted_tuple[0][0]  # vote 最多的group

        total_num_1[name_1_gt]+=1
        if name_1_gt==name_1_pred:
            tp_1[name_1_gt]+=1

        #在给定level-1 gt的情况下，计算level-2的 各个类别的 tp 和总个数
        level_2_gt = each_track[0]['level-2 gt']

        if level_2_gt!='-1': #只有level-2有 gt的，才进行level-2的准确率计算
            name_2_gt = level_2_names[int(level_2_gt)]
            sub_majority_vote_level_2 = get_sub_dict_level_2(majority_vote_level_2, name_1_gt)
            sorted_tuple = sorted(sub_majority_vote_level_2.items(), key=lambda item: item[1], reverse=True)
            name_2_pred = sorted_tuple[0][0]

            total_num_2[name_2_gt] += 1
            if name_2_gt == name_2_pred:
                tp_2[name_2_gt] += 1

            # 先找到level-1最大的max，再找相应Head里最大的max，计算level-2的 各个类别的 tp 和总个数
            if name_1_pred not in ['Bird', 'Bait', 'Fishing Gear', 'Misc_Other']:
                sub_majority_vote_level_2 = get_sub_dict_level_2(majority_vote_level_2, name_1_pred)
                sorted_tuple = sorted(sub_majority_vote_level_2.items(), key=lambda item: item[1], reverse=True)
                name_2_pred = sorted_tuple[0][0]

                total_num_2_maxmax[name_2_gt] += 1
                if name_2_gt == name_2_pred:
                    tp_2_maxmax[name_2_gt] += 1
            else:
                total_num_2_maxmax[name_2_gt] += 1


            # 找到31个(已经有product)里面最大的，计算level-2的 各个类别的 tp 和总个数
            sorted_tuple = sorted(majority_vote_level_2.items(), key=lambda item: item[1], reverse=True)
            name_2_pred = sorted_tuple[0][0]

            total_num_2_31[name_2_gt] += 1
            if name_2_gt == name_2_pred:
                tp_2_31[name_2_gt] += 1





        #majority vote选出Level-2的预测，计算平均conf， 当前track的预测可以stop at level-1，也可以留在level-2，如果31个max 的平均值conf <0.95
        sorted_tuple = sorted(majority_vote_level_2.items(), key=lambda item: item[1], reverse=True)
        name_2_pred = sorted_tuple[0][0]
        majority_avg_conf = majority_conf_level_2[name_2_pred]/sorted_tuple[0][1]


        ## greedy search to make sure the same accuracy as level-1
        # conf_threshold = 0.975 # model-7
        conf_threshold = 0.955  # model-7  more
        # conf_threshold = 0.955 # model-4
        # conf_threshold = 0.96  # model-6

        if majority_avg_conf>=conf_threshold and level_2_gt!='-1':  #只有level 2的gt的才可以进入level-2计算准确率
            if name_2_pred==name_2_gt:
                tp_can_stop_at_level_1_maj +=1
            num_stop_at_level_2_maj +=1
        else:                                          #没有level 2的gt的，直接进入level-1计算准确率
            if name_1_pred==name_1_gt:
                tp_can_stop_at_level_1_maj += 1
            num_stop_at_level_1_maj += 1
            species_stop_at_level_1_maj[name_2_gt] +=1



    ###计算 准确率

    avg_level_1_acc, acc_1 = avg_acc_track_based(total_num_1, tp_1, acc_1)
    avg_level_2_acc, acc_2 = avg_acc_track_based(total_num_2, tp_2, acc_2)
    avg_level_2_acc_31, acc_2_31 = avg_acc_track_based(total_num_2_31, tp_2_31, acc_2_31)
    avg_level_2_acc_maxmax, acc_2_maxmax = avg_acc_track_based(total_num_2_maxmax, tp_2_maxmax, acc_2_maxmax)
    avg_acc_can_stop_level_1 = tp_can_stop_at_level_1_maj / j



    return avg_level_1_acc, avg_level_2_acc, avg_level_2_acc_31, avg_level_2_acc_maxmax, \
            acc_1, acc_2, acc_2_31, acc_2_maxmax, avg_acc_can_stop_level_1, num_stop_at_level_1_maj, num_stop_at_level_2_maj, species_stop_at_level_1_maj

import matplotlib.pyplot as plt
def plot_hist(all_conf, key, flag):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    # plt.hist(np.array(all_conf), bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.hist(np.array(all_conf),  bins=50,range=(0,1),
             histtype='bar', label='predicted conf on train data', edgecolor='k', color='darkorange',alpha=0.8)
    # 显示横轴标签
    plt.xlabel("confidence")
    # 显示纵轴标签
    plt.ylabel("count")
    # 显示图标题
    plt.legend()
    plt.title(key + ' confidence histogram'+'-'+flag)
    plt.show()

from skimage import filters
def calculate_threshold(track_predict_level_1, Otsu_Threshold_level_1, flag):

    for key in track_predict_level_1:
        all_conf = track_predict_level_1[key]
        plot_hist(all_conf, key, flag)


        thresh = filters.threshold_otsu(np.array(all_conf))
        # thresh = filters.threshold_multiotsu(np.array(all_conf), classes=2)
        # embed()

        Otsu_Threshold_level_1[key] = thresh

    return Otsu_Threshold_level_1



def Otsu_Threshold(save_path, epoch):

    train_track = read_csv_as_dict(os.path.join(save_path,'epoch_'+str(epoch)+".csv"))

    img_predict_level_1 = {}  #level-1所有图片gt的预测 存在这里
    track_predict_level_1 = {}  # level-1所有trac gt的预测 存在这里
    Otsu_Threshold_level_1_img = {}
    Otsu_Threshold_level_1_track = {}


    for key in list(hierarchy_dict.keys()):
        img_predict_level_1[key]=[]
        track_predict_level_1[key]=[]
        Otsu_Threshold_level_1_img[key] = 0 #最后的每个类别有一个threshold
        Otsu_Threshold_level_1_track[key] = 0


    for j, track_id in enumerate(train_track,1):
        each_track = train_track[track_id]  # alist

        level_1_gt = each_track[0]['level-1 gt']
        name_1_gt = level_1_names[int(level_1_gt)]

        #当前track里所有帧的conf 相加
        track_avg_conf=0
        for idx, each_frame in enumerate(each_track, 1):
            img_predict_level_1[name_1_gt].append(float(each_frame[name_1_gt]))
            track_avg_conf+=float(each_frame[name_1_gt])
        track_avg_conf/=idx
        track_predict_level_1[name_1_gt].append(track_avg_conf)
    # embed()

    # debug
    # img_num = 0
    # for key in track_predict_level_1:
    #     all_conf = track_predict_level_1[key]
    #     img_num+=len(all_conf)

    Otsu_Threshold_level_1_img = calculate_threshold(img_predict_level_1, Otsu_Threshold_level_1_img,'img')
    Otsu_Threshold_level_1_track = calculate_threshold(track_predict_level_1, Otsu_Threshold_level_1_track,'track')
    embed()



    return Otsu_Threshold_level_1


### May 28th 2021, for equlaization loss, modifyied from https://github.com/tztztztztz/eql.detectron2/issues/9
def name_to_logit_model7(name, hierarchy_dict):
    logits = []

    # name can be level-1 or level-2
    level_1 = np.array(list(hierarchy_dict.keys()))
    # label只标到 level-1，logits应该是 只有一个数字
    if len(np.where(level_1 == name)[0]) != 0:
        index = np.where(level_1 == name)[0][0]
        logits.append(index)
        logits.append(-1) #each element in list of batch should be of equal size，所以没有level-2的Label的  用-1 替代

    else:  # label不是level-1。是level-2，logit因该有两个数字!!!
        index_ = 0
        for group_id, group_name in enumerate(level_1):

            if name in hierarchy_dict[group_name]:
                logits.append(group_id)
                index = np.where(np.array(hierarchy_dict[group_name]) == name)[0][0]
                logits.append(index+index_)
                break
            else:
                index_ += len(hierarchy_dict[group_name])

    return np.array(logits)  # array之后才能直接转成tensor

from collections import Counter


def get_eql_class_weights(lambda_1,lambda_2, file_name):

    class_weights_level_1 = torch.tensor(np.zeros(len(level_1_names))).cuda()
    class_weights_level_2 = torch.tensor(np.zeros(len(level_2_names))).cuda()

    labels_level_1 = []
    labels_level_2 = []
    df = pd.read_csv(file_name)
    for name in df['class'].values:
        logits = name_to_logit_model7(name, hierarchy_dict)
        labels_level_1.append(logits[0])
        labels_level_2.append(logits[1])
    label_count_level_1 = Counter(labels_level_1)
    label_count_level_2 = Counter(labels_level_2)

    for j, label_count in enumerate([label_count_level_1, label_count_level_2]):
        print('level: {}'.format(j+1) )
        for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
            weights = [class_weights_level_1, class_weights_level_2][j]
            weights[label] = 1 if count > [lambda_1, lambda_2][j] else 0
            # if count == 1637:
            #     embed()
            print(' idx: {}, cls: {} img: {}, weight: {}'.format(idx, id_31_to_level_2_name(label) if j==1 else label, count, weights[label]))

    return class_weights_level_1, class_weights_level_2


def replace_masked_values(tensor, mask, replace_with):
    assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
    one_minus_mask = 1 - mask
    values_to_add = replace_with * one_minus_mask

    return tensor * mask + values_to_add

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class SoftmaxEQL(object):
    def __init__(self, lambda_1,lambda_2,  ignore_prob, file_name):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.ignore_prob = ignore_prob
        self.class_weights_level_1, self.class_weights_level_2 = get_eql_class_weights(self.lambda_1, self.lambda_2,file_name)
        self.idx_2, self.idx_3, self.idx_4, self.idx_5, self.idx_6, self.idx_7,self.idx_8,self.idx_9,self.idx_10 = get_index_for_model_2(hierarchy_dict)
        self.idx_list = [self.idx_2, self.idx_3, self.idx_4, self.idx_5, self.idx_6, self.idx_7,self.idx_8,self.idx_9,self.idx_10]



    def __call__(self, logits_list, targets):
        logits_0 = logits_list[0]
        target_0 = targets[:, 0]

        N, C = logits_0.shape


        not_ignored = self.class_weights_level_1.view(1, C).repeat(N, 1) # fixed values
        over_prob = (torch.rand(logits_0.shape).cuda() > self.ignore_prob).float()
        is_gt = target_0.new_zeros((N, C)).float()
        is_gt[torch.arange(N), target_0] = 1

        weights = ((not_ignored + over_prob + is_gt) > 0).float()# for head classes, not_ignored=1; for tail classes, if it's target or over prob, then weight=1,else weight =0
        logits_0 = replace_masked_values(logits_0, weights, 1e-7)
        loss_level_1 = F.cross_entropy(logits_0, target_0) #This criterion combines log_softmax and nll_loss in a single function.



        # level-2 loss
        # mask logits
        target_2 = targets[:, 1] # 0-30, 取值还可能是-1!

        idx = torch.where(target_2 != -1)
        target_2_ = target_2[idx]

        is_gt = target_2_.new_zeros((idx[0].shape[0], len(level_2_names))).float()
        is_gt[torch.arange(idx[0].shape[0]), target_2_] = 1


        logits_list_eq = []
        for k, logits in enumerate(logits_list):
            if k==0:
                continue
            logits = logits[idx]
            N, C = logits.shape
            not_ignored = self.class_weights_level_2[self.idx_list[k-2] if k-2>=0 else 0 :self.idx_list[k-1] if k-1 <len(self.idx_list) else None].view(1, C).repeat(N, 1) # fixed values
            over_prob = (torch.rand(logits.shape).cuda() > self.ignore_prob).float()
            weights = ((not_ignored + over_prob + is_gt[:,self.idx_list[k-2] if k-2>=0 else 0 :self.idx_list[k-1] if k-1 <len(self.idx_list) else None]) > 0).float()# for head classes, not_ignored=1; for tail classes, if it's trget or over prob, then weight=1,else weight =0
            logits = replace_masked_values(logits, weights, 1e-7)
            logits_list_eq.append(logits)

        probas_0 = F.softmax(logits_0[idx], dim=1)
        probas_1 = F.softmax(logits_list_eq[0], dim=1) * probas_0[:,0:1] #第2Head是'Skates'的 2 类
        probas_2 = F.softmax(logits_list_eq[1], dim=1) * probas_0[:,1:2] #第3Head是'Sharks'的 4 类
        probas_3 = F.softmax(logits_list_eq[2], dim=1) * probas_0[:,2:3]
        probas_4 = F.softmax(logits_list_eq[3], dim=1) * probas_0[:,3:4]
        probas_5 = F.softmax(logits_list_eq[4], dim=1) * probas_0[:,4:5]
        probas_6 = F.softmax(logits_list_eq[5], dim=1) * probas_0[:,5:6]
        probas_7 = F.softmax(logits_list_eq[5], dim=1) * probas_0[:, 6:7]
        probas_8 = F.softmax(logits_list_eq[5], dim=1) * probas_0[:, 7:8]
        probas_9 = F.softmax(logits_list_eq[5], dim=1) * probas_0[:, 8:9]
        probas_10 = F.softmax(logits_list_eq[5], dim=1) * probas_0[:, 9:]
        probas_level2 = torch.cat((probas_1, probas_2, probas_3, probas_4, probas_5, probas_6,probas_7, probas_8,probas_9,probas_10), dim=1)

        loss_level_2 = torch.nn.NLLLoss()(torch.log(probas_level2), target_2_)

        return loss_level_1, loss_level_2


def id_31_to_level_2_name(id_in_31):

    total_species_num = len(level_2_names)
    i=0
    for key in hierarchy_dict:
        groups = hierarchy_dict[key]

        for species in groups:
            if id_in_31==i:
                name = species
                return name
            else:
                i+=1
    if i==total_species_num:
        name='No level 2 labels'


    return name


# EQL_loss = SoftmaxEQL(lambda_1=20000, lambda_2=5000, ignore_prob=0.5, file_name='./labels_track_based/fish-rail-train.csv')
