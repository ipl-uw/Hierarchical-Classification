# from Model_0 import resnet101

from Model_7 import resnet101
# from hierarchy_cls_train import model_save_path,train_loader,valid_loader,DEVICE,NUM_CLASSES, GRAYSCALE
import torch
import torch.nn as nn
import os
from util import compute_accuracy_model0,calculate_num_class,hierarchy_dict, \
    calculate_num_class_model0, compute_accuracy_model12, \
    compute_accuracy_model7_track_based, track_based_accuracy,\
    track_based_accuracy_majority_vote,Otsu_Threshold
from IPython import embed
from torchvision import transforms
from fish_rail_dataloader_track_based import Fish_Rail_Dataset
from torch.utils.data import DataLoader



GRAYSCALE = False
# NUM_CLASSES = calculate_num_class(hierarchy_dict)  #37
# NUM_CLASSES = calculate_num_class_model0(hierarchy_dict)  # model0   31
NUM_level_1_CLASSES,  NUM_level_2_CLASSES= calculate_num_class(hierarchy_dict)

# model_save_path = './checkpoints-model7-track_based-Eq loss 0.8 shark'
model_save_path = './checkpoints_plus_sleeper_shark_nonfish'
# model_save_path = './checkpoints-model6-track_based'
DEVICE =  'cuda:0'
BATCH_SIZE=1024 +512+256+256

# model-7
# save_path_val = './per img predictions val model7-track_based-Eq loss 0.8 shark'
# save_path_tr = './per img predictions tr model7-track_based-Eq loss 0.8 shark'
save_path_val = './per img predictions val plus_sleeper_shark_nonfish'
save_path_tr = './per img predictions tr plus_sleeper_shark_nonfish'
#model-6
# save_path_val = './per img predictions val model6-track_based 2nd'
# save_path_tr = './per img predictions tr model6-track_based 2nd'

custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       # transforms.CenterCrop((178, 178)),
                                       #transforms.Grayscale(),
                                       #transforms.Lambda(lambda x: x/255.),
                                       transforms.ToTensor()])


valid_gt_path = 'Z:/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-valid-plus_sleeper_shark_nonfish.csv'
train_gt_path = 'Z:/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/labels_track_based/fish-rail-train-plus_sleeper_shark_nonfish.csv'
img_dir = 'Z:/Jie Mei/rail data/hierarchy_data_for_Transformer-SVM/cropped_box_with_sleeper_shark_non_fish'

train_dataset = Fish_Rail_Dataset(csv_path=train_gt_path,
                              img_dir=img_dir,
                              transform=custom_transform,
                              hierarchy = hierarchy_dict)


valid_dataset = Fish_Rail_Dataset(csv_path=valid_gt_path,
                              img_dir=img_dir,
                              transform=custom_transform,
                              hierarchy = hierarchy_dict)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=0)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=0)


### load model
# best_epoch=138   #model-7
# best_epoch=135   #model-7  more
# best_epoch=90    #model-6
# best_epoch=126   #model-7  EQ
# best_epoch=87   #model-7  EQ 0.8
# best_epoch=86   #model-7  EQ 0.8 sleeper sharks
best_epoch=65     # nonfish + sleeper shark
stop_at_level_1_threshold=0.85
model = resnet101(NUM_level_1_CLASSES,  NUM_level_2_CLASSES, GRAYSCALE)
PATH = os.path.join(model_save_path,'parameters_epoch_'+str(best_epoch)+'.pkl')
model.load_state_dict(torch.load(PATH))
model.to(DEVICE)


### 最后测试一下  for model7
model.eval()

# for model7
with torch.set_grad_enabled(False):  # save memory during inference

    ### evaluate train data
    # print('evaluating train data...')
    # avg_level_1_acc_tr, avg_level_2_acc_tr, avg_level_2_acc_p1p2_31_tr, avg_level_2_acc_p1p2_maxmax_tr, \
    # acc_1_tr, acc_2_tr, acc_2_p1p2_31_tr, acc_2_p1p2_maxmax_tr, avg_acc_can_stop_level_1_tr, all_num_level_1_tr, all_num_level_2_tr, species_stop_at_level_1_tr = compute_accuracy_model7_track_based(
    #     model, train_loader, best_epoch, DEVICE, save_path_tr, stop_at_level_1_threshold)
    #
    # ##根据记录下来的confidence，计算tarck-based的accuracy
    # avg_level_1_acc_tr_track, avg_level_2_acc_tr_track, avg_level_2_acc_p1p2_31_tr_track, avg_level_2_acc_p1p2_maxmax_tr_track, \
    # acc_1_tr_track, acc_2_tr_track, acc_2_p1p2_31_tr_track, acc_2_p1p2_maxmax_tr_track, avg_acc_can_stop_level_1_tr_track, all_num_level_1_tr_track, all_num_level_2_tr_track, species_stop_at_level_1_tr_track = \
    #     track_based_accuracy(save_path_tr, best_epoch)
    #
    # print(
    #     'Track-based Epoch: %03d | Train: Level-1 Avg: %.3f%%,  Level-2 Avg: %.3f%%,  Level-2 Avg p1p2 max out of 31: %.3f%%, Level-2 Avg p1p2 maxmax: %.3f%%, , Level-2 Avg can stop at level-1: %.3f%%, num level-1: %d, num level-2: %d' % (
    #         best_epoch,
    #         # avg_level_1_acc_tr * 100,
    #         # avg_level_2_acc_tr * 100,
    #         # avg_level_2_acc_p1p2_31_tr * 100,
    #         # avg_level_2_acc_p1p2_maxmax_tr * 100,
    #         # avg_acc_can_stop_level_1_tr * 100,
    #         # all_num_level_1_tr,
    #         # all_num_level_2_tr,
    #         avg_level_1_acc_tr_track * 100,
    #         avg_level_2_acc_tr_track * 100,
    #         avg_level_2_acc_p1p2_31_tr_track * 100,
    #         avg_level_2_acc_p1p2_maxmax_tr_track * 100,
    #         avg_acc_can_stop_level_1_tr_track * 100,
    #         all_num_level_1_tr_track,
    #         all_num_level_2_tr_track
    #     ))
    #
    # print('Track-based Individual accuracy: Train: '
    #       'Level-1:', acc_1_tr_track,
    #       'Level-2:', acc_2_tr_track,
    #       'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_tr_track,
    #       'Level-2 p1p2 maxmax:', acc_2_p1p2_maxmax_tr_track,
    #       'species stop at level1:', species_stop_at_level_1_tr_track)
    #
    # print(
    #     'Image-based Epoch: %03d | Train: Level-1 Avg: %.3f%%,  Level-2 Avg: %.3f%%,  Level-2 Avg p1p2 max out of 31: %.3f%%, Level-2 Avg p1p2 maxmax: %.3f%%, , Level-2 Avg can stop at level-1: %.3f%%, num level-1: %d, num level-2: %d' % (
    #         best_epoch,
    #         # avg_level_1_acc_tr * 100,
    #         # avg_level_2_acc_tr * 100,
    #         # avg_level_2_acc_p1p2_31_tr * 100,
    #         # avg_level_2_acc_p1p2_maxmax_tr * 100,
    #         # avg_acc_can_stop_level_1_tr * 100,
    #         # all_num_level_1_tr,
    #         # all_num_level_2_tr,
    #         avg_level_1_acc_tr * 100,
    #         avg_level_2_acc_tr * 100,
    #         avg_level_2_acc_p1p2_31_tr * 100,
    #         avg_level_2_acc_p1p2_maxmax_tr * 100,
    #         avg_acc_can_stop_level_1_tr * 100,
    #         all_num_level_1_tr,
    #         all_num_level_2_tr
    #     ))
    #
    # print('Image-based Individual accuracy: Train: '
    #       'Level-1:', acc_1_tr,
    #       'Level-2:', acc_2_tr,
    #       'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_tr,
    #       'Level-2 p1p2 maxmax:', acc_2_p1p2_maxmax_tr,
    #       'species stop at level1:', species_stop_at_level_1_tr)


    ### Otsu Threshold by using training data,读取记录下来的confidence，计算threshold
    # Otsu_Threshold_level_1 = Otsu_Threshold(save_path_tr, best_epoch)



    ### evaluate valid data
    print('evaluating valid data...')
    avg_level_1_acc_val, avg_level_2_acc_val, avg_level_2_acc_p1p2_31_val, avg_level_2_acc_p1p2_maxmax_val, \
    acc_1_val, acc_2_val, acc_2_p1p2_31_val, acc_2_p1p2_maxmax_val, avg_acc_can_stop_level_1_val, all_num_level_1_val, all_num_level_2_val, species_stop_at_level_1_val = compute_accuracy_model7_track_based(
        model, valid_loader, best_epoch, DEVICE, save_path_val, stop_at_level_1_threshold)


    ##根据记录下来的confidence，计算tarck-based的accuracy
    print('avg conf video-based method: ')
    avg_level_1_acc_val_track, avg_level_2_acc_val_track, avg_level_2_acc_p1p2_31_val_track, avg_level_2_acc_p1p2_maxmax_val_track, \
    acc_1_val_track, acc_2_val_track, acc_2_p1p2_31_val_track, acc_2_p1p2_maxmax_val_track, avg_acc_can_stop_level_1_val_track, all_num_level_1_val_track, all_num_level_2_val_track, species_stop_at_level_1_val_track= \
        track_based_accuracy(save_path_val, best_epoch, stop_at_level_1_threshold)

    print(
        'Track-based Epoch: %03d | Valid: Level-1 Avg: %.3f%%,  Level-2 Avg: %.3f%%,  Level-2 Avg p1p2 max out of 31: %.3f%%, Level-2 Avg p1p2 maxmax: %.3f%%, Level-2 can stop at level-1: %.3f%%, num level-1: %d, num level-2: %d' % (
            best_epoch ,
            # avg_level_1_acc_tr * 100,
            # avg_level_2_acc_tr * 100,
            # avg_level_2_acc_p1p2_31_tr * 100,
            # avg_level_2_acc_p1p2_maxmax_tr * 100,
            # avg_acc_can_stop_level_1_tr * 100,
            # all_num_level_1_tr,
            # all_num_level_2_tr,
            avg_level_1_acc_val_track * 100,
            avg_level_2_acc_val_track * 100,
            avg_level_2_acc_p1p2_31_val_track * 100,
            avg_level_2_acc_p1p2_maxmax_val_track * 100,
            avg_acc_can_stop_level_1_val_track * 100,
            all_num_level_1_val_track,
            all_num_level_2_val_track
        ))

    print('Track-based Individual accuracy: Valid: '
          'Level-1:', acc_1_val_track,
          'Level-2:', acc_2_val_track,
          'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_val_track,
          'Level-2 p1p2 maxmax:', acc_2_p1p2_maxmax_val_track,
          'species stop at level-1(avg conf):', species_stop_at_level_1_val_track)


    print('Majority vote video-based method: ')
    avg_level_1_acc_val_track, avg_level_2_acc_val_track, avg_level_2_acc_p1p2_31_val_track, avg_level_2_acc_p1p2_maxmax_val_track, \
    acc_1_val_track, acc_2_val_track, acc_2_p1p2_31_val_track, acc_2_p1p2_maxmax_val_track, avg_acc_can_stop_level_1_val_track, all_num_level_1_val_track, all_num_level_2_val_track, species_stop_at_level_1_val_track = \
        track_based_accuracy_majority_vote(save_path_val, best_epoch)

    print('Track-based Epoch: %03d | Valid: Level-1 Avg: %.3f%%,  Level-2 Avg: %.3f%%,  Level-2 Avg p1p2 max out of 31: %.3f%%, Level-2 Avg p1p2 maxmax: %.3f%%, Level-2 can stop at level-1: %.3f%%, num level-1: %d, num level-2: %d' % (
            best_epoch,
            # avg_level_1_acc_tr * 100,
            # avg_level_2_acc_tr * 100,
            # avg_level_2_acc_p1p2_31_tr * 100,
            # avg_level_2_acc_p1p2_maxmax_tr * 100,
            # avg_acc_can_stop_level_1_tr * 100,
            # all_num_level_1_tr,
            # all_num_level_2_tr,
            avg_level_1_acc_val_track * 100,
            avg_level_2_acc_val_track * 100,
            avg_level_2_acc_p1p2_31_val_track * 100,
            avg_level_2_acc_p1p2_maxmax_val_track * 100,
            avg_acc_can_stop_level_1_val_track * 100,
            all_num_level_1_val_track,
            all_num_level_2_val_track
        ))

    print('Track-based Individual accuracy: Valid: '
          'Level-1:', acc_1_val_track,
          'Level-2:', acc_2_val_track,
          'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_val_track,
          'Level-2 p1p2 maxmax:', acc_2_p1p2_maxmax_val_track,
          'species stop at level-1:', species_stop_at_level_1_val_track)




    print(
        'Image-based Epoch: %03d | Valid: Level-1 Avg: %.3f%%,  Level-2 Avg: %.3f%%,  Level-2 Avg p1p2 max out of 31: %.3f%%, Level-2 Avg p1p2 maxmax: %.3f%%, , Level-2 Avg can stop at level-1: %.3f%%, num level-1: %d, num level-2: %d' % (
            best_epoch,
            # avg_level_1_acc_tr * 100,
            # avg_level_2_acc_tr * 100,
            # avg_level_2_acc_p1p2_31_tr * 100,
            # avg_level_2_acc_p1p2_maxmax_tr * 100,
            # avg_acc_can_stop_level_1_tr * 100,
            # all_num_level_1_tr,
            # all_num_level_2_tr,
            avg_level_1_acc_val * 100,
            avg_level_2_acc_val * 100,
            avg_level_2_acc_p1p2_31_val * 100,
            avg_level_2_acc_p1p2_maxmax_val * 100,
            avg_acc_can_stop_level_1_val * 100,
            all_num_level_1_val,
            all_num_level_2_val
        ))

    print('Image-based Individual accuracy: Valid: '
          'Level-1:', acc_1_val,
          'Level-2:', acc_2_val,
          'Level-2 p1p2 max out of 31:', acc_2_p1p2_31_val,
          'Level-2 p1p2 maxmax:', acc_2_p1p2_maxmax_val,
          'species stop at level1:', species_stop_at_level_1_val)






embed()