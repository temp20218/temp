import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import math
import torch.nn.functional as F
from easydict import EasyDict as easydict
import json
from models.ResNetMRM2 import ResNetMRM2
from models.InceptionTimeMRM2 import InceptionTimeMRM2
from models.OS_CNNMRM2 import OS_CNNMRM2
from models.FCNMRM2 import FCNMRM2
from models.FCN_LSTMMRM2 import FCNLSTMMRM2
from OS_CNN_Structure_build import generate_layer_parameter_list
from utils import set_mrm2_parser, readuea, MultiTSDataset
import heapq



flist =[
        'ArticularyWordRecognition',
    'AtrialFibrillation',
        'BasicMotions',
    'CharacterTrajectories',
    'Cricket',
    'DuckDuckGeese',
        'EigenWorms',
    'Epilepsy',
    'EthanolConcentration',
        'ERing',
    'FaceDetection',
    'FingerMovements',
    'HandMovementDirection',
        'Handwriting',
        'Heartbeat',
    'InsectWingbeat',
    'JapaneseVowels',
        'Libras',
        'LSST',
    'MotorImagery',
        'NATOPS',
    'PenDigits',
    'PEMS-SF',
        'PhonemeSpectra',
        'RacketSports',
        'SelfRegulationSCP1',
    'SelfRegulationSCP2',
        'SpokenArabicDigits',
    'StandWalkJump',
    'UWaveGestureLibrary'
]
from collections import Counter
import random
lambda_gamma = [
    [0., 0.],
    # [0, 0.25],
    [0, 0.5],
    # [0, 0.75],
    [0, 1.0],

    [0.25, 0],
    # [0.25, 0.25],
    [0.25, 0.5],
    # [0.25, 0.75],
    [0.25, 1.0],

    [0.5, 0],
    # [0.5, 0.25],
    #     [0.5, 0.5],
    # [0.5, 0.75],
    [0.5, 1.0],

    [0.75, 0],
    # [0.75, 0.25],
    [0.75, 0.5],
    # [0.75, 0.75],
    [0.75, 1.0]
]

if __name__ == '__main__':
    train_config = set_mrm2_parser()
    data_dir = train_config.data_dir
    max_epoch = train_config.max_epoch
    out_dir = train_config.out_dir
    batch_size = train_config.batch_size
    # config = easydict(json.load(open('config.json','r')))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    Max_kernel_size = 89
    # paramenter_number_of_layer_list = [8 * 128, 5 * 128 * 256 + 2 * 256 * 128]
    start_kernel_size = 1

    flist = flist[train_config.start_dataset_id:]
    result = {'Name': [], 'val_loss': [], 'val_tcrm2_acc': [], 'val_ttrm2_acc': [],'val_tcrm2_acc_final': [], 'val_ttrm2_acc_final': [],'lambda':[],'gamma':[]}
    json.dump(train_config.to_dict(), open(os.path.join(out_dir, 'train_config.json'), 'w'), indent=4)
    device = torch.device('cuda:{}'.format(train_config.start_cuda_id)) \
        if train_config.gpu_nums > 0 and torch.cuda.is_available() else torch.device('cpu')

    origin_pred= pd.read_csv('./result_fixedR_803/fcn_lstm_mts0_0/Result_0.csv',index_col= 0)
    mrm2_pred  = pd.read_csv('./result_fixedR_803/fcn_lstm_mts/Result_0.csv',index_col=0)
    for fname in (flist):
        #         import pdb
        #         pdb.set_trace()
        if origin_pred[origin_pred['Name']==fname]['val_tcrm2_acc'].tolist()[0]<mrm2_pred[mrm2_pred['Name']==fname]['val_tcrm2_acc'].tolist()[0]:
            continue
        if origin_pred[origin_pred['Name']==fname]['val_tcrm2_acc'].tolist()[0]==1.0:
            continue

        for lambda_loss,gamma_loss in lambda_gamma:
            x_train, y_train = readuea(data_dir ,fname ,'TRAIN')
            x_test, y_test = readuea(data_dir , fname , 'TEST')
            x_train_num =int( x_train.shape[0])
            x_test_num = int(x_test.shape[0])
            x_channel = int(x_train.shape[1])

            print('train num is: {}, test num is: {}'.format(x_train.shape[0],x_test.shape[0]))
            y = np.concatenate([y_train, y_test], axis=0)
            nb_classes = len(np.unique(y_test))
            class_label = torch.eye(nb_classes, dtype=torch.int64, device=device, requires_grad=False)
            label2id = {}
            for i, key in enumerate(set(y)):
                label2id[key] = i
            y_test = np.array([label2id[key] for key in y_test])
            y_train = np.array([label2id[key] for key in y_train])
            train_label2idx = {}
            for i in range(len(y_train)):
                if y_train[i] not in train_label2idx:
                    train_label2idx[y_train[i]] = []
                train_label2idx[y_train[i]].append(i)
            if x_channel*x_train.shape[-1]>10000:
                batch_size = 32
            else:
                batch_size = 512

            batch_size_train, batch_size_val = int(x_train_num * 0.5),   batch_size -int(x_train_num * 0.5)
            if batch_size_train >= batch_size*0.5:
                batch_size_train, batch_size_val = int(batch_size * 0.5), batch_size - int(batch_size * 0.5)
            train_config.batch_size_train = batch_size_train

            train_is_train, train_ids = np.ones(x_train.shape[0]).astype(np.bool), np.arange(0, x_train.shape[0])
            test_dataset = MultiTSDataset(x_test, y_test, is_train=False)
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size_val,
                shuffle=True,
                num_workers=8,
                pin_memory=True
            )

            # device=torch.device('cpu')
            paramenter_number_of_layer_list =[8*128*x_channel, 5*128*256 + 2*256*128]

            if train_config.model_name.lower() == 'fcn':
                model = FCNMRM2(input_features_d=x_channel,n_feature_maps=128,nb_classes=nb_classes).to(device)
            elif train_config.model_name.lower() =='fcn_lstm':
                model= FCNLSTMMRM2(x_channel,128,nb_classes,8).to(device)
            elif train_config.model_name.lower()=='resnet':
                model = ResNetMRM2(x_channel,64,nb_classes).to(device)
            elif train_config.model_name.lower()=='inceptiontime':
                model= InceptionTimeMRM2(x_channel,nb_classes=nb_classes).to(device)
            else:
                receptive_field_shape = min(int(x_train.shape[-1] / 4), Max_kernel_size)
                layer_parameter_list = generate_layer_parameter_list(start_kernel_size,
                                                                     receptive_field_shape,
                                                                     paramenter_number_of_layer_list,
                                                                     in_channel=x_channel#int(x_train.shape[1])
                                                                     )
                model = OS_CNNMRM2(layer_parameter_list, nb_classes, False).to(device)
            # model = ResNetMRM2(1, 64, nb_classes)
            if len(train_config.pretrain_model_dir) != 0:
                # load pre-train model  parameters
                pretrain_model_path = os.path.join(train_config.pretrain_model_dir, '{}_best_model.pth'.format(fname))
                if os.path.exists(pretrain_model_path):
                    print('there have the path' ,pretrain_model_path)
                    encoder_dic = torch.load(pretrain_model_path, map_location='cpu')
                    encoder_dic = encoder_dic['state_dict'] if 'state_dict' in encoder_dic else encoder_dic
                    model_dict = model.state_dict()
                    encoder_dic = {k: v for k, v in encoder_dic.items() if
                                   (k in model_dict) and encoder_dic[k].shape == model_dict[k].shape}
                    model_dict.update(encoder_dic)
                    model.load_state_dict(model_dict, strict=False)

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr,
                                         weight_decay=train_config.weight_decay)


            criterion = nn.CrossEntropyLoss(reduction='none')
            bin_criterion_mean = nn.BCEWithLogitsLoss(reduction='mean')
            multi_criterion_mean = nn.CrossEntropyLoss(reduction='mean')
            # criterion_attn = nn.BCELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                                   verbose=True, min_lr=0.0001)
            # best_losses = []
            best_loss = 1e9
            # heapq.heapify(best_losses)
            final_res = []
            print('-------------- start training dataset : {} ---------------'.format(fname))
            train_weight = np.array(list(dict(sorted(Counter(y_train).items())).values()))/sum(list(dict(sorted(Counter(y_train).items())).values()))
            train_weight_pre_sum = [sum(train_weight[:l]) for l in range(1,len(train_weight)+1)]
            train_weight = torch.tensor(train_weight,requires_grad= True).to( torch.float32).unsqueeze(0).to(device)
            print(train_weight)
            def shuffle_choice(batch_size_train):
                sub_x_train_ids = []

                #         keys =list( train_label2idx.keys())
                while True:
                    p = random.random()

                    for y in range(nb_classes):
                        if p < train_weight_pre_sum[y]:

                            sub_x_train_ids.append(np.random.choice(train_label2idx[y], 1)[0])
                            if len(sub_x_train_ids) >= batch_size_train:
                                return np.array(sub_x_train_ids)
                            break

            # def shuffle_choice(batch_size_train):
            #     sub_x_train_ids = []
            #     while True:
            #         for y in train_label2idx:
            #             sub_x_train_ids.append(np.random.choice(train_label2idx[y], 1)[0])
            #             if len(sub_x_train_ids) >= batch_size_train:
            #                 return np.array(sub_x_train_ids)

            max_epoch = (train_config.max_epoch//(1+int(x_test_num*batch_size_train/(batch_size_val*x_train_num))))
            print('the max epoch is: ',max_epoch)
            for epoch in range(max_epoch):
                model.train()
                running_loss, valing_loss ,ttrm2_running_loss, ttrm2_valing_loss= 0.0, 0.0,0.0, 0.0
                start = time.time()
                train_pred, train_gt, test_pred, test_gt = [], [], [], []
                train_pred_ttrm2,test_pred_ttrm2 = [],[]

                for i, (xs, labels, is_train, ids) in (enumerate(test_dataloader, 0)):
                    optimizer.zero_grad()
                    attn_label = None

                    sub_x_train_ids = shuffle_choice(batch_size_train)
                    sub_xs_train = torch.from_numpy(x_train[sub_x_train_ids]).type(torch.float32)
                    #                 print(y_train[sub_x_train_ids])
                    sub_labels_train = torch.from_numpy(y_train[sub_x_train_ids])
                    sub_is_train_train = torch.from_numpy(train_is_train[sub_x_train_ids])
                    sub_ids_train = torch.from_numpy(sub_x_train_ids)

                    xs = torch.cat([xs, sub_xs_train], dim=0)
                    labels = torch.cat([labels, sub_labels_train], dim=0)
                    is_train = torch.cat([is_train, sub_is_train_train], dim=0)
                    ids = torch.cat([ids, sub_ids_train], dim=0)

                    xs, labels = xs.to(device), labels.to(device)
                    is_test = ~is_train

                    import copy
                    labels_copy = copy.deepcopy(labels)
                    labels_copy[is_test] = -1

                    out_all = model(xs, labels_copy)
                    out = out_all['tcrm2_out']

                    softmax_out = torch.softmax(out, dim=-1)
                    pred_prob, pred = softmax_out.max(dim=-1)
                    out_ttrm2 = out_all['ttrm2_out']
                    softmax_out_ttrm2 = torch.softmax(out_ttrm2,dim = -1)
                    pred_prob_ttrm2, pred_ttrm2 = softmax_out_ttrm2.max(dim=-1)

                    losses = criterion(out, labels)
                    losses_ttrm2 = criterion(out_ttrm2,labels)
                    loss = sum(losses[is_train]) / (sum(is_train.long())+1e-6) + \
                           lambda_loss * sum(losses_ttrm2[is_train]) / (sum(is_train.long())+1e-6)
                    if sum(is_train.long())!=0:
                        loss.backward()
                        optimizer.step()
                    running_loss += sum(losses[is_train]) / (sum(is_train.long())+1e-6).item()
                    ttrm2_running_loss+= sum(losses_ttrm2[is_train]) / (sum(is_train.long())+1e-6).item()

                    loss_val = sum(losses[is_test]) / (sum(is_test.long())+1e-6)
                    valing_loss += loss_val.item()
                    ttrm2_loss_val = sum(losses_ttrm2[is_test]) / (sum(is_test.long())+1e-6)
                    ttrm2_valing_loss += ttrm2_loss_val.item()

                    train_gt.append(labels[is_train].detach().to('cpu').numpy())
                    test_gt.append(labels[is_test].detach().to('cpu').numpy())
                    train_pred.append(pred[is_train].detach().to('cpu').numpy())
                    test_pred.append(pred[is_test].detach().to('cpu').numpy())
                    train_pred_ttrm2.append(pred_ttrm2[is_train].detach().to('cpu').numpy())
                    test_pred_ttrm2.append(pred_ttrm2[is_test].detach().to('cpu').numpy())


                train_pred = np.concatenate(train_pred, axis=0)
                train_gt = np.concatenate(train_gt, axis=0)
                test_gt = np.concatenate(test_gt, axis=0)
                test_pred = np.concatenate(test_pred, axis=0)
                train_pred_ttrm2 = np.concatenate(train_pred_ttrm2, axis=0)
                test_pred_ttrm2 = np.concatenate(test_pred_ttrm2, axis=0)

                acc = accuracy_score(y_true=train_gt, y_pred=train_pred)
                acc_val = accuracy_score(y_true=test_gt, y_pred=test_pred)  # sum(gt==pred)/gt.shape[0]
                acc_ttrm2 = accuracy_score(y_true=train_gt, y_pred=train_pred_ttrm2)
                acc_ttrm2_val = accuracy_score(y_true=test_gt, y_pred=test_pred_ttrm2)
                running_loss = running_loss / (i + 1)
                valing_loss = valing_loss / (i + 1)
                ttrm2_running_loss = ttrm2_running_loss / (i + 1)
                ttrm2_valing_loss = ttrm2_valing_loss / (i + 1)

                print('[%3d/ %3d]'
                      ' train_loss: %.4f'
                      ' train_ttrm2_loss: %.4f'
                      ' train_acc: %.4f'
                      ' train_ttrm2_acc: %.4f'
                      ' test_loss: %.4f'
                      ' test_ttrm2_loss: %.4f'
                      ' test_acc: %.4f'
                      ' test_ttrm2_acc: %.4f'
                      ' time: %f s' %
                      (epoch + 1, max_epoch,
                       running_loss,
                       ttrm2_running_loss,
                       acc,acc_ttrm2,
                       valing_loss,ttrm2_valing_loss, acc_val,acc_ttrm2_val, time.time() - start))


                scheduler.step(running_loss+ttrm2_running_loss)

                class_embed = torch.cat([model.linear.weight, model.linear.bias.unsqueeze(-1)], dim=1)
                mold_class_embed = torch.sqrt(torch.sum(class_embed * class_embed, dim=1, keepdim=True))
                mold_class_embedx = torch.matmul(mold_class_embed, mold_class_embed.T)
                alpha_angle = 3
                class_loss = gamma_loss * multi_criterion_mean(torch.matmul(class_embed, class_embed.t())/mold_class_embedx*alpha_angle,
                                                               torch.arange(0, nb_classes, dtype=torch.int64, device=device,
                                                                            requires_grad=False))

                optimizer.zero_grad()
                class_loss.backward()
                optimizer.step()
                if epoch > 0.3 * max_epoch:
                    if best_loss > running_loss+ttrm2_running_loss:
                        best_loss = running_loss+ttrm2_running_loss
                        torch.save(model.state_dict(), os.path.join(out_dir, '{}_best_model.pth'.format(fname)))
                        final_res = [valing_loss+ttrm2_valing_loss, acc_val, acc_ttrm2_val]


            pretrain_model_path = os.path.join(out_dir, '{}_best_model.pth'.format(fname))
            encoder_dic = torch.load(pretrain_model_path)
            encoder_dic = encoder_dic['state_dict'] if 'state_dict' in encoder_dic else encoder_dic
            model_dict = model.state_dict()
            encoder_dic = {k: v for k, v in encoder_dic.items() if
                           (k in model_dict) and encoder_dic[k].shape == model_dict[k].shape}
            model_dict.update(encoder_dic)
            model.load_state_dict(model_dict, strict=False)
            x_train_embeddings = []
            x_train_pred = []
            model.eval()
            for start in range(0,x_train_num,4):
                end = start+4
                in_tensor = torch.from_numpy(x_train[start:end]).type(torch.float32).to(device)
                out_all = model(in_tensor,None)
                x_train_embeddings.append(out_all['x_embeddings'].detach().cpu())
                x_train_pred.append(out_all['tcrm2_out'].detach().cpu())

            x_train_embeddings = torch.cat(x_train_embeddings, dim = 0) # n x128
            x_train_pred = torch.cat(x_train_pred, dim = 0)
            x_test_pred ,x_test_embeddings = [],[]
            for start in range(0,x_test.shape[0],4):
                end = start+4
                in_tensor = torch.from_numpy(x_test[start:end]).type(torch.float32).to(device)
                out_all = model(in_tensor,None)
                x_test_embeddings.append(out_all['x_embeddings'].detach().cpu())
                x_test_pred.append(out_all['tcrm2_out'].detach().cpu())

            x_test_embeddings = torch.cat(x_test_embeddings, dim = 0) # n x128
            x_test_pred = torch.cat(x_test_pred, dim = 0)
            x_embeddings= torch.cat([x_train_embeddings,x_test_embeddings],dim = 0)
            x_pred = torch.softmax(torch.cat([x_train_pred,x_test_pred],dim = 0),dim = 1)
            similar_x_test = torch.matmul(x_test_embeddings,x_embeddings.T)
            similar_x_test_stand = (similar_x_test-similar_x_test.mean(1,keepdim = True))/similar_x_test.std(1,keepdim = True)

            x_test_ttrm2_pred = torch.matmul((similar_x_test_stand),x_pred/x_pred.sum(dim = 0,keepdim = True))


            acc_tcrm2_val_final = accuracy_score(y_true=y_test, y_pred=x_test_pred.argmax(dim = 1))
            acc_ttrm2_val_final = accuracy_score(y_true=y_test, y_pred=x_test_ttrm2_pred.argmax(dim = 1))

            if len(result['Name'])>0 and result['Name'][-1]==fname:
                if result['val_tcrm2_acc'][-1]<=final_res[1]:
                    result['val_tcrm2_acc'][-1]= (final_res[1])
                    result['val_ttrm2_acc'][-1]= (final_res[2])
                    result['val_tcrm2_acc_final'][-1]= (acc_tcrm2_val_final)
                    result['val_ttrm2_acc_final'][-1]= (acc_ttrm2_val_final)
                    result['val_loss'][-1]= (final_res[0])
                    result['lambda'][-1]= (lambda_loss)
                    result['gamma'][-1]= (gamma_loss)

            else:
                result["Name"].append(fname)
                result['val_tcrm2_acc'].append(final_res[1])
                result['val_ttrm2_acc'].append(final_res[2])
                result['val_tcrm2_acc_final'].append(acc_tcrm2_val_final)
                result['val_ttrm2_acc_final'].append(acc_ttrm2_val_final)
                result['val_loss'].append(final_res[0])
                result['lambda'].append(lambda_loss)
                result['gamma'].append(gamma_loss)

            df_result = pd.DataFrame(result)
            df_result.to_csv(os.path.join(out_dir, 'Result_{}.csv'.format(train_config.start_dataset_id)))
