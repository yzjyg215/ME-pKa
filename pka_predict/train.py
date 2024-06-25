# encoding=utf-8
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from pKa_net import PkaNetBN4F20S21ELU, PkaNetBN4F20S21ELU_ONLYRES

import os
from pka_dataset import PkaDatasetCSV
from pka_criterion import criterion
from evaluate import evaluate_model
from tensorboardX import SummaryWriter
from fasta import load_fasta_info
import gc
import time


def train():
    """
    train model
    :return: None
    """
    dim = 640
    repr_layers = [30]
    res_only = False
    RMSE_chushi=10.0
    fasta_tensors = {}
    # torch.autograd.set_detect_anomaly(True)

    time1 = time.time()
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')  # if has cuda, use gpu divice, if not, use cpu divice.
    # train dataset relative files
    train_data_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_train_expand/finetune_train_pka.csv'
    train_center_coors_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_train_expand/finetune_train_PKAD2_all_center_coors.csv'
    train_protein_features_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_train_expand/finetune_train_data_pdb_WT_fixed_mol2.csv'
    # test dataset  relative files
    test_data_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_small/finetune_test_pka.csv'
    test_center_coors_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_small/finetune_test_PKAD2_all_center_coors.csv'
    test_protein_features_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_small/finetune_test_data_pdb_WT_fixed_mol2.csv'

    batch_size = 6   # mini batch size, how many data forward and backward every time.
    total_epoch = 500
    start_epoch = 0
    radii = 10
    train_info_save_dir = '/data2/rymiao/propka/save/info_last'
    train_info_save_path = os.path.join(train_info_save_dir, 'pka_train_info.txt')
    model_save_dir = '/data2/rymiao/propka/save/model_last'
    rotate_angle = 90
    is_rotate = False
    normalize = True
    fill_charge = 'grid charge'             # 'grid charge', 'box charge' or 'atom charge'
    load_weight_path = True
    # load_weight_path = './model/model_s21_relu_n252_f19_r4_atomcharge_adam_chimera/pka_net_epoch469.pt'

    mini_loss = 100000.0
    best_model_name = 'fasta_best_finetune.pt'
    if not os.path.exists(train_info_save_dir):
        os.mkdir(train_info_save_dir)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    # load model
    if res_only:
        net = PkaNetBN4F20S21ELU_ONLYRES()
    else:
        net = PkaNetBN4F20S21ELU(len(repr_layers), dim)
    if load_weight_path:
        # get best model loss
        # best_path = os.path.join(model_save_dir, 'pka_net_best_fine_tune.pt')
        best_path='/data2/rymiao/propka/save/fasta_model_xu/pka_net_epoch52.pt'
        net.load_state_dict(torch.load(best_path), strict=False)
        # net.eval()
        # mini_loss1, RMSE1, _, _ = evaluate_model(net=net, device=device, test_data_path=test_data_path,
        #                                      batch_size=batch_size, is_rotate=False, fill_charge=fill_charge,
        #                                      normalize=normalize, center_coors_path=test_center_coors_path,
        #                                      protein_features_path=test_protein_features_path, res_only=res_only,radii=radii)

        # # get loaded model loss
        # net.load_state_dict(torch.load(load_weight_path), strict=False)
        # net.eval()
        # mini_loss2, RMSE2, _, _ = evaluate_model(net=net, device=device, test_data_path=test_data_path,
        #                                      batch_size=batch_size, is_rotate=False, fill_charge=fill_charge,
        #                                      normalize=normalize, center_coors_path=test_center_coors_path,
        #                                      protein_features_path=test_protein_features_path, res_only=res_only,radii=radii)

        # # get mini loss
        # mini_loss = min(RMSE1, RMSE2)

    # set running device, set optim
    # for param in net.parameters():
    #     param.requires_grad = False
    # net.fc3 = nn.Linear(256,len(repr_layers))
    print(net)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.994)

    writer1 = SummaryWriter(train_info_save_dir)
    # writer1.add_graph(model=PkaNetBN2())
    # because set is_rotate 'True', so every time load a diffrent dataset.
    # (some data possible has be rotated.)
    train_dataset = PkaDatasetCSV(data_path=train_data_path, is_rotate=is_rotate, rotate_angle=rotate_angle,
                                  fill_charge=fill_charge, normalize=normalize, center_coors_path=train_center_coors_path,
                                  proteins_features_path=train_protein_features_path, radii=radii, is_train=True,res_only=res_only)
    test_dataset = PkaDatasetCSV(data_path=test_data_path, is_rotate=False, rotate_angle=rotate_angle,
                                 fill_charge=fill_charge, normalize=normalize, center_coors_path=test_center_coors_path,
                                 proteins_features_path=test_protein_features_path, radii=radii, is_train=False,res_only=res_only)
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(start_epoch, total_epoch):
        running_loss = 0.0
        net.train()  # set train model
        train_dataset.flash_batch_data()
        while True:
            for i, data in enumerate(trainloader):
                # get the inputs: data is a list of [input, labels]
                if res_only:
                    inputs,labels=data
                else:
                    inputs, labels, fastas = data
                pkas = labels[:, :1]
                name_idxes = labels[:, 1:]
                inputs = inputs.to(device)
                pkas = pkas.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # try:
                if res_only:
                    outputs = net(inputs)
                else:
                    fastas=load_fasta_info(fastas,False,repr_layers=repr_layers)
                    outputs=net(inputs,fastas)
                    # fasta_values = []
                    # for fasta in fastas:
                    #     if fasta in fasta_tensors.keys():
                    #         fasta_values.append(fasta_tensors[fasta])
                    #     else:
                    #         #fasta_value = torch.concat([item.sum(dim=-2).squeeze().view(-1).cpu() for item in load_fasta_info(fasta, False, repr_layers=repr_layers)])
                    #         fasta_value=load_fasta_info(fastas,False,repr_layers=repr_layers)
                    #         print(fasta_value.shape)
                    #         fasta_values.append(fasta_value)
                    #         fasta_tensors[fasta] = fasta_value
                    # print([f.shape for f in fasta_values])
                    # fastas = torch.stack(fasta_values)
                    # fastas = fastas.to(device)
                    # outputs = net(inputs, fastas)
                # except Exception as e:
                #     print(e)
                #     # print(inputs)
                #     print(inputs.shape)
                #     exit(0)
                loss = criterion(outputs, pkas)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss * pkas.shape[0]
                print('train: [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))

            # if there is any remaining batch data, load the remaining batch data.
            if not train_dataset.is_empty():
                train_dataset.batch_load_data()
            else:
                break
        # change lr
        # scheduler.step()
        # write training info in files.
        train_total_len = train_dataset.get_total_len()
        running_loss_train = running_loss / train_total_len
        train_info = 'train: [epoch {}] loss: {}'.format(epoch, running_loss_train).ljust(45, ' ')
        with open(train_info_save_path, 'a') as f:
            f.write(train_info)

        # test model

        test_dataset.flash_batch_data()
        running_loss_test, RMSE_test, R2_test, R2_test_shift,RMSE_BACK = evaluate_model(net=net, device=device,
                                                                              test_data_path=None,
                                                                              test_dataset=test_dataset,
                                                                              batch_size=batch_size,
                                                                              is_rotate=True,
                                                                              rotate_angle=rotate_angle,
                                                                              fill_charge=fill_charge,
                                                                              normalize=normalize,
                                                                              center_coors_path=test_center_coors_path,
                                                                              protein_features_path=test_protein_features_path,
                                                                              radii=radii,
                                                                              res_only=res_only,
                                                                              RMSERMSE=RMSE_chushi,
                                                                              repr_layers=repr_layers)
        test_info = 'test: [epoch {}] loss: {}\n'.format(epoch, running_loss_test).ljust(50, ' ')
        RMSE_chushi=RMSE_BACK
        with open(train_info_save_path, 'a') as f:
            f.write(test_info)

        # draw plot in tensorboard
        writer1.add_scalars('loss[MAE]', tag_scalar_dict={'train': running_loss_train, 'test': running_loss_test},
                            global_step=epoch)
        writer1.add_scalars('RMSE', tag_scalar_dict={'test': RMSE_test}, global_step=epoch)
        writer1.add_scalars('R2', tag_scalar_dict={'test': R2_test, 'test_shift': R2_test_shift}, global_step=epoch)

        # better epoch save model, and save best performance model
        epoch_model_save_name = 'pka_net_epoch{}.pt'.format(epoch)
        best_model_save_name = 'pka_net_best_fine_tune.pt'
        if mini_loss > RMSE_test:
            model_save_path = os.path.join(model_save_dir, epoch_model_save_name)
            torch.save(net.state_dict(), model_save_path)

        # save best epoch model as best model
        if mini_loss > RMSE_test:
            model_save_path = os.path.join(model_save_dir, best_model_save_name)
            torch.save(net.state_dict(), model_save_path)
            mini_loss = RMSE_test
            best_model_name = epoch_model_save_name

        print(gc.collect())
    time2 = time.time()
    print('use time: {} min'.format((time2 - time1) / 60))
    print('best_model_name: {}'.format(best_model_name))
    print('model save dir: {}'.format(model_save_dir))
    print('train_info save dir: {}'.format(train_info_save_dir))
    print('Finished Training')


if __name__ == '__main__':
    train()
