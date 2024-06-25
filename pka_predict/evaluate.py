import torch
from torch.utils.data.dataloader import DataLoader
from pKa_net import PkaNetBN4F20S21ELU_ONLYRES
from pKa_net import PkaNetBN4F20S21ELU
from pka_dataset import PkaDatasetCSV, PkaDatasetHDF
from pka_criterion import criterion, criterion2
from tensorboardX import SummaryWriter
from pandas import DataFrame
from fasta import load_fasta_info
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import json

model_pka = {
    'ASP': 3.7,
    'GLU': 4.2,
    'HIS': 6.34,
    'LYS': 10.4,
    'CYS': 8.5,
}


def eval_result(in_csv_df):
    """

    :param csv_path:
    :return: None
    """
    csv_df = in_csv_df.copy()
    # pka
    csv_df.loc[:, 'P T sub'] = csv_df['Predict pKa'] - csv_df['Target pKa']
    csv_df.loc[:, 'P T sub abs'] = csv_df['P T sub'].abs()
    target_mean = csv_df['Target pKa'].mean()
    csv_df.loc[:, 'T T_mean sub'] = csv_df['Target pKa'] - target_mean
    csv_df.loc[:, 'P T sub square'] = csv_df['P T sub'].map(lambda x: x ** 2)
    csv_df.loc[:, 'T T_mean sub square'] = csv_df['T T_mean sub'].map(lambda x: x ** 2)
    csv_df.loc[:, 'P T sub square'] = csv_df['P T sub'].map(lambda x: x ** 2)

    MAE = csv_df['P T sub abs'].mean()
    RMSE = math.sqrt(csv_df['P T sub square'].mean())
    SSE = csv_df['P T sub square'].mean()
    SST = csv_df['T T_mean sub square'].mean()
    R2 = 1 - SSE / SST

    # pka shift
    csv_df.loc[:, 'P T sub shift'] = csv_df['Predict pKa shift'] - csv_df['Target pKa shift']
    csv_df.loc[:, 'P T sub abs shift'] = csv_df['P T sub shift'].abs()
    target_mean_shift = csv_df['Target pKa shift'].mean()
    csv_df.loc[:, 'T T_mean sub shift'] = csv_df['Target pKa shift'] - target_mean_shift
    csv_df.loc[:, 'P T sub square shift'] = csv_df['P T sub shift'].map(lambda x: x ** 2)
    csv_df.loc[:, 'T T_mean sub square shift'] = csv_df['T T_mean sub shift'].map(lambda x: x ** 2)
    csv_df.loc[:, 'P T sqrt sub shift'] = csv_df['P T sub shift'].map(lambda x: x ** 2)

    SSE_shift = csv_df['P T sub square shift'].mean()
    SST_shift = csv_df['T T_mean sub square shift'].mean()
    R2_shift = 1 - (SSE_shift / SST_shift)

    return MAE, RMSE, R2, R2_shift


def save_predict_to_csv(file_name_list, predict_pka_list, target_pka_list, csv_path):
    """
    this function will read predict information and label information, then saved the information in csv files.
    :param file_name_list: List[String, ...], each file name's form like (PDB Id)_(Chain)_(Ori Res Id)_(Res Name)_(New Res Id).
    :param predict_pka_list: List[Float, ...], each pka is predicted by model.
    :param target_pka_list: List[Float, ...], each pka is target value.
    :return:None
    """
    pdb_id_list = [file_name.split('_')[0] for file_name in file_name_list]
    chain_list = [file_name.split('_')[1] for file_name in file_name_list]
    ori_res_id_list = [int(file_name.split('_')[2]) for file_name in file_name_list]
    res_name_list = [file_name.split('_')[3] for file_name in file_name_list]
    global model_pka
    predict_dict = {
        'PDB ID': pdb_id_list,
        'Chain': chain_list,
        'Res ID': ori_res_id_list,
        'Res Name': res_name_list,
        'Predict pKa': predict_pka_list,
        'Target pKa': target_pka_list
    }
    # js=json.dumps(predict_dict)
    # file=open('/data2/rymiao/propka/pka_predict/temp_result.txt','w')
    # file.write(js)
    # file.close()
    predict_df = DataFrame(predict_dict)
    predict_df['model pKa'] = predict_df['Res Name'].apply(lambda x: model_pka[x])
    predict_df['Predict pKa shift'] = predict_df['Predict pKa'] - predict_df['model pKa']
    predict_df['Target pKa shift'] = predict_df['Target pKa'] - predict_df['model pKa']
    predict_df.to_csv(csv_path, index=False)


def draw_predict_target_scatter(choose_res_names, csv_path=None, csv_df=None, save_dir=None, save_name_suffix=''):
    """
    This function will read information form csv file, then draw Predict pKa - Target pKa plot, the csv file must
    contain columns ['PDB ID', 'Chain', 'Res ID', 'Res Name', 'Predict pKa', 'Target pKa'].
    :param csv_path: String, The path of csv file, the csv file contain predict information.
    :param choose_res_name: String, Choosed residue name should be draw, residue name must in
                            ['ASP', 'GLU', 'HIS', 'CYS', 'LYS']
    :param save_dir: Stirng, the save directory of draw plot, if None, will not saved.
    :param save_name_suffix: String, It is suffix of scatter save name.
    :return: None.
    """
    res_color = {
        'ASP': 'red',
        'GLU': 'gold',
        'HIS': 'green',
        'LYS': 'blue',
        'CYS': 'purple'
    }

    if csv_df is None:
        predict_df = pd.read_csv(csv_path)
    elif csv_path is None:
        predict_df = csv_df
        csv_path = 'temp.csv'
    else:
        raise Exception('both of csv_df and csv_path are not None, one of them must be None!')

    ax = None
    title = '{} : {}'.format(csv_path.split('/')[-1], choose_res_names)
    for res_name in choose_res_names:
        predict_df_temp = predict_df.loc[predict_df['Res Name'] == res_name]
        ax = predict_df_temp.plot.scatter(x='Target pKa', y='Predict pKa', marker='o', alpha=0.5, title=title,
                                          legend=True, c=res_color[res_name], ax=ax)
    plt.plot(range(-1, 16), range(-1, 16), linestyle='--', color='r')
    plt.axis([-1, 15, -1, 15])
    plt.axes().set_aspect('equal')
    save_name = '{}_{}{}.jpg'.format(csv_path.split('/')[-1].split('.')[0], '_'.join(choose_res_names),
                                     save_name_suffix)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.show()

    ax = None
    title = '{} shift : {}'.format(csv_path.split('/')[-1], choose_res_names)
    for res_name in choose_res_names:
        predict_df_temp = predict_df.loc[predict_df['Res Name'] == res_name]
        ax = predict_df_temp.plot.scatter(x='Target pKa shift', y='Predict pKa shift', marker='o', alpha=0.5,
                                          title=title,
                                          legend=True, c=res_color[res_name], ax=ax)
    plt.plot(range(-6, 8), range(-6, 8), linestyle='--', color='r')
    plt.axis([-6, 7, -6, 7])
    plt.axes().set_aspect('equal')
    save_name = '{}_{}{}_shift.jpg'.format(csv_path.split('/')[-1].split('.')[0], '_'.join(choose_res_names),
                                           save_name_suffix)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    print(save_path)
    plt.show()


def model_pka_predict(test_data_path, is_rotate=False):
    test_dataset = PkaDatasetHDF(data_path=test_data_path, is_rotate=is_rotate)
    testloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    total_loss = 0
    while True:
        for i, data in enumerate(testloader):
            inputs, labels = data

            pkas = labels[:, :1]
            name_idxes = labels[:, 1:]
            model_pkas = np.array(name_idxes)
            model_pkas = DataFrame(model_pkas)
            model_pkas = model_pkas.applymap(lambda x: test_dataset.batch_names[int(x)].split('_')[3])
            model_pkas = model_pkas.applymap(lambda x: model_pka[x])
            model_pkas = model_pkas.to_numpy()
            model_pkas = torch.from_numpy(model_pkas)
            loss = criterion(pkas, model_pkas)
            total_loss += loss * pkas.shape[0]
            print('model pka: {}  labels: {}  distance: {}  loss: {}  file_name: {}'
                  .format(model_pkas[0][0], pkas[0][0], (pkas - pkas)[0][0], loss,
                          [test_dataset.batch_names[int(idx)] for idx in name_idxes[:, 0]]))

        if not test_dataset.is_empty():
            test_dataset.batch_load_data()
        else:
            break
    test_total_len = test_dataset.get_total_len()
    mean_loss = total_loss / test_total_len
    print('model_predict -> mean loss: {}'.format(mean_loss))
    return mean_loss


def evaluate_model(net, device, batch_size, test_data_path=None, test_dataset=None, is_rotate=True, rotate_angle=90,
                   fill_charge='grid charge', normalize=True, center_coors_path=None, protein_features_path=None,
                   csv_path='.temp.csv', radii=10, res_only=False,RMSERMSE=10, repr_layers=[-1]):
    """
    use model to predict
    :return: None
    """

    # set test model, set running device
    net = net.eval()  # test model will close drop out layer and normalize layer
    net = net.to(device)

    # load dataset
    if test_data_path is not None:
        test_dataset = PkaDatasetCSV(data_path=test_data_path, is_rotate=is_rotate, rotate_angle=rotate_angle,
                                     fill_charge=fill_charge, normalize=normalize,
                                     center_coors_path=center_coors_path,
                                     proteins_features_path=protein_features_path, radii=radii, is_train=False, res_only=res_only)
    elif test_dataset is None:
        raise Exception('One of test_data_path and dataset must not be None.')

    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    file_name_list = []
    predict_pka_list = []
    target_pke_list = []
    total_loss = 0
    test_dataset.flash_batch_data()
    with torch.no_grad():
        while True:
            for i, data in enumerate(testloader):
                # get the inputs: data is a list of [input, labels]
                if res_only:
                    inputs,labels=data
                else:
                    inputs, labels, fastas = data

                pkas = labels[:, :1]
                name_idxes = labels[:, 1:]
                inputs = inputs.to(device)
                pkas = pkas.to(device)

                net = net.eval()
                # forward
                # print(inputs[0, 12, 8, 11, 10])
                if res_only:
                    outputs = net(inputs)
                else:
                    # fastas = [
                    #     torch.concat([item.view(-1).cpu() for item in load_fasta_info(fasta, False, repr_layers=repr_layers)[0]])
                    #     for fasta in fastas
                    # ]
                    # fastas = torch.stack(fastas).to(device)
                    fastas=load_fasta_info(fastas,False,repr_layers=repr_layers)
                    # print(inputs.detach().cpu().numpy().tolist())
                    # print(fastas.detach().cpu().numpy().tolist())
                    # exit()
                    outputs = net(inputs, fastas)
                loss = criterion(outputs, pkas)

                print('output: {}  labels: {}  distance: {}  loss: {}  file_name: {}'
                      .format(outputs[0][0], pkas[0][0], (outputs - pkas)[0][0], loss,
                              [test_dataset.batch_names[int(idx)] for idx in name_idxes[:, 0]]))
                total_loss += loss * pkas.shape[0]

                # save info in list
                file_name_list += [test_dataset.batch_names[int(idx)] for idx in name_idxes.view(-1)]
                predict_pka_list += [pka.item() for pka in outputs.view(-1)]
                target_pke_list += [pka.item() for pka in pkas.view(-1)]
            if not test_dataset.is_empty():
                test_dataset.batch_load_data()
            else:
                break

    save_predict_to_csv(file_name_list, predict_pka_list, target_pke_list, csv_path)
    csv_df = pd.read_csv(csv_path)
    MAE, RMSE, R2, R2_shift = eval_result(csv_df)
    if RMSE < RMSERMSE:
        csv_df.to_excel('/data2/rymiao/propka/save/result/best_valid_test.xlsx')
        RMSERMSE = RMSE

    test_total_len = test_dataset.get_total_len()
    mean_loss = total_loss / test_total_len
    print('mean loss: {}'.format(mean_loss))
    print('MAE: {}, RMSE: {}, R2: {}, R2_shift: {}'.format(MAE, RMSE, R2, R2_shift))
    return MAE, RMSE, R2, R2_shift, RMSERMSE


def evaluate_all_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_rotate = False
    normalize = True
    fill_charge = False
    batch_size = 128
    load_model_dir = '/home/czt/project/Predict_pKa/model/model_bn4_elu_n228_f19_r4_zscore_adam'
    train_info_save_dir = '/home/czt/project/Predict_pKa/train_info/model_bn4_elu_n228_f19_r4_zscore_adam'

    test_data_path = '/home/czt/data/pka_data_new/final_val_data_new/val_f19_r4_incphmd.csv'
    test_center_coors_path = '/home/czt/data/pka_data_new/final_val_data_new/final_expt_pka_center_coors.csv'
    test_protein_features_path = '/home/czt/data/pka_data_new/final_val_data_new/data_pdb_WT_fixed.csv'

    net = PkaNetBN4F20S21ELU()

    model_name_list = os.listdir(load_model_dir)
    mini_loss = 10000
    best_model_path = ''
    model_name_list.remove('pka_net_best.pt')
    epochs = [int(name.split('.pt')[0].split('epoch')[1]) for name in model_name_list]
    epochs.sort()
    writer1 = SummaryWriter(train_info_save_dir)

    data_set1 = PkaDatasetCSV(data_path=test_data_path, is_rotate=False, fill_charge=fill_charge,
                                 normalize=normalize, center_coors_path=test_center_coors_path,
                                 proteins_features_path=test_protein_features_path)

    for epoch in epochs:
        print('epoch : {}'.format(epoch))
        model_path = os.path.join(load_model_dir, 'pka_net_epoch{}.pt'.format(epoch))
        net.load_state_dict(torch.load(model_path), strict=True)
        loss, RMSE, R2, R2_shift = evaluate_model(net=net, device=device, test_data_path=None, test_dataset=data_set1,
                                                  batch_size=batch_size, is_rotate=is_rotate, normalize=True,
                                                  fill_charge=True, center_coors_path=test_center_coors_path,
                                                  protein_features_path=test_protein_features_path, res_only=res_only, repr_layers=[0, 29, 30])

        writer1.add_scalars('R2', tag_scalar_dict={'test': R2, 'test_shift': R2_shift}, global_step=epoch)
        writer1.add_scalars('MAE', tag_scalar_dict={'test': loss}, global_step=epoch)
        writer1.add_scalars('RMSE', tag_scalar_dict={'test': RMSE}, global_step=epoch)

        if RMSE <= mini_loss:
            mini_loss = RMSE
            best_model_path = model_path
    print('best model path: {}\nmini loss: {}'.format(best_model_path, mini_loss))


def evaluate_single_model(res_only=False):
    dim = 640
    repr_layers = [30]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # test_data_path = '../data/model_input/final_val_data/val_n27_f20_n4.csv'
    # test_center_coors_path = '../data/model_input/final_val_data/CpHMD_pka27_center_coors.csv'
    # test_protein_features_path = '../data/model_input/final_val_data/data_pdb_CpHMD27_fixed_mol2.csv'
    test_data_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_small/finetune_test_pka.csv'
    test_center_coors_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_small/finetune_test_PKAD2_all_center_coors.csv'
    test_protein_features_path = '/data2/rymiao/propka/data/fine_tune/fine_tune_small/finetune_test_data_pdb_WT_fixed_mol2.csv'
    batch_size = 6
    is_rotate = True
    rotate_angle = 90
    normalize = True
    radii = 10
    fill_charge = 'grid charge'
    model_dir_name = '/data2/rymiao/propka/save/model_test'
    model_dir = os.path.join('./model/', model_dir_name)
    load_weight_path = os.path.join(model_dir, 'pka_net_epoch1.pt')
    val_data_name = test_data_path.split('/')[-1]
    save_csv_path = os.path.join('./train_info/', model_dir_name, val_data_name)
    #save_plot_dir = os.path.join('./train_info/', model_dir_name)
    choose_draw_res_names = ['ASP', 'HIS', 'GLU', 'LYS']

    # load model
    net = PkaNetBN4F20S21ELU(len(repr_layers), dim)
    if load_weight_path:
        net.load_state_dict(torch.load(load_weight_path), strict=True)
    net.eval()


    evaluate_model(net=net, device=device, test_data_path=test_data_path, normalize=normalize, fill_charge=fill_charge,
                   center_coors_path=test_center_coors_path, protein_features_path=test_protein_features_path,
                   batch_size=batch_size, is_rotate=is_rotate, rotate_angle=rotate_angle, csv_path=save_csv_path,
                   radii=radii, res_only=res_only, repr_layers=[30])

    # draw plot
    # draw_predict_target_scatter(csv_path=save_csv_path, choose_res_names=choose_draw_res_names, save_dir=save_plot_dir, save_name_suffix='')


def evaluate_CpHMD_result():
    CpHMD_csv_path = '/media/huang/My Passport/czt/data/pka_data_new/cphmd_csv/CpHMD_pka_WT69.csv'
    WT_cleaned_csv_path = '/media/huang/My Passport/czt/data/pka_data_new/expt_cleaned_csv/final_expt_pka.csv'
    save_merge_csv_path = 'CpHMD_predict_info/CpHMD_predict_WT69.csv'
    cphmd_df = pd.read_csv(CpHMD_csv_path)
    wt_df = pd.read_csv(WT_cleaned_csv_path)
    wt_df = wt_df.rename(columns={'pKa': 'Target pKa'})
    cphmd_df = cphmd_df.rename(columns={'pKa': 'Predict pKa'})
    cphmd_df = cphmd_df.drop(columns=['Chain'])
    cphmd_predict_df = pd.merge(cphmd_df, wt_df)
    global model_pka
    cphmd_predict_df['model pKa'] = cphmd_predict_df['Res Name'].apply(lambda x: model_pka[x])
    cphmd_predict_df['Target pKa shift'] = cphmd_predict_df['Target pKa'] - cphmd_predict_df['model pKa']
    cphmd_predict_df['Predict pKa shift'] = cphmd_predict_df['Predict pKa'] - cphmd_predict_df['model pKa']
    cphmd_predict_df.to_csv(save_merge_csv_path, index=False)
    choose_draw_res_names = ['ASP', 'HIS', 'GLU', 'LYS']

    # draw relative scatter
    save_plot_dir = 'CpHMD_predict_info'
    # draw_predict_target_scatter(csv_path=save_merge_csv_path, choose_res_names=choose_draw_res_names,
    #                             save_dir=save_plot_dir)

    # calulate ASP GLU HIS LYS result
    merge_csv_df = pd.read_csv(save_merge_csv_path)

    # choose residue in XXXX.csv
    temp_csv_path = '../data/model_input/final_val_data/val_chimera_f19_r4_incphmd_undersample.csv'
    temp_csv_df = pd.read_csv(temp_csv_path)
    temp_csv_df = temp_csv_df.loc[:, ['file_name']].drop_duplicates()
    temp_csv_df['PDB ID'] = temp_csv_df['file_name'].apply(lambda x: x.split('_')[0])
    temp_csv_df['Res ID'] = temp_csv_df['file_name'].apply(lambda x: int(x.split('_')[2]))
    temp_csv_df['Res Name'] = temp_csv_df['file_name'].apply(lambda x: x.split('_')[3])
    temp_csv_df = temp_csv_df.loc[:, ['PDB ID', 'Res ID', 'Res Name']]
    print(temp_csv_df.count())
    merge_csv_df = pd.merge(merge_csv_df, temp_csv_df)

    merge_csv_path = 'CpHMD_predict_info/val_chimera_f19_r4_incphmd_undersample.csv'
    merge_csv_df.to_csv(merge_csv_path, index=False)
    print(merge_csv_df.count())

    draw_predict_target_scatter(csv_path=merge_csv_path, choose_res_names=choose_draw_res_names,
                                save_dir=save_plot_dir, save_name_suffix='_WT69')

    choose_resdues = ['ASP', 'GLU', 'HIS', 'LYS']
    merge_csv_df['choose'] = merge_csv_df['Res Name'].map(lambda x: x in choose_resdues)
    merge_csv_df = merge_csv_df.loc[merge_csv_df['choose']]
    MAE, RMSE, R2, R2_shift = eval_result(merge_csv_df)
    print('choose ASP, GLU, HIS, LYS')
    print('MAE: {}, RMSE: {}, R2: {}, R2_shift: {}'.format(MAE, RMSE, R2, R2_shift))

    # find shift large point
    # merge_csv_df3 = merge_csv_df
    # merge_csv_df3['|sub|'] = (merge_csv_df3['Predict pKa'] - merge_csv_df3['Target pKa']).abs()
    # merge_csv_df3 = merge_csv_df3.loc[merge_csv_df3['|sub|'] >= 2]
    # print(merge_csv_df3)


if __name__ == '__main__':
    res_only=False
    evaluate_single_model(res_only=res_only)
    # evaluate_CpHMD_result()
    # evaluate_all_model()
