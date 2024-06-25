import re
import json
from pandas import DataFrame, Series
import pandas as pd
import matplotlib.pyplot as plt
import preprocess_pdb
import os
import shutil
import random
import numpy as np
from preprocess_csv import model_pka, save_protein_features_as_csv
from preprocess_pdb import pdb_to_df, df_to_pdb_string


def test_re():
    str = '10'
    result = re.match('^\d+(\.\d+)?$', str)
    print(result)


def test_json():
    json_path = '/media/czt/My Passport/czt/data/total_output/' \
                '1ajw_output_replica/1ajw_input/1ajw_run_info.json'
    with open(json_path, 'r') as json_f:
        run_info_dict = json.load(json_f)
    print(run_info_dict['phs'])


def test_devide_zero():
    a = DataFrame([[1, 2, -3], [4, 5, 6], [7, 8, 9]])
    b = DataFrame([[1, 2, 0], [4, 0, 6], [7, 8, 0]])
    # c = b / a
    c = a / b
    c = c - 0.5
    print(c)


def test_apply():
    a = DataFrame([[1, 2, -3], [4, 5, 6], [7, 8, 9]])
    print(a)
    b = a.apply(lambda x: print(x))
    # print()


def process(a, b):
    # print(a)
    print(a.name)
    # print(b)
    return a


def test_apply_args():
    a = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    b = DataFrame([2, 3, 4], index=['A', 'B', 'C'])
    print(a)
    print(b)
    print(a.apply(process, args=(b,)))


def test_append():
    a = Series()
    b = Series([3, 4, 5])
    c = Series([4, 5, 6])
    a = pd.concat([b, c])
    print(b)
    print(c)
    print(a)


def test_hist():
    a = Series([1, 2, 3, 4, 3, 1.2, 3.4, 0.5, 0.6])
    a.hist(bins=20, figsize=(20, 15))
    plt.show()


def test_group_apply(x, y):
    return x.count() + y


def test_series_to_frame():
    a = Series({
        ('a', 'one', 'I'): 1,
        ('a', 'two', 'II'): 2,
        ('b', 'one', 'I'): 10,
        ('b', 'two', 'II'): 20,
    })
    print(a)
    b = a.to_frame().groupby(level=[1, 2]).apply(test_group_apply, (1,))
    print(b)


def test_get_lines_from_serise():
    a = Series({
        ('a', 'one', 'I'): 1,
        ('a', 'two', 'II'): 2,
        ('b', 'one', 'I'): 10,
        ('b', 'two', 'II'): 20,
    })
    print(a)
    idx = pd.IndexSlice
    b = a.loc[idx[:, 'one', :]]
    print(b)


def test_multi_index():
    a = Series({
        ('a', 'one', 'I'): 1,
        ('a', 'two', 'II'): 2,
        ('b', 'one', 'I'): 10,
        ('b', 'two', 'II'): 20,
    })

    b = Series({
        ('a', 'one', 'I'): 10,
        ('a', 'two', 'II'): 20,
        ('b', 'one', 'I'): 100,
        ('b', 'two', 'II'): 200,
        ('c', 'three', 'III'): 600,
    })
    indexs = a.index
    c = b[indexs]
    print(c)
    c = c.to_frame()
    # print(c.index.name)
    c = c.reset_index()
    print(c)
    c = c.rename(columns={'level_0': 'chain', 'level_1': 'id'})
    print(c)


def test_apply_add_chain():
    b = Series({
        ('a', 'one', 'I'): 10,
        ('a', 'two', 'II'): 20,
        ('b', 'one', 'I'): 100,
        ('b', 'two', 'II'): 200,
        ('c', 'three', 'III'): 600,
    })
    b = b.to_frame()
    b['happy'] = None
    b.loc['a']['happy'] = 'a'
    print(b.to_csv())


def test_lower():
    a = 'Efe.pdb'
    print(a.lower())


def test_fix_pdb():
    source_pdb_path = '/media/czt/TOSHIBA SSD/science_work/data/DownloadPDB/data_pdb_CpHMD/3CC1.pdb'
    target_pdb_path = './charmm_script/fix_3cc1.pdb'
    preprocess_pdb.fix_pdb_file(source_pdb_path, target_pdb_path)


def test_select_rows():
    a = DataFrame({'A': ['a', 'b', 'c'], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    # a = a.set_index(['A', 'B'])
    c = {'c': 5, 'b': 5, 'a': 5}
    a['model'] = a['A'].apply(lambda x: c[x])
    print(a)
    a['dist'] = (a['B'] - a['model']).abs()
    # b = a.applymap(lambda x: x.index)
    print(a.loc[(a['A'] in ['a', 'b'])])

    # print(a.loc[(a['C'])])


def test_read_exp_csv():
    csv_path = '/home/czt/data/pka_data/exp_pka.csv'
    pka_dataframe = pd.read_csv(csv_path)
    useful_res_name = ['ASP', 'GLU', 'LYS', 'HIS']
    model_pka = {
        'ASP': 3.7,
        'GLU': 4.2,
        'HIS': 6.3,
        'LYS': 10.4,
        'CYS': 8.5,
    }
    mini_shift = 1
    pka_dataframe['Res Name'] = pka_dataframe['Res Name'].apply(lambda x: x.upper())
    pka_dataframe['pKa useful'] = pka_dataframe['Expt. pKa'].apply(lambda x: re.match('^\d+(\.\d+)?$', x) is not None)
    pka_dataframe = pka_dataframe.loc[pka_dataframe['pKa useful']]
    pka_dataframe['pKa useful'] = pka_dataframe['Res Name'].apply(lambda x: x in useful_res_name)
    pka_dataframe = pka_dataframe.loc[pka_dataframe['pKa useful']]
    pka_dataframe['model pKa'] = pka_dataframe['Res Name'].apply(lambda x: model_pka[x])
    pka_dataframe['float pKa'] = pka_dataframe['Expt. pKa'].apply(lambda x: float(x))
    pka_dataframe['shift'] = (pka_dataframe['float pKa'] - pka_dataframe['model pKa']).abs()
    pka_dataframe = pka_dataframe.loc[pka_dataframe['shift'] >= mini_shift]
    # exp_df = pka_dataframe['PDB ID'].drop_duplicates().reset_index()
    pka_dataframe = pka_dataframe.reset_index().groupby('PDB ID').count()
    pka_dataframe.to_csv('temp.csv')
    print(pka_dataframe)


def test_draw_predict_target_plot(df, choose_res_names, title, save_dir=None):
    """
    This function will read information form csv file, then draw Predict pKa - Target pKa plot, the csv file must
    contain columns ['PDB ID', 'Chain', 'Res ID', 'Res Name', 'Predict pKa', 'Target pKa'].
    :param csv_path: String, The path of csv file, the csv file contain predict information.
    :param choose_res_name: String, Choosed residue name should be draw, residue name must in
                            ['ASP', 'GLU', 'HIS', 'CYS', 'LYS']
    :param save_dir: Stirng, the save directory of draw plot, if None, will not saved.
    :return: None.
    """
    predict_df = df
    predict_df = predict_df.loc[predict_df['Res Name'].isin(choose_res_names)]
    predict_df.plot.scatter(x='Expt. pKa', y='pKa', title=title, marker='o', alpha=0.5)
    plt.plot(range(-1, 15), range(-1, 15), linestyle='--', color='r')
    plt.axis([-1, 15, -1, 15])
    plt.axes().set_aspect('equal')
    # print(predict_df)
    save_name = '{}.jpg'.format(title)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.show()


def test_draw_exp_cphmd_scatter():
    exp_csv_path = '/home/czt/data/pka_data/exp_pka.csv'
    cphmd_csv_path = '/home/czt/data/pka_data/WT_shift21_pka.csv'
    exp_df = pd.read_csv(exp_csv_path)
    exp_df['pKa useful'] = exp_df['Expt. pKa'].apply(lambda x: re.match('^\d+(\.\d+)?$', x) is not None)
    exp_df = exp_df.loc[exp_df['pKa useful']]
    exp_df = exp_df[['PDB ID', 'Res ID', 'Res Name', 'Expt. pKa']]
    exp_df['Expt. pKa'] = exp_df['Expt. pKa'].apply(lambda x: float(x))
    cphmd_df = pd.read_csv(cphmd_csv_path)
    cphmd_df = cphmd_df[['PDB ID', 'Res ID', 'Res Name', 'pKa']]
    join_df = pd.merge(exp_df, cphmd_df)
    choosed_res_names = [['ASP', 'GLU', 'HIS', 'LYS'], ['ASP'], ['GLU'], ['HIS'], ['LYS']]
    for choosed_res_name in choosed_res_names:
        save_dir = '/media/czt/My Passport/czt/img/scatter'
        title = '_'.join(choosed_res_name)
        test_draw_predict_target_plot(join_df, choosed_res_name, title, save_dir)
    print(join_df)


def test_copy_rest_accurate_pdb():
    copy_dir = '/home/czt/data/pka_data_new/data_pdb_WT_accurate'
    delete_dir = '/home/czt/data/pka_data_new/data_pdb_WT_shift21'
    rest_dir = '/home/czt/data/pka_data_new/data_pdb_WT101'
    copy_names = os.listdir(copy_dir)
    delete_names = os.listdir(delete_dir)
    rest_names = list(set(copy_names).difference(set(delete_names)))

    if not os.path.exists(rest_dir):
        os.mkdir(rest_dir)
    else:
        shutil.rmtree(rest_dir)
        os.mkdir(rest_dir)
    print(copy_names)
    print(delete_names)
    for name in rest_names:
        source_path = os.path.join(copy_dir, name)
        target_path = os.path.join(rest_dir, name)
        shutil.copyfile(source_path, target_path)
        print('{} -copy-> {}'.format(source_path, target_path))


def find_same_pdb_id_and_delete_it():
    dir1 = '/home/czt/data/pka_data_new/data_pdb'
    dir2 = '/home/czt/data/pka_data_new/data_pdb_CpHMD6000_single_chain'
    name_list1 = os.listdir(dir1)
    name_list2 = os.listdir(dir2)
    same_name_list = [x for x in name_list2 if x in name_list1]
    for x in same_name_list:
        name_path = os.path.join(dir2, x)
        print('remove', name_path)
        os.remove(name_path)


def count_SBOND():
    csv_file = '/media/czt/My Passport/czt/data/pka_data_new/expt_cleaned_csv/WT_cleaned_pka2.csv'
    pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_WT'
    new_pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_WT_SSBOND'
    if not os.path.exists(new_pdb_dir):
        os.mkdir(new_pdb_dir)
    csv_df = pd.read_csv(csv_file)
    csv_df = csv_df.loc[:, ['PDB ID', 'Chain']].drop_duplicates()
    total = 0
    count = 0
    for i, v in csv_df['PDB ID'].items():
        pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(v))
        # print('read {}'.format(pdb_path))
        flag = False
        print(pdb_path)
        with open(pdb_path, 'r') as f:
            line = f.readline()
            while (line):
                if line.find('SBOND'):
                    result = line.find('SBOND')
                    if result != -1:
                        str_list = line.split()
                        if str_list[3] == str_list[6] and str_list[6] == csv_df['Chain'][i]:
                            print('*' * 5, csv_df['Chain'][i])
                            flag = True
                            print(line)
                line = f.readline()
        if flag:
            count += 1
            new_pdb_path = os.path.join(new_pdb_dir, '{}.pdb'.format(v))
            shutil.copyfile(pdb_path, new_pdb_path)
        total += 1

    print('total: {}, count: {}'.format(total, count))


def is_SSBOND_in_first_chain(pdb_file):
    SSBOND_content = preprocess_pdb.get_SSBOND_content(pdb_file)
    if SSBOND_content != '':
        SSBOND_lines = SSBOND_content.split('\n')
        SSBOND_lines.remove('')
        SSBOND_chains = [line.split()[3] for line in SSBOND_lines]
        first_chain = ''
        with open(pdb_file, 'r') as f:
            line = f.readline()
            while line:
                str_list = preprocess_pdb.split_pdb_line(line)
                if str_list[0] == 'ATOM' or str_list[0] == 'HETATM':
                    # if str contain amino acid.
                    if str_list[3][-3:] in preprocess_pdb.amino_acids_names:
                        first_chain = str_list[4]
                line = f.readline()
        result = False
        if first_chain in SSBOND_chains:
            result = True
    else:
        result = False
    return result


def test_find_SSBOND_pdb_files():
    pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb'
    pdb_list = os.listdir(pdb_dir)
    count_all = 0
    count_true = 0
    ssbond_pdbid_list = []
    for pdb_name in pdb_list:
        pdb_path = os.path.join(pdb_dir, pdb_name)
        pdb_id = pdb_name.split('.')[0].upper()
        if is_SSBOND_in_first_chain(pdb_path):
            print(pdb_path)
            count_true += 1
            ssbond_pdbid_list.append(pdb_id)
        count_all += 1
    print("{}/{}".format(count_true, count_all))
    return ssbond_pdbid_list


def test_find_finished_CpHMD_pdb():
    total_output_dir = '/media/czt/My Passport/czt/data/total_output'
    wrong_total_output_dir = '/media/czt/My Passport/czt/data/wrong_total_output'
    output_list = os.listdir(total_output_dir)
    count_all = 0
    count_finished = 0
    finished_pdbid_list = []
    for output_dir in output_list:
        pdb_id = output_dir.split('_')[0]
        output_path = os.path.join(total_output_dir, output_dir)
        pka_path = os.path.join(output_path, '{}_pKa'.format(pdb_id), '{}_pka_table.txt'.format(pdb_id))
        if os.path.exists(pka_path):
            count_finished += 1
            finished_pdbid_list.append(pdb_id.upper())
        else:
            wrong_output_path = os.path.join(wrong_total_output_dir, output_dir)
            if not os.path.exists(wrong_total_output_dir):
                os.mkdir(wrong_total_output_dir)
            shutil.move(output_path, wrong_output_path)
            print(output_path)
        count_all += 1
    print('{}/{}'.format(count_finished, count_all))
    return finished_pdbid_list


def find_right_finished_pdb_id():
    ssbond_pdbid_list = test_find_SSBOND_pdb_files()
    finished_pdbid_list = test_find_finished_CpHMD_pdb()
    right_pdbid_list = [x for x in finished_pdbid_list if x not in ssbond_pdbid_list]
    print(ssbond_pdbid_list)
    print(finished_pdbid_list)
    print(len(right_pdbid_list), right_pdbid_list)


def copy_ssbond_pdb_in_one_dir():
    pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_CpHMD6000'
    save_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_CpHMD_SSBOND21'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ssbond_pdbid_list = test_find_SSBOND_pdb_files()
    for pdb_id in ssbond_pdbid_list:
        source_pdb = os.path.join(pdb_dir, '{}.pdb'.format(pdb_id))
        target_pdb = os.path.join(save_dir, '{}.pdb'.format(pdb_id))
        print('{}->{}'.format(source_pdb, target_pdb))
        shutil.copyfile(source_pdb, target_pdb)


def copy_latest_pdb_in_one_dir():
    pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_CpHMD6000'
    save_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_CpHMD6000_latest'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ssbond_pdbid_list = test_find_SSBOND_pdb_files()
    finished_pdbid_list = test_find_finished_CpHMD_pdb()
    ssbond_pdbid_set = set(ssbond_pdbid_list)
    finished_pdbid_set = set(finished_pdbid_list)
    union_pdbid_list = list(ssbond_pdbid_set | finished_pdbid_set)
    source_pdb_list = os.listdir(pdb_dir)
    source_pdbid_list = [x.split('.')[0] for x in source_pdb_list]
    latest_pdbid_list = [pdb_id for pdb_id in source_pdbid_list if pdb_id not in union_pdbid_list]

    for pdb_id in latest_pdbid_list:
        source_pdb = os.path.join(pdb_dir, '{}.pdb'.format(pdb_id))
        target_pdb = os.path.join(save_dir, '{}.pdb'.format(pdb_id))
        print('{}->{}'.format(source_pdb, target_pdb))
        shutil.copyfile(source_pdb, target_pdb)


def copy_SSBOND_lambda_proteins_to_new_dir():
    # CpHMD_dir = '/media/czt/My Passport/czt/data/total_output'
    CpHMD_dir = '/media/czt/My Passport/czt/data/total_output_WT'
    pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_WT'
    wrong_CpHMD_SSBOND_dir = '/media/czt/My Passport/czt/data/wrong_total_output_WT_SSBOND'
    if not os.path.exists(wrong_CpHMD_SSBOND_dir):
        os.mkdir(wrong_CpHMD_SSBOND_dir)
    file_names = os.listdir(CpHMD_dir)
    pdb_id_list = [file_name.split('_')[0].upper() for file_name in file_names]
    source_pdb_path_list = [os.path.join(pdb_dir, '{}.pdb'.format(pdb_id)) for pdb_id in pdb_id_list]
    count_all = 0
    count_num = 0
    pdb_id_list = []
    for pdb_path in source_pdb_path_list:
        if is_SSBOND_in_first_chain(pdb_path):
            pdb_id = pdb_path.split('/')[-1].split('.')[0].lower()
            pdb_id_list.append(pdb_id)
            print(pdb_path)
            count_num += 1
        count_all += 1
    print('{}/{}'.format(count_num, count_all))

    for pdb_id in pdb_id_list:
        replica_path = os.path.join(CpHMD_dir, '{}_output_replica'.format(pdb_id))
        new_replica_path = os.path.join(wrong_CpHMD_SSBOND_dir, '{}_output_replica'.format(pdb_id))
        print('{} -move-> {}'.format(replica_path, new_replica_path))
        shutil.move(replica_path, new_replica_path)


def count_pka_data(csv_df):
    global model_pka
    csv_df['model pKa'] = csv_df['Res Name'].map(lambda x: model_pka[x])
    csv_df['shift pKa'] = csv_df['pKa'] - csv_df['model pKa']
    csv_df['abs shift pKa'] = csv_df['shift pKa'].map(lambda x: abs(x))

    print(csv_df.count())
    print('total proteins' + '*' * 100)
    print(csv_df['PDB ID'].drop_duplicates().count())

    print('total' + '*' * 100)
    print(csv_df.groupby('Res Name').count())

    csv_df1 = csv_df[csv_df['abs shift pKa'] <= 0.5]
    print('[0, 0.5]' + '*' * 100)
    print(csv_df1.groupby('Res Name').count())

    csv_df2 = csv_df[(csv_df['abs shift pKa'] > 0.5) & (csv_df['abs shift pKa'] <= 1)]
    print('(0.5, 1]' + '*' * 100)
    print(csv_df2.groupby('Res Name').count())

    csv_df3 = csv_df[(csv_df['abs shift pKa'] > 1) & (csv_df['abs shift pKa'] <= 1.5)]
    print('(1, 1.5]' + '*' * 100)
    print(csv_df3.groupby('Res Name').count())

    csv_df4 = csv_df[(csv_df['abs shift pKa'] > 1.5) & (csv_df['abs shift pKa'] <= 2)]
    print('(1.5, 2]' + '*' * 100)
    print(csv_df4.groupby('Res Name').count())

    csv_df5 = csv_df[(csv_df['abs shift pKa'] > 2)]
    print('(2, ++)' + '*' * 100)
    print(csv_df5.groupby('Res Name').count())


def test_count_pka_data():
    csv_path = '../data/model_input/final_val_data/val_n27_f19_n4.csv'
    # csv_path = '/home/huang/newdisk01/czt/安装pka_predict/pka_predict/final_val_data_new2/val_chimera_f19_r4_incphmd.csv'
    # csv_path = '/home/huang/newdisk01/czt/安装pka_predict/pka_predict/final_val_data_new2/val_chimera_f19_r4_incphmd_cgdtb.csv'
    csv_df = pd.read_csv(csv_path)
    csv_df['PDB ID'] = csv_df['file_name'].map(lambda x: x.split('_')[0])
    csv_df['Chain'] = csv_df['file_name'].map(lambda x: x.split('_')[1])
    csv_df['Res Name'] = csv_df['file_name'].map(lambda x: x.split('_')[3])
    csv_df['Res ID'] = csv_df['file_name'].map(lambda x: x.split('_')[2])
    csv_df.rename(columns={'pka': 'pKa'}, inplace=True)
    csv_df = csv_df.loc[:, ['PDB ID', 'Res Name', 'Chain', 'Res ID', 'pKa']]
    csv_df = csv_df.drop_duplicates()
    count_pka_data(csv_df)

    # csv_path = '/home/huang/project/pka_predict/train_info/model_bn4_elu_n228_f19_r4_zscore_fcharge_adam_L2/test_f19_r4_inSSbondCpHMD.csv'
    # csv_path = '/home/huang/project/pka_predict/train_info/model_bn4_elu_n228_f19_r4_zscore_fcharge_adam_L2/test_f19_r4_inSSbondCpHMD_42_29.csv'
    # csv_path = '/home/huang/project/pka_predict/train_info/model_bn4_elu_n228_f19_r4_zscore_fcharge_adam_L2/test_f19_r4_cgdtb_inSSbondCpHMD_average_RMSE.csv'
    # csv_path = '/home/czt/project/Predict_pKa/train_info/model_bn4_elu_n228_f19_r4_zscore_fcharge_adam2/val_f19_r4_incphmd_cgdtb.csv'
    # csv_df = pd.read_csv(csv_path)
    # csv_df = csv_df.loc[:, ['PDB ID', 'Res Name', 'Chain', 'Res ID', 'Target pKa']]
    # csv_df.rename(columns={'Target pKa': 'pKa'}, inplace=True)
    # count_pka_data(csv_df)


def test_find_shift_asp():
    csv_path = '/home/huang/project/pka_predict/train_info/model_bn4_elu_n228_f19_r4_zscore_fcharge_adam_L2/test_f19_r4_cgdtb_inSSbondCpHMD_average_RMSE.csv'
    csv_df = pd.read_csv(csv_path)
    csv_df = csv_df.loc[:, ['PDB ID', 'Res Name', 'Chain', 'Res ID', 'Target pKa']]
    csv_df.rename(columns={'Target pKa': 'pKa'}, inplace=True)
    csv_df = csv_df[csv_df['Res Name'] == 'ASP']

    csv_df['model pKa'] = csv_df['Res Name'].map(lambda x: model_pka[x])
    csv_df['shift pKa'] = csv_df['pKa'] - csv_df['model pKa']
    csv_df['abs shift pKa'] = csv_df['shift pKa'].map(lambda x: abs(x))
    csv_df = csv_df[csv_df['abs shift pKa'] > 3]
    print(csv_df)


def is_same_protein(test_test_df, pdbid1, pdbid2):
    is_same = False
    choose_df = test_test_df[(test_test_df['pdbid_x'] == pdbid1) & (test_test_df['pdbid_y'] == pdbid2) |
                             (test_test_df['pdbid_x'] == pdbid2) & (test_test_df['pdbid_y'] == pdbid1)]
    result_num = int(choose_df.count(axis=0)['identity'])
    if result_num != 0:
        identity = choose_df['identity'].max()
        if identity >= 100:
            is_same = True
    return is_same


def find_same_protein_pdb_ids(test_test_df):
    pdbid_list = []
    for idx, rows in test_test_df.iterrows():
        if rows['identity'] >= 100:
            pdbid_list.append(rows['pdbid_x'])
            pdbid_list.append(rows['pdbid_y'])
    pdbid_list = list(set(pdbid_list))
    return pdbid_list


class union_set():

    def __init__(self, data_list):
        """初始化两个字典，一个保存节点的父节点，另外一个保存父节点的大小
        初始化的时候，将节点的父节点设为自身，size设为1"""
        self.father_dict = {}
        self.size_dict = {}
        self.data_list = data_list

        for node in data_list:
            self.father_dict[node] = node
            self.size_dict[node] = 1

    def find_head(self, node):
        """使用递归的方式来查找父节点

        在查找父节点的时候，顺便把当前节点移动到父节点上面
        这个操作算是一个优化
        """
        father = self.father_dict[node]
        if (node != father):
            father = self.find_head(father)
        self.father_dict[node] = father
        return father

    def is_same_set(self, node_a, node_b):
        """查看两个节点是不是在一个集合里面"""
        return self.find_head(node_a) == self.find_head(node_b)

    def union(self, node_a, node_b):
        """将两个集合合并在一起"""
        if node_a is None or node_b is None:
            return

        a_head = self.find_head(node_a)
        b_head = self.find_head(node_b)

        if (a_head != b_head):
            a_set_size = self.size_dict[a_head]
            b_set_size = self.size_dict[b_head]
            if (a_set_size >= b_set_size):
                self.father_dict[b_head] = a_head
                self.size_dict[a_head] = a_set_size + b_set_size
            else:
                self.father_dict[a_head] = b_head
                self.size_dict[b_head] = a_set_size + b_set_size

    def get_all_head_nodes(self):
        head_nodes = []
        for node in self.data_list:
            if node == self.father_dict[node]:
                head_nodes.append(node)
        return head_nodes


def test_find_same_residue():
    pka_data_csv = '/media/czt/My Passport/czt/data/pka_data_new/expt_cleaned_csv/WT_cleaned_pka2.csv'
    test_test_csv = '/media/czt/My Passport/czt/data/pka_data_new/同源对比数据/test_test.csv'
    pka_data_df = pd.read_csv(pka_data_csv)
    test_test_df = pd.read_csv(test_test_csv)
    pdbid_list = find_same_protein_pdb_ids(test_test_df)
    print(pdbid_list, len(pdbid_list))
    pka_data_df = pka_data_df[pka_data_df['PDB ID'].map(lambda x: x in pdbid_list)]
    print(is_same_protein(test_test_df, '1LSE', '2LZT'))

    pka_data_df['ID'] = pka_data_df['PDB ID'].map(str) + '_' + pka_data_df['Chain'] + '_' \
                        + pka_data_df['Res Name'] + '_' + pka_data_df['Res ID'].map(str)
    data_list = pka_data_df['ID'].to_list()
    uf = union_set(data_list)

    for idx, rows in pka_data_df.iterrows():
        for idx2, rows2 in pka_data_df.iterrows():
            if idx < idx2:
                if rows['PDB ID'] != rows2['PDB ID']:
                    if is_same_protein(test_test_df, rows['PDB ID'], rows2['PDB ID']):
                        if rows['ID'][5:] == rows2['ID'][5:]:
                            print('same:' + '*' * 100)
                            uf.union(rows['ID'], rows2['ID'])
                            print(rows['ID'], rows2['ID'])

    head_list = uf.get_all_head_nodes()
    save_df = pd.DataFrame({'ID': head_list})
    delete_list = [x for x in data_list if x not in head_list]
    delete_df = pd.DataFrame({'ID': delete_list})
    save_df['PDB ID'] = save_df['ID'].map(lambda x: x.split('_')[0])
    save_df['Chain'] = save_df['ID'].map(lambda x: x.split('_')[1])
    save_df['Res Name'] = save_df['ID'].map(lambda x: x.split('_')[2])
    save_df['Res ID'] = save_df['ID'].map(lambda x: x.split('_')[3])

    delete_df['PDB ID'] = delete_df['ID'].map(lambda x: x.split('_')[0])
    delete_df['Chain'] = delete_df['ID'].map(lambda x: x.split('_')[1])
    delete_df['Res Name'] = delete_df['ID'].map(lambda x: x.split('_')[2])
    delete_df['Res ID'] = delete_df['ID'].map(lambda x: x.split('_')[3])

    save_df.to_csv('save_test_data_residue.csv')
    delete_df.to_csv('delete_test_data_residue.csv')


def choose_cols():
    csv_path = 'delete_test_data_residue.csv'
    df = pd.read_csv(csv_path)
    df = df.loc[:, ['PDB ID', 'Chain', 'Res Name', 'Res ID']]
    print(df)
    df.to_csv(csv_path, index=False)


def test_clean_data():
    cleaned_csv = '/media/czt/My Passport/czt/data/pka_data_new/expt_cleaned_csv/WT_cleaned_pka2.csv'
    delete_protein = '1UBQ'
    delete_residues_csv = '/media/czt/My Passport/czt/data/pka_data_new/同源对比数据/delete_test_data_residue.csv'
    add_csv = '/media/czt/My Passport/czt/data/pka_data_new/expt_cleaned_csv/add_pka.csv'
    can_calculate_csv = '/media/czt/My Passport/czt/data/pka_data_new/同源对比数据/test_f19_r4_inSSbondCpHMD.csv'

    cleaned_df = pd.read_csv(cleaned_csv)
    delete_residues_df = pd.read_csv(delete_residues_csv)
    add_df = pd.read_csv(add_csv)
    can_caculate_df = pd.read_csv(can_calculate_csv)

    cleaned_df = cleaned_df[~(cleaned_df['Res Name'] == delete_protein)]
    cleaned_df['ID'] = cleaned_df['PDB ID'].map(str) + '_' + cleaned_df['Chain'] + '_' \
                       + cleaned_df['Res Name'] + '_' + cleaned_df['Res ID'].map(str)
    delete_residues_df['ID'] = delete_residues_df['PDB ID'].map(str) + '_' + delete_residues_df['Chain'] + '_' \
                               + delete_residues_df['Res Name'] + '_' + delete_residues_df['Res ID'].map(str)
    can_caculate_df['ID'] = can_caculate_df['PDB ID'].map(str) + '_' + can_caculate_df['Chain'] + '_' \
                            + can_caculate_df['Res Name'] + '_' + can_caculate_df['Res ID'].map(str)

    delete_list = delete_residues_df['ID'].to_list()
    incphmd_list = can_caculate_df['ID'].to_list()

    cleaned_df = cleaned_df[~(cleaned_df['ID'].map(lambda x: x in delete_list))]
    # cleaned_df = cleaned_df[(cleaned_df['ID'].map(lambda x: x in incphmd_list))]
    union_df = pd.merge(cleaned_df, add_df, how='outer')
    union_df = union_df.loc[:, ['PDB ID', 'Res Name', 'Chain', 'Res ID', 'pKa']]
    union_df.to_csv('/media/czt/My Passport/czt/data/pka_data_new/expt_cleaned_csv/final_expt_pka.csv', index=False)
    print(union_df)


def select_choosed_CpHMD_WT_pka_data():
    CpHMD_WT_pka_data = '/media/czt/My Passport/czt/data/pka_data_new/cphmd_csv/CpHMD_pka_WT116.csv'
    cleaned_expt_data = '/media/czt/My Passport/czt/data/pka_data_new/expt_cleaned_csv/final_expt_pka.csv'

    CpHMD_WT_df = pd.read_csv(CpHMD_WT_pka_data)
    cleaned_expt_df = pd.read_csv(cleaned_expt_data)
    cleaned_expt_df = cleaned_expt_df.loc[:, 'PDB ID', 'Chain', 'Res Name', 'Res ID']
    merge_df = pd.merge(CpHMD_WT_df, cleaned_expt_df)
    merge_df.to('/media/czt/My Passport/czt/data/pka_data_new/cphmd_csv/CpHMD_pka_WT116_cleaned.csv')
    print(merge_df)


def test_numpy():
    a = np.zeros((3,))
    print(a)


def test_os_system():
    # os.system(r'echo \"open {}' + '\n' + 'addh' + '\n' + 'addcharge \n write format mol2 0 {} \n stop\"')
    pass


def data_work_on_XXXX():
    features_csv = '/media/czt/TOSHIBA SSD/science_work/code/pka_process_new/pka_data_new/fixed_single_chain_features_csv/data_pdb_WT_fixed_mol2.csv'
    pdb_file = '/media/czt/TOSHIBA SSD/science_work/code/pka_process_new/pka_data_new/data_pdb_WT_fixed/3RUZ_A.pdb'
    features_df = pd.read_csv(features_csv)
    pdb_df = pdb_to_df(pdb_file)
    features_df = features_df.loc[:, ['x', 'y', 'z', 'partialcharge']]
    features_df = features_df.rename(columns={'partialcharge': 'temp1'})
    features_df['x'] = features_df['x'].map(lambda x: round(x, 3))
    features_df['y'] = features_df['y'].map(lambda x: round(x, 3))
    features_df['z'] = features_df['z'].map(lambda x: round(x, 3))
    pdb_df = pdb_df.loc[:, ['model', 'atom', 'idx', 'heavy_atom', 'amino_acid', 'chain',
                            'res_id', 'x', 'y', 'z', 'temp2', 'temp3']]

    merge_df = pd.merge(features_df, pdb_df)
    print(merge_df)
    merge_df['temp1'] = merge_df['temp1'].map(lambda x: '{0:.3f}'.format(round(x, 3)))
    new_pdb_string = df_to_pdb_string(merge_df)
    print(new_pdb_string)
    with open('3ruz_charge.pdb', 'w') as f:
        f.write(new_pdb_string)


def get_3ruz_features():
    # read fixed one chain pdb files caculate protein features and save as csv file.
    fixed_pdb_directory = '/media/czt/My Passport/czt/data/pka_data_new/3ruz_temp'
    csv_directory = '/media/czt/My Passport/czt/data/pka_data_new/fixed_single_chain_features_csv'
    protein_format = 'pdb'
    save_protein_features_as_csv(protein=fixed_pdb_directory, protein_format=protein_format, output_dir=csv_directory)


def test_find_used_train_residues():
    input_train_data = "../data/model_input/final_train_data/train_n279_f19_n4.csv"
    source_train_pka = '../data/cleaned_source_data/cphmd_csv/CpHMD_pka282.csv'
    new_source_train_pka = '../data/cleaned_source_data/cphmd_csv/CpHMD_pka279.csv'

    input_df = pd.read_csv(input_train_data)
    source_pka_df = pd.read_csv(source_train_pka)

    input_df['PDB ID'] = input_df['file_name'].map(lambda x: x.split('_')[0])
    input_df['Chain'] = input_df['file_name'].map(lambda x: x.split('_')[1])
    input_df['Res ID'] = input_df['file_name'].map(lambda x: int(x.split('_')[2]))
    input_df['Res Name'] = input_df['file_name'].map(lambda x: x.split('_')[3])

    input_df = input_df.loc[:, ['PDB ID', 'Chain', 'Res ID', 'Res Name']].drop_duplicates()
    merge_df = pd.merge(source_pka_df, input_df)
    print(merge_df)
    merge_df.to_csv(new_source_train_pka, index=False)


def test_find_used_val_residues():
    input_train_data = "../data/model_input/final_val_data/val_chimera_f19_r4_incphmd.csv"
    source_train_pka = '../data/cleaned_source_data/cphmd_csv/CpHMD_pka_WT116.csv'
    new_source_train_pka = '../data/cleaned_source_data/cphmd_csv/CpHMD_pka_WT69.csv'

    input_df = pd.read_csv(input_train_data)
    source_pka_df = pd.read_csv(source_train_pka)

    input_df['PDB ID'] = input_df['file_name'].map(lambda x: x.split('_')[0])
    input_df['Chain'] = input_df['file_name'].map(lambda x: x.split('_')[1])
    input_df['Res ID'] = input_df['file_name'].map(lambda x: int(x.split('_')[2]))
    input_df['Res Name'] = input_df['file_name'].map(lambda x: x.split('_')[3])

    input_df = input_df.loc[:, ['PDB ID', 'Chain', 'Res ID', 'Res Name']].drop_duplicates()
    merge_df = pd.merge(source_pka_df, input_df)
    print(merge_df)
    merge_df.to_csv(new_source_train_pka, index=False)


def test_delete_redundant_pdb_files():
    csv_file = '../data/cleaned_source_data/cphmd_csv/CpHMD_pka279.csv'
    csv_file = '../data/cleaned_source_data/cphmd_csv/CpHMD_pka_WT69.csv'
    pdb_dir = '../data/cleaned_source_data/data_pdb_CpHMD282'
    pdb_dir = '../data/cleaned_source_data/data_pdb_WT_accurate'
    csv_df = pd.read_csv(csv_file)
    used_pdb_names = (csv_df['PDB ID'] + '_' + csv_df['Chain']).drop_duplicates().to_list()
    used_pdb_names = csv_df['PDB ID'].drop_duplicates().to_list()
    used_pdb_names = ['{}.pdb'.format(x) for x in used_pdb_names]
    total_pdb_names = os.listdir(pdb_dir)
    redundant_pdb_names = list(set(total_pdb_names) - set(used_pdb_names))
    print(redundant_pdb_names, len(redundant_pdb_names))
    for name in redundant_pdb_names:
        file_path = os.path.join(pdb_dir, name)
        os.remove(file_path)


def test_divide_train_data_into_train_and_val_dataset():
    """
    This function is using for dividing old train data(279 proteins) into new train data(252 proteins)
    and new validate data(27 proteins).
    (This function is used temporarily)
    """
    old_coors_file = '../data/model_input/old_train_data/CpHMD_pka279_center_coors.csv'
    old_residues_features_file = '../data/model_input/old_train_data/train_n279_f19_n4.csv'
    old_proteins_features_file = '../data/model_input/old_train_data/data_pdb_CpHMD279_fixed_mol2.csv'

    val_coors_file = '../data/model_input/final_val_data/CpHMD_pka27_center_coors.csv'
    val_residues_features_file = '../data/model_input/final_val_data/val_n27_f19_n4.csv'
    val_proteins_features_file = '../data/model_input/final_val_data/data_pdb_CpHMD27_fixed_mol2.csv'

    train_coors_file = '../data/model_input/final_train_data/CpHMD_pka252_center_coors.csv'
    train_residues_features_file = '../data/model_input/final_train_data/train_n252_f19_n4.csv'
    train_proteins_features_file = '../data/model_input/final_train_data/data_pdb_CpHMD252_fixed_mol2.csv'

    old_coors_df = pd.read_csv(old_coors_file)
    old_residues_features_df = pd.read_csv(old_residues_features_file)
    old_proteins_features_df = pd.read_csv(old_proteins_features_file)

    file_names = old_proteins_features_df['file_name'].drop_duplicates().to_list()  # ep: [4NTQ_A_235_ASP, ...]
    proteins_chain = ['{}_{}'.format(x.split('_')[0], x.split('_')[1]) for x in file_names]  # ep: [4NTQ_A, ...]
    proteins_chain = list(set(proteins_chain))  # drop duplicates, sort list for random sampling
    proteins_chain.sort()  # sort list for random sampling

    random.seed(10)
    proteins_chain_slice = random.sample(proteins_chain, 27)  # selecting validate proteins chain.

    # selecting validate data according proteins chain.
    select_str = '|'.join(proteins_chain_slice)
    val_coors_df = old_coors_df[old_coors_df['file_name'].str.contains(select_str)]
    val_residues_features_df = old_residues_features_df[old_residues_features_df['file_name'].str.contains(select_str)]
    val_proteins_features_df = old_proteins_features_df[old_proteins_features_df['file_name'].str.contains(select_str)]
    print(val_coors_df)
    print(val_residues_features_df)
    print(val_proteins_features_df)

    # the last data is train data
    train_coors_df = old_coors_df.drop(val_coors_df.index)
    train_residues_features_df = old_residues_features_df.drop(val_residues_features_df.index)
    train_proteins_features_df = old_proteins_features_df.drop(val_proteins_features_df.index)

    # new train and validate data saved as csv files
    train_coors_df.to_csv(train_coors_file, index=False)
    train_residues_features_df.to_csv(train_residues_features_file, index=False)
    train_proteins_features_df.to_csv(train_proteins_features_file, index=False)

    val_coors_df.to_csv(val_coors_file, index=False)
    val_residues_features_df.to_csv(val_residues_features_file, index=False)
    val_proteins_features_df.to_csv(val_proteins_features_file, index=False)

    # check train data distribution and validate data distribution
    train_residues_features_df = train_residues_features_df[train_residues_features_df['pka'] > -0.5]
    train_residues_features_df.loc[:, ['idx', 'pka']].drop_duplicates().hist('pka', bins=200)
    val_residues_features_df.loc[:, ['idx', 'pka']].drop_duplicates().hist('pka', bins=200)

    plt.show()
    print('finish')


def test_divide_cphmd_info_into_train_info_and_val_info():
    """
    This function will divide cphmd data info into train data info and validate data info
    according to their titratable sites.
     (This function used for temporarily)
    """
    cphmd_info_path = 'cphmd_data_info/cphmd_chimera_residues_info.csv'
    train_data_path = '../data/model_input/final_train_data/train_n252_f19_n4.csv'
    val_data_path = '../data/model_input/final_val_data/val_n27_f19_n4.csv'
    train_info_save_path = 'train_data_info'
    val_info_save_path = 'val_data_info'

    # read files with dataframe type
    cphmd_info_df = pd.read_csv(cphmd_info_path)
    train_data_df = pd.read_csv(train_data_path)
    val_data_df = pd.read_csv(val_data_path)

    # get titratable sites
    train_titratable_sites_df = pd.DataFrame()
    train_titratable_sites_df['PDB ID'] = train_data_df['file_name'].map(lambda x: x.split('_')[0])
    train_titratable_sites_df['Res ID'] = train_data_df['file_name'].map(lambda x: int(x.split('_')[2]))
    train_titratable_sites_df['Res Name'] = train_data_df['file_name'].map(lambda x: x.split('_')[3])
    train_titratable_sites_df = train_titratable_sites_df.drop_duplicates()

    val_titratable_sites_df = pd.DataFrame()
    val_titratable_sites_df['PDB ID'] = val_data_df['file_name'].map(lambda x: x.split('_')[0])
    val_titratable_sites_df['Res ID'] = val_data_df['file_name'].map(lambda x: int(x.split('_')[2]))
    val_titratable_sites_df['Res Name'] = val_data_df['file_name'].map(lambda x: x.split('_')[3])
    val_titratable_sites_df = val_titratable_sites_df.drop_duplicates()

    # select info by titratable sites
    train_info_df = pd.merge(cphmd_info_df, train_titratable_sites_df)
    val_info_df = pd.merge(cphmd_info_df, val_titratable_sites_df)

    # save train info and validate info as csv files
    residue_names = ['ASP', 'GLU', 'HIS', 'LYS']
    list1 = ['train', 'val']
    info_dict = {'train': train_info_df, 'val': val_info_df}
    dir_dict = {'train': train_info_save_path, 'val': val_info_save_path}
    for x in list1:
        save_name = '{}_chimera_residues_info.csv'.format(x)
        save_path = os.path.join(dir_dict[x], save_name)
        info_dict[x].to_csv(save_path, index=False)
        for res_name in residue_names:
            total_info_df = info_dict[x]
            res_info_df = total_info_df[total_info_df['Res Name'] == res_name]
            save_name = '{}_chimera_residues_info_{}.csv'.format(x, res_name.lower())
            save_path = os.path.join(dir_dict[x], save_name)
            res_info_df.to_csv(save_path, index=False)
    print(train_info_df)
    print(val_info_df)


def get_ires_type(ires_name):
    '''
    return number to represent ires type
    :param ires_name:
    :return: int, represent ires type by name
    '''
    ires_type = -1
    if ires_name == 'ASP':
        ires_type = 0
    elif ires_name == 'GLU':
        ires_type = 1
    elif ires_name == 'LYS':
        ires_type = 2
    elif ires_name == 'HIS':
        ires_type = 3
    elif ires_name == 'CYS':
        ires_type = 4
    return ires_type


def test_turn_F19_to_F18_and_F20():
    """
    This function will turn 19 features to 18 features and 20 features.
    20 features: ['idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'is_center_residue', 'res_type', 'hydrophobic',
                 'aromatic', 'acceptor', 'donor', 'ring']
    19 features: ['idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'is_center_residue', 'hydrophobic',
                 'aromatic', 'acceptor', 'donor', 'ring']
    18 features: ['idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'hydrophobic',
                 'aromatic', 'acceptor', 'donor', 'ring']
    (attention: the number of features not count ['idx', 'pka', 'file_name', 'x', 'y', 'z'] columns.)
    (This function only used for temporarily.)
    """
    f19_features_data_path = '../data/model_input/final_val_data/val_n27_f19_n4.csv'
    f20_features_data_save_path = f19_features_data_path.replace('f19', 'f20')
    f18_features_data_save_path = f19_features_data_path.replace('f19', 'f18')

    f19_features_df = pd.read_csv(f19_features_data_path)
    f20_features_df = f19_features_df.copy()
    f20_features_df['res_type'] = f20_features_df['file_name'].map(lambda x: get_ires_type(x.split('_')[3]))
    f20_features_df = f20_features_df.loc[:,
                      ['idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                       'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'is_center_residue', 'res_type',
                       'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']]
    f18_features_df = f20_features_df.loc[:,
                      ['idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                       'hyb', 'heavyvalence', 'heterovalence', 'partialcharge',
                       'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']]
    print(f20_features_df)
    print(f18_features_df)
    f20_features_df.to_csv(f20_features_data_save_path, index=False)
    f18_features_df.to_csv(f18_features_data_save_path, index=False)


if __name__ == '__main__':
    test_turn_F19_to_F18_and_F20()
