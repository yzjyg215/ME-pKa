import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pandas import DataFrame, Series
from preprocess_pdb import split_pdb_line
from preprocess_csv import model_pka

CAN_DEPRO_IERS_NAMES = ['ASP', 'GLU', 'LYS', 'HIS', 'CYS']


def henderson_hasselbalch_for_fitting(ph, hill, pka):
    return 1 / (1 + 10 ** (hill * (pka - ph)))


def compute_pka_and_hill(fit_formula, phs, s_values, model_pka):
    initial_params = [1, model_pka]
    try:
        fit = curve_fit(fit_formula, phs, s_values, p0=initial_params)
        hill = fit[0][0]
        pka = fit[0][1]
    except:
        hill = None
        pka = None
    return hill, pka


def store_lambda_into_dataframe(lambda_file_path, id_name_dict):
    """
    This function will read lambda files, and remove cols with second same residue id,
    than store the data in dataframe.
    :param lambda_file_path: String, the path of lambda file.
    :param lambda_iresID_list: List, lambda residue id
    :parm id_name_dict: Dictionary, the map between pdb residue id and names.
    :return: DataFrame, the DataFrame contain lambda data, columns is steps,
                        indexes is titrable residue (ids, names), value is lambda.
    """
    # read the lambda file
    with open(lambda_file_path, 'r') as f:
        lines = f.readlines()

    split_line_list = []
    for line in lines:
        split_line_list.append(line.split())

    # get titrable residue ids
    ires_ids = Series(data=split_line_list[1][2:])
    ires_ids = ires_ids.drop_duplicates(keep='first')

    # get pdb residue ids
    pdb_res_ids = list(id_name_dict.keys())
    pdb_res_ids.sort()

    # get lambda data
    data_np = np.array(split_line_list[4:]).astype(float)
    lambda_data = DataFrame(data=data_np[:, 1:], index=data_np[:, :1].squeeze().astype(int))
    lambda_data = lambda_data.loc[:, ires_ids.keys()]   # remove cols with second same residue id
    # lambda_data.columns = [int(ires_id) + relative_value for ires_id in ires_ids] # change cols name to residue id
    lambda_data.columns = [int(pdb_res_ids[int(lambda_ires_id) - 2]) for lambda_ires_id in ires_ids]
    lambda_data = lambda_data.T
    lambda_data.index = lambda_data.index.map(lambda x: (x, id_name_dict[x]))
    return lambda_data


def store_lambda_into_multi_index_dataframe(protein_dir):
    """
    This function will read lambda files, and remove cols with second same residue id,
    than store the data in dataframe, then concat every pH dataframe into one multi index dataframe.
    :param protein_dir: String, the path of protein_dir.
    :return: DataFrame, the dataframe with multi index contains lambda data.
                        first level index is pHs, second level index is steps,
                        columns is titrable residue ids, value is lambda.
    """
    multi_lambda_data = None
    # delete last '/'
    if protein_dir[-1] == '/':
        protein_dir = protein_dir[:-1]
    protein_name = protein_dir.split('/')[-1].split('_')[0]
    input_dir = os.path.join(protein_dir, '{}_input'.format(protein_name))
    lambda_dir = os.path.join(protein_dir, '{}_lambda'.format(protein_name))
    run_info_path = os.path.join(input_dir, '{}_run_info.json'.format(protein_name))
    pdb_path = os.path.join(input_dir, '{}.pdb'.format(protein_name))

    # get residue ids:names map
    id_name_dict = get_map_pdb_residue_id2name(protein_dir)

    # read pHs
    if os.path.exists(run_info_path):
        with open(run_info_path, 'r') as json_f:
            run_info_dict = json.load(json_f)
        phs = run_info_dict['phs']
        data_frame_list = []
        for ph in phs:
            lambda_file_path = os.path.join(lambda_dir, '{}_{}.lambda'.format(protein_name, ph))
            data_frame = store_lambda_into_dataframe(lambda_file_path, id_name_dict)
            data_frame_list.append(data_frame)
        multi_lambda_data = pd.concat(objs=data_frame_list, keys=phs)

    return multi_lambda_data


def pka_apply_function(S_group, args):
    """
    This function use for DataFrame apply function, it will caculate pka for each titrable residue.
    :param S_col: DataFrame, a series S vaues of one titrable, it's index is (phs, ires_id, ires_name),
                            it's column is 0(only 0), it's value is S value.
    :param model_pkas:  Series, a series model pka values of all titrable residue ids, it's value is
                                model pka, use for fitting S-curve, it's index is titrable residue ids,
                                it's value is model pkas.
    :return pka: float, pka value.
    """
    # get pHs
    phs = list(set(S_group.index.get_level_values(0).astype(float)))
    phs.sort()
    # get residue id
    ires_id = S_group.index.get_level_values(1)[0]
    ires_name = S_group.index.get_level_values(2)[0]
    # change string index into float index
    S_group.index = S_group.index.map(lambda x: float(x[0]))
    # sort S_col according index and get list of s_values
    s_list = list(S_group[0].sort_index(level=0))
    # get model pka
    model_pkas = args[0]
    model_pka = model_pkas.loc[ires_id].loc[ires_name]
    # get pka
    fit_formula = henderson_hasselbalch_for_fitting
    hill, pka = compute_pka_and_hill(fit_formula, phs, s_list, model_pka)
    return pka


def compute_pkas(multi_lambda_data, cut_max=0.8, cut_min=0.2):
    """
    This function, will read lambda data, and cpmpute pka value for every titrable residue.
    :param multi_lambda_data: DataFrame, multi index DataFrame,
                        first level index is pHs, second level index is residue ids,
                        third index is residue names, columns is steps, value is lambda.
    :param cut_max: float, threshold, if lambda value large than cut_max, it's mean deprotonation,
    :param cut_min: float, threshold, if lambda value small than cut_min, it's mean protonamtion.
    :return: pkas: Series, index is titrable (residue ids, residue names), value is pka.
    """

    # depro_count: count number of lambda value larger than 'CUT_MAX'
    depro_count = multi_lambda_data[multi_lambda_data >= cut_max].count(axis=1)
    # depro_count: count number of lambda value smaller than 'CUT_MIN'
    pro_count = multi_lambda_data[multi_lambda_data <= cut_min].count(axis=1)
    # S = depro / (depro + pro)
    S_values = (depro_count / (pro_count + depro_count)).replace([np.inf, -np.inf], 0)
    # caculate model pka
    model_pkas = (S_values - 0.5).abs().groupby(level=[1, 2]).idxmin().map(lambda x: float(x[0]))
    # caculate pka, hill
    pkas = S_values.to_frame().groupby(level=[1, 2]).apply(pka_apply_function, (model_pkas,))

    return pkas


def calculate_sub_pkas(protein_dir, last_sub_time):
    """
    This function will read protein data saved in protein directory, and then calculate the pka
    distance between total time and  less 'last_sub_time' than total time. The time mean 
    molecular dynamics simulation time, different simulation will calculate different pka.
    :param protein_dir: String, the directory saved protein data.
    :param last_sub_time: Float, the sub time of molecular dynamics simulation time.
    :return: sub_pkas: Series, the first index is titrable residue id, the second index is
                        titrable residue name, the value is the pka distance between total
                        time and  less 'last_sub_time' than total time.
    """
    sub_pkas = None
    # delete last '/'
    if protein_dir[-1] == '/':
        protein_dir = protein_dir[:-1]
    protein_name = protein_dir.split('/')[-1].split('_')[0]
    input_dir = os.path.join(protein_dir, '{}_input'.format(protein_name))
    run_info_path = os.path.join(input_dir, '{}_run_info.json'.format(protein_name))

    # read NPrint_PHMD
    if os.path.exists(run_info_path):
        with open(run_info_path, 'r') as json_f:
            run_info_dict = json.load(json_f)
        NPrint_PHMD = run_info_dict['NPrint_PHMD']
        dt = 2e-6   # every simulate step using time
        distance_steps = int(last_sub_time / dt)
        total_multi_lambda_data = store_lambda_into_multi_index_dataframe(protein_dir)
        last_steps = total_multi_lambda_data.columns.max()
        less_steps = last_steps - distance_steps

        if less_steps <= 0:
            raise ValueError('last sub time {} ns too long for protein {}!'.format(last_sub_time, protein_name.upper()))
        else:
            less_multi_lambda_data = total_multi_lambda_data.loc[:, :less_steps]
            total_pkas = compute_pkas(total_multi_lambda_data)
            less_pkas = compute_pkas(less_multi_lambda_data)
            sub_pkas = (total_pkas - less_pkas).abs()

    return sub_pkas

def calculate_cut_pkas(protein_dir, cut_time):
    """
    This function will read protein data saved in protein directory, and then calculate the pka
    with cut time steps.
    :param protein_dir: String, the directory saved protein data.
    :param cut_time: Float, the cut time of molecular dynamics simulation time.
    :return: cut_pkas: Series, the first index is titrable residue id, the second index is
                        titrable residue name, the value is the pka distance between total
                        time and  less 'last_sub_time' than total time.
    """
    cut_pkas = None
    # delete last '/'
    if protein_dir[-1] == '/':
        protein_dir = protein_dir[:-1]
    protein_name = protein_dir.split('/')[-1].split('_')[0]
    input_dir = os.path.join(protein_dir, '{}_input'.format(protein_name))
    run_info_path = os.path.join(input_dir, '{}_run_info.json'.format(protein_name))

    # read NPrint_PHMD
    if os.path.exists(run_info_path):
        with open(run_info_path, 'r') as json_f:
            run_info_dict = json.load(json_f)
        NPrint_PHMD = run_info_dict['NPrint_PHMD']
        dt = 2e-6   # every simulate step using time
        total_multi_lambda_data = store_lambda_into_multi_index_dataframe(protein_dir)
        last_steps = total_multi_lambda_data.columns.max()
        cut_steps = int(cut_time / dt)

        if last_steps < cut_steps:
            raise ValueError('cut time {} ns too large for protein {}!'.format(cut_time, protein_name.upper()))
        else:
            less_multi_lambda_data = total_multi_lambda_data.loc[:, :cut_steps]
            cut_pkas = compute_pkas(less_multi_lambda_data)

    return cut_pkas


def get_relative_value(protein_dir):
    """
    The titrable residue id between pdb files and lambda files is different, so this function will read pdb and lambda
    files, than calculate the relative distance between lambda residue id and pdb residue id with same residue.
    :param protein_dir: String, the directory contain molecular dynamics simulate data.
    :return relative_value: Integer, the relative distance between lambda residue id
                            and pdb residue id with same residue.
    """
    relative_value = None
    # delete last '/'
    if protein_dir[-1] == '/':
        protein_dir = protein_dir[:-1]
    protein_name = protein_dir.split('/')[-1].split('_')[0]
    input_dir = os.path.join(protein_dir, '{}_input'.format(protein_name))
    lambda_dir = os.path.join(protein_dir, '{}_lambda'.format(protein_name))
    run_info_path = os.path.join(input_dir, '{}_run_info.json'.format(protein_name))
    pdb_path = os.path.join(input_dir, '{}.pdb'.format(protein_name))

    # read pHs
    if os.path.exists(run_info_path):
        with open(run_info_path, 'r') as json_f:
            run_info_dict = json.load(json_f)
        phs = run_info_dict['phs']

        # read pdb first titrable residue id
        with open(pdb_path, 'r') as f:
            line_str = f.readline()
            while(line_str):
                line_list = split_pdb_line(line_str)
                if line_list[0] == 'ATOM':
                    if line_list[3][-3:] in CAN_DEPRO_IERS_NAMES:
                        first_ires_id = int(line_list[5])
                        break
                line_str = f.readline()

        # read lambda first titrable residue id
        one_lambda_path = os.path.join(lambda_dir, '{}_{}.lambda'.format(protein_name, str(phs[0])))
        with open(one_lambda_path, 'r') as f:
            line_str = f.readline()
            while line_str:
                line_list = line_str.split()
                if line_list[1] == 'ires':
                    first_changed_ires_id = int(line_list[2])
                    break
                line_str = f.readline()
        relative_value = first_ires_id - first_changed_ires_id
    return relative_value



def get_lambda_iresID_list(protein_dir):
    """
    The titrable residue id between pdb files and lambda files is different, so this function will read pdb and lambda
    files, than calculate the relative distance between lambda residue id and pdb residue id with same residue.
    :param protein_dir: String, the directory contain molecular dynamics simulate data.
    :return lambda_iresID_list: List, lambda residue id.
    """
    lambda_iresID_list = []
    # delete last '/'
    if protein_dir[-1] == '/':
        protein_dir = protein_dir[:-1]
    protein_name = protein_dir.split('/')[-1].split('_')[0]
    input_dir = os.path.join(protein_dir, '{}_input'.format(protein_name))
    lambda_dir = os.path.join(protein_dir, '{}_lambda'.format(protein_name))
    run_info_path = os.path.join(input_dir, '{}_run_info.json'.format(protein_name))
    pdb_path = os.path.join(input_dir, '{}.pdb'.format(protein_name))

    # read pHs
    if os.path.exists(run_info_path):
        with open(run_info_path, 'r') as json_f:
            run_info_dict = json.load(json_f)
        phs = run_info_dict['phs']

        one_lambda_path = os.path.join(lambda_dir, protein_name + '_' + str(phs[0]) + '.lambda')
        with open(one_lambda_path, 'r') as f:
            line_str = f.readline()
            while (line_str):
                line_list = line_str.split()
                if line_list[1] == 'ires':
                    lambda_iresID_list = [int(id) for id in line_list[2:]]
                    break
                line_str = f.readline()
        lambda_iresID_list = list(set(lambda_iresID_list))
        lambda_iresID_list.sort()

    return lambda_iresID_list



def get_map_pdb_residue_id2name(protein_dir):
    """
    This functon will read pdb residue id and names, and return a map between them.
    :param protein_dir: String, the directory contain molecular dynamics simulate data.
    :return id_name_dict: Dictionary, the map between residue id and names.
    """
    id_name_dict = {}
    # delete last '/'
    if protein_dir[-1] == '/':
        protein_dir = protein_dir[:-1]
    protein_name = protein_dir.split('/')[-1].split('_')[0]
    input_dir = os.path.join(protein_dir, '{}_input'.format(protein_name))
    pdb_path = os.path.join(input_dir, '{}.pdb'.format(protein_name))

    # read pdb first titrable residue id
    with open(pdb_path, 'r') as f:
        line_str = f.readline()
        while (line_str):
            line_list = split_pdb_line(line_str)
            if line_list[0] == 'ATOM':
                ires_id = int(line_list[5])
                ires_name = line_list[3]
                id_name_dict[ires_id] = ires_name
            line_str = f.readline()
    return id_name_dict


def get_total_pka(total_dir):
    """
    This function will read dir contain all proteins'  molecular dynamics simulate data and, caculate
    their the pka.
    :param total_dir: String, the directory contain all proteins' molecular dynamics simulate data.
    :return total_pkas: Series, the first index is protein dir name, the second index is titrable residue
                            id, the third index in titrable name, the value is the pka value.
    """
    protein_dir_names = os.listdir(total_dir)
    total_pkas = Series()
    protein_series_list = []
    protein_names = []
    for protein_dir_name in protein_dir_names:
        print(protein_dir_name)
        protein_dir = os.path.join(total_dir, protein_dir_name)
        try:
            multi_lambda_data = store_lambda_into_multi_index_dataframe(protein_dir)
            pkas = compute_pkas(multi_lambda_data)
        except Exception as e:
            print(e)
            pkas = None
        if pkas is None:
            print('{} is None.'.format(protein_dir_name))
        else:
            protein_series_list.append(pkas)
            protein_names.append(protein_dir_name)
    total_pkas = pd.concat(protein_series_list, keys=protein_names)
    return total_pkas


def get_total_sub_pka(total_dir, last_sub_time):
    """
    This function will read dir contain all proteins'  molecular dynamics simulate data and, caculate
    their the pka distance between total time and  less 'last_sub_time' than total time.

    :param total_dir: String, the directory contain all proteins' molecular dynamics simulate data.
    :param last_sub_time: Float, the sub time of molecular dynamics simulation time.
    :return total_sub_pkas: Series, the first index is protein dir name, the second index is titrable residue
                            id, the third index in titrable name, the value is the pka distance between total
                        time and  less 'last_sub_time' than total time.
    """
    protein_dir_names = os.listdir(total_dir)
    total_sub_pkas = Series()
    protein_series_list = []
    protein_names = []
    for protein_dir_name in protein_dir_names:
        print(protein_dir_name)
        protein_dir = os.path.join(total_dir, protein_dir_name)
        try:
            sub_pkas = calculate_sub_pkas(protein_dir, last_sub_time=last_sub_time)
        except Exception as e:
            print(e)
            sub_pkas = None
        if sub_pkas is None:
            print('{} is None.'.format(protein_dir_name))
        else:
            protein_series_list.append(sub_pkas)
            protein_names.append(protein_dir_name)
    total_sub_pkas = pd.concat(protein_series_list, keys=protein_names)
    return total_sub_pkas


def get_total_cut_pka(total_dir, cut_time):
    """
    This function will read dir contain all proteins'  molecular dynamics simulate data and, caculate
    their the pka distance between total time and  less 'last_sub_time' than total time.

    :param total_dir: String, the directory contain all proteins' molecular dynamics simulate data.
    :param last_sub_time: Float, the sub time of molecular dynamics simulation time.
    :return total_sub_pkas: Series, the first index is protein dir name, the second index is titrable residue
                            id, the third index in titrable name, the value is the pka distance between total
                        time and  less 'last_sub_time' than total time.
    """
    protein_dir_names = os.listdir(total_dir)
    total_cut_pkas = Series()
    protein_series_list = []
    protein_names = []
    for protein_dir_name in protein_dir_names:
        print(protein_dir_name)
        protein_dir = os.path.join(total_dir, protein_dir_name)
        try:
            cut_pkas = calculate_cut_pkas(protein_dir, cut_time=cut_time)
        except Exception as e:
            print(e)
            cut_pkas = None
        if cut_pkas is None:
            print('{} is None.'.format(protein_dir_name))
        else:
            protein_series_list.append(cut_pkas)
            protein_names.append(protein_dir_name)
    total_cut_pkas = pd.concat(protein_series_list, keys=protein_names)
    return total_cut_pkas


def get_total_protein_chain_series(total_dir):
    """
    This function will read dir contain all proteins'  molecular dynamics simulate data,
    and find their chain information, then saved the information in series.
    :param total_dir: String, the directory contain all proteins' molecular dynamics simulate data.
    :return chain_info: Series,  the index is titrable residue id, the value is chain name.
    """
    chain_info = Series()
    protein_dir_names = os.listdir(total_dir)
    for protein_dir_name in protein_dir_names:
        print('get protein chain: {}'.format(protein_dir_name) )
        protein_dir = os.path.join(total_dir, protein_dir_name)
        protein_name = protein_dir.split('/')[-1].split('_')[0]
        input_dir = os.path.join(protein_dir, '{}_input'.format(protein_name))
        pdb_path = os.path.join(input_dir, '{}.pdb'.format(protein_name))

        # read pdb first titrable residue id
        if os.path.exists(pdb_path):
            with open(pdb_path, 'r') as f:
                line_str = f.readline()
                while (line_str):
                    line_list = split_pdb_line(line_str)
                    if line_list[0] == 'ATOM':
                        chain_info[protein_name.upper()] = line_list[4]
                        break
                    line_str = f.readline()
    return chain_info


def screen_convergent_residue_to_csv(total_dir, last_sub_time, choosed_ires_names, save_csv_path):
    """
    This function will screen convergent residue info , then saved in csv files.
    :param total_dir: String, the directory contain all proteins' molecular dynamics simulate data.
    :param last_sub_time: Float, the sub time of molecular dynamics simulation time.
    :param choosed_ires_names: List, must be a subset of ['ASP', 'LYS', 'GLU', 'HIS', 'CYS'].
    :return: None
    """
    # read lambda files, caculate all titrable residue pkas
    total_pkas = get_total_pka(total_dir=total_dir)

    # read lambda files, caculate all titrable residue sub pkas
    total_sub_pkas = get_total_sub_pka(total_dir=total_dir, last_sub_time=last_sub_time)

    # select convergent titrable residue id and pka.
    max_cut = 0.2  # when sub pka less then max cut, we cognizance the titrable resudue is convergent.
    indexes = total_sub_pkas[total_sub_pkas < max_cut].dropna().index
    convergent_pka = total_pkas[indexes]

    # select choosed titrable residue
    idx = pd.IndexSlice
    convergent_pka = convergent_pka.loc[idx[:, :, choosed_ires_names]]

    # add chain info in pka DataFrame
    chain_info = get_total_protein_chain_series(total_dir)
    convergent_pka = convergent_pka.to_frame()
    convergent_pka['Chain'] = None  # add new column
    filenames = [filename for filename in list(set(convergent_pka.index.get_level_values(level=0)))]

    # add value in column 'Chain'
    for filename in filenames:
        pdb_id = filename.split('_')[0].upper()
        convergent_pka.loc[filename]['Chain'] = chain_info[pdb_id]
    convergent_pka = convergent_pka.reset_index()
    convergent_pka = convergent_pka.rename(columns={'level_0': 'PDB ID', 'level_1': 'Res ID',
                                                    'level_2': 'Res Name', 0: 'pKa'})
    # change value of column 'PDB ID'
    convergent_pka['PDB ID'] = convergent_pka['PDB ID'].map(lambda x: x.split('_')[0][0:4].upper())

    # save info in csv file.
    convergent_pka.to_csv(save_csv_path, index=False)




def show_pka(total_pkas, choosed_ires_names, save_path=None):
    """
    This function will show a plot of distribution of total pka.
    :param total_pkas:  Series, the first index is protein dir name, the second index is titrable residue
                        id, the third index in titrable name, the value is the pka value.
    :param choosed_ires_names:  List, must be a subset of ['ASP', 'LYS', 'GLU', 'HIS', 'CYS']
    :param save_path: String, the save path of plot, if None, the function will not save plot.
    :return: None
    """
    max_show = 20
    idx = pd.IndexSlice
    total_pkas = total_pkas.loc[idx[:, :, choosed_ires_names]]  # choose residue names.
    total_pkas[total_pkas < max_show].hist(bins=1000, figsize=(20, 15))
    plt.title('total proteins with {}'.format(choosed_ires_names))
    print('*' * 50)
    print('choosed titrable residue names: {}'.format(', '.join(choosed_ires_names)))
    print('count total: {}'.format(len(total_pkas)))
    print('count NaN: {}'.format(total_pkas.isnull().sum()))
    # cut_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for cut_value in cut_values:
    #     print('less than {} count: {}'.format(cut_value,
    #                                           total_pkas[total_pkas <= cut_value].count()))
    if not save_path is None:
        plt.savefig(save_path)
    plt.show()


def show_sub_pka(total_sub_pkas, choosed_ires_names, save_path=None, max_show=20):
    """
    This function will show the distribution of choosed residue's sub_pka.
    :param total_sub_pkas: Series, the first index is protein dir name, the second index is titrable residue
                            id, the third index in titrable name, the value is the pka distance between total
                        time and  less 'last_sub_time' than total time.
    :param choosed_ires_names: List, must be a subset of ['ASP', 'LYS', 'GLU', 'HIS', 'CYS']
    :param max_show: The max value of distance showed in graph.
    :return: None.
    """
    idx = pd.IndexSlice
    total_sub_pkas = total_sub_pkas.loc[idx[:, :, choosed_ires_names]]     # choose residue names.
    total_sub_pkas[total_sub_pkas < max_show].hist(bins=1000, figsize=(10, 7.5))
    plt.title('total proteins with {}'.format(choosed_ires_names))
    print('*' * 50)
    print('choosed titrable residue names: {}'.format(', '.join(choosed_ires_names)))
    print('count total: {}'.format(len(total_sub_pkas)))
    print('count NaN: {}'.format(total_sub_pkas.isnull().sum()))
    cut_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8, 0.9, 1.0]
    for cut_value in cut_values:
        print('less than {} count: {}'.format(cut_value,
                                              total_sub_pkas[total_sub_pkas <= cut_value].count()))
    plt.xlabel('pka abs diff')
    plt.ylabel('num')
    plt.title(choosed_ires_names)
    if not save_path is None:
        plt.savefig(save_path)
    plt.show()


def test_read_3i7j():
    protein_dir = '/media/czt/My Passport/czt/data/total_output/3i7j_output_replica'
    sub_pkas = calculate_sub_pkas(protein_dir, 0.5)
    print(sub_pkas)


def test_read_3vep():
    protein_dir = '/media/czt/My Passport/czt/data/total_output/3vep_output_replica'
    sub_pkas = calculate_sub_pkas(protein_dir, 0.5)
    print(sub_pkas)


def test_show_sub_pkas():
    total_dir = '/media/czt/My Passport/czt/data/total_output/'
    last_sub_time = 0.5
    total_sub_pkas = get_total_sub_pka(total_dir, last_sub_time)
    max_show = 1
    choosed_ires_names_list = [['ASP', 'LYS', 'GLU', 'HIS'],
                               ['ASP'], ['LYS'], ['GLU'], ['HIS']]
    save_path_list = ['/media/czt/My Passport/czt/img/5type_sub_pka.jpg',
                      '/media/czt/My Passport/czt/img/ASP_sub_pka.jpg',
                      '/media/czt/My Passport/czt/img/LYS_sub_pka.jpg',
                      '/media/czt/My Passport/czt/img/GLU_sub_pka.jpg',
                      '/media/czt/My Passport/czt/img/HIS_sub_pka.jpg']
    for idx, choosed_ires_names in enumerate(choosed_ires_names_list):
        show_sub_pka(total_sub_pkas, choosed_ires_names, save_path=save_path_list[idx], max_show=max_show)


def test_show_pkas():
    total_dir = '/media/czt/My Passport/czt/data/total_output/'
    total_pkas = get_total_pka(total_dir)
    # pka_csv = '/home/czt/data/pka_data/CpHMD_pka.csv'
    # total_pkas = pd.read_csv(pka_csv)
    choosed_ires_names_list = [['ASP', 'LYS', 'GLU', 'HIS', 'CYS'],
                               ['ASP'], ['LYS'], ['GLU'], ['HIS'], ['CYS']]
    save_path_list = ['/media/czt/My Passport/czt/img/5type_pka.jpg',
                      '/media/czt/My Passport/czt/img/ASP_pka.jpg',
                      '/media/czt/My Passport/czt/img/LYS_pka.jpg',
                      '/media/czt/My Passport/czt/img/GLU_pka.jpg',
                      '/media/czt/My Passport/czt/img/HIS_pka.jpg',
                      '/media/czt/My Passport/czt/img/CYS_pka.jpg']
    for idx, choosed_ires_names in enumerate(choosed_ires_names_list):
        show_pka(total_pkas, choosed_ires_names, save_path=save_path_list[idx])


def test_get_ASP_shiffe_large():
    total_dir = '/media/czt/My Passport/czt/data/total_output/'
    total_pkas = get_total_pka(total_dir)
    idx = pd.IndexSlice
    choosed_ires_names = ['ASP']
    total_pkas = total_pkas.loc[idx[:, :, choosed_ires_names]]  # choose residue names.
    total_pkas = total_pkas[total_pkas > 7]
    print(total_pkas)


def test_get_total_cut_pka():
    total_dir = '/home/huang/newdisk01/czt/data/total_output'
    cut_time = 1.5
    cut_df = get_total_cut_pka(total_dir=total_dir, cut_time=cut_time)
    cut_df = cut_df.reset_index().rename(
        columns={'level_0': 'file_name', 'level_1': 'Res ID', 'level_2': 'Res Name', '0': '1.5 ns pKa'})
    df = get_total_pka(total_dir=total_dir)
    df = df.reset_index().rename(
        columns={'level_0': 'file_name', 'level_1': 'Res ID', 'level_2': 'Res Name', '0': '2.0 ns pKa'})
    cut_df.to_csv('./train_data_info/train_chimera_cut_1_5.csv', index=False)
    df.to_csv('./train_data_info/train_chimera_cut_2_0.csv', index=False)


def count_csv_info():
    global model_pka
    cut_csv = './train_data_info/train_chimera_cut_1_5.csv'
    total_csv = './train_data_info/train_chimera_cut_2_0.csv'
    cut_df = pd.read_csv(cut_csv)
    total_df = pd.read_csv(total_csv)
    cut_df = cut_df.rename(columns={'file_name': 'PDB ID', '0': '1.5 ns pKa'})
    total_df = total_df.rename(columns={'file_name': 'PDB ID', '0': '2.0 ns pKa'})
    cut_df['PDB ID'] = cut_df['PDB ID'].map(lambda x: x.split('_')[0][0:4].upper())
    total_df['PDB ID'] = total_df['PDB ID'].map(lambda x: x.split('_')[0][0:4].upper())

    choose_residues = ['ASP', 'GLU', 'HIS', 'LYS']
    merge_df = pd.merge(cut_df, total_df)
    merge_df = merge_df[merge_df['Res Name'].map(lambda x: x in choose_residues)]
    merge_df['sub pKa'] = merge_df['2.0 ns pKa'] - merge_df['1.5 ns pKa']
    merge_df['abs sub pKa'] = merge_df['sub pKa'].abs()
    merge_df['model pKa'] = merge_df['Res Name'].map(lambda x: model_pka[x])
    merge_df['2.0 ns shift pKa'] = merge_df['2.0 ns pKa'] - merge_df['model pKa']
    merge_df['2.0 ns abs shift pKa'] = merge_df['2.0 ns shift pKa'].abs()
    merge_df.to_csv('./train_data_info/train_chimera_residues_info.csv', index=False)

    merge_df_asp = merge_df[merge_df['Res Name'] == 'ASP']
    merge_df_glu = merge_df[merge_df['Res Name'] == 'GLU']
    merge_df_his = merge_df[merge_df['Res Name'] == 'HIS']
    merge_df_lys = merge_df[merge_df['Res Name'] == 'LYS']

    merge_df_asp.to_csv('./train_data_info/train_chimera_resudues_info_asp.csv', index=False)
    merge_df_glu.to_csv('./train_data_info/train_chimera_resudues_info_glu.csv', index=False)
    merge_df_his.to_csv('./train_data_info/train_chimera_resudues_info_his.csv', index=False)
    merge_df_lys.to_csv('./train_data_info/train_chimera_resudues_info_lys.csv', index=False)



def main():

    test_get_total_cut_pka()
    # test_read_3vep()
    # test_show_pkas()
    # test_get_ASP_shiffe_large()
    # test_show_pkas()
    count_csv_info()

if __name__ == '__main__':
    main()