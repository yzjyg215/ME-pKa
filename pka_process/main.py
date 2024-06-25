import os
import random
import shutil
from preprocess_csv import save_protein_features_as_csv, clean_csv, model_pka
from preprocess_pdb import download_pdb_from_csv, crop_pdb_ires_v1, copy_CpHMD_pdb, fix_pdb_batch, extract_single_chain, \
    extract_first_fixed_constant_chain, download_pdb_from_txt, protonated_and_charged
from preprocess_lambda import screen_convergent_residue_to_csv
import pandas as pd
import math


def devide_train_val_test_set(input_pdb_dir, output_pdb_dir, train_weight=6, val_weight=1, test_weight=1):
    """
    this function will devide one dir contains crop pdb files in three dir : train, validate, test in output dir.
    :param train_weight: Integer or float, weight of training set.
    :param val_weight: Integer or float, weight of validate set.
    :param test_weight: Integer or float, weight of test set.
    :return:
    """
    # caculate how many rate of train, val, test set.
    total_weight = train_weight + val_weight + test_weight
    train_rate = train_weight / total_weight
    val_rate = val_weight / total_weight
    test_rate = test_weight / total_weight

    # caculate how many files of train, val, test set.
    total_filenames = os.listdir(input_pdb_dir)
    total_num = len(total_filenames)
    train_num = int(train_rate * total_num)
    val_num = int(val_rate * total_num)
    test_num = total_num - train_num - val_num

    # devide total date set
    random.shuffle(total_filenames)  # disorder date set
    if not os.path.exists(output_pdb_dir):
        os.mkdir(output_pdb_dir)
    else:
        shutil.rmtree(output_pdb_dir)
        os.mkdir(output_pdb_dir)
    devide_list = ['train', 'val', 'test']
    devide_area = {
        'train': [0, train_num],
        'val': [train_num, train_num + val_num],
        'test': [train_num + val_num, total_num]
    }
    for devide_type in devide_list:
        save_dir = os.path.join(output_pdb_dir, devide_type)
        start_idx = devide_area[devide_type][0]
        end_idx = devide_area[devide_type][1]
        if start_idx != end_idx:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for filename in total_filenames[start_idx:end_idx]:
                filepath = os.path.join(input_pdb_dir, filename)
                distpath = os.path.join(save_dir, filename)
                shutil.copy(filepath, distpath)  # copy file
                print('{} -copy-> {}'.format(filename, distpath))


def pre_trainsform_data(in_csv_file, out_csv_file, change_distribution=False, in_cphmd=False):
    """
    this function will trainsform data.
    :param in_csv_file: String, input csv files.
    :param out_csv_file: String, transformed csv files.
    :return: None.
    """
    # choose_features = ['idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
    #                    'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'is_center_residue', 'res_type',
    #                    'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']

    choose_features = ['idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                       'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'is_center_residue',
                       'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']

    # choose_res_name = ['GLU', 'ASP', 'HIS', 'LYS', 'CYS']
    choose_res_name = ['GLU', 'ASP', 'HIS', 'LYS']
    csv_df = pd.read_csv(in_csv_file)

    # select choosed residue
    csv_df['choose'] = csv_df['file_name'].apply(lambda x: x.split('_')[3] in choose_res_name)
    csv_df = csv_df.loc[csv_df['choose']]

    # choose residue in in CpHMD predicted residues
    if in_cphmd:
        cphmd_predict_csv_path = '/home/huang/newdisk01/czt/data/pka_data_new/cphmd_csv/CpHMD_pka_WT69.csv'
        cphmd_df = pd.read_csv(cphmd_predict_csv_path)
        csv_df['PDB ID'] = csv_df['file_name'].apply(lambda x: x.split('_')[0])
        csv_df['Res ID'] = csv_df['file_name'].apply(lambda x: int(x.split('_')[2]))
        csv_df['Res Name'] = csv_df['file_name'].apply(lambda x: x.split('_')[3])
        csv_df['Chain'] = csv_df['file_name'].apply(lambda x: x.split('_')[1])
        csv_df = pd.merge(csv_df, cphmd_df)


    # change distribution
    if change_distribution:
        temp_csv_df = csv_df.loc[:, ['file_name', 'pka']].drop_duplicates()
        temp_csv_df['Res Name'] = temp_csv_df['file_name'].apply(lambda x: x.split('_')[3])
        temp_csv_df['model pKa'] = temp_csv_df['Res Name'].apply(lambda x: model_pka[x])
        temp_csv_df['shift pKa abs'] = (temp_csv_df['pka'] - temp_csv_df['model pKa']).abs()
        print('all count:{}'.format(temp_csv_df.count()))
        temp_csv_df1 = temp_csv_df[temp_csv_df['shift pKa abs'] > 2]
        temp_csv_df2 = temp_csv_df[(temp_csv_df['shift pKa abs'] > 1.5) & (temp_csv_df['shift pKa abs'] <= 2)]
        temp_csv_df3 = temp_csv_df[(temp_csv_df['shift pKa abs'] > 1) & (temp_csv_df['shift pKa abs'] <= 1.5)]
        temp_csv_df4 = temp_csv_df[(temp_csv_df['shift pKa abs'] > 0.5) & (temp_csv_df['shift pKa abs'] <= 1)]
        temp_csv_df5 = temp_csv_df[temp_csv_df['shift pKa abs'] <= 0.5]

        print('>2 count:{}'.format(temp_csv_df1.count()))
        print('2> >1.5 count:{}'.format(temp_csv_df2.count()))
        print('1.5> >1 count:{}'.format(temp_csv_df3.count()))
        print('1> >0.5 count:{}'.format(temp_csv_df4.count()))
        print('< 0.5 count:{}'.format(temp_csv_df5.count()))

        temp_csv_df = pd.concat([temp_csv_df1, temp_csv_df2,
                                 temp_csv_df3.sample(n=43, random_state=1),
                                 temp_csv_df4.sample(n=43, random_state=1),
                                 temp_csv_df5.sample(n=43, random_state=1)])
        csv_df = pd.merge(temp_csv_df, csv_df)

    # select choosed features
    csv_df = csv_df.loc[:, choose_features].drop_duplicates()
    csv_df.to_csv(out_csv_file, index=False)
    print('num of data: ', csv_df['idx'].drop_duplicates().count())
    print('new csv data saved in {}'.format(out_csv_file))


def preprocess_CpHMD_data():
    """
    This function will pretreat process of CpHMD data.
    The process include:
    :return:
    """
    # screen convergent residue info and save it into csv file
    # total_dir = '/home/czt/data/CpHMD_data'
    total_dir = '/home/huang/newdisk01/czt/data/total_output'
    last_sub_time = 0.5
    # choosed_ires_names = ['ASP', 'GLU', 'LYS', 'HIS', 'CYS']
    choosed_ires_names = ['ASP', 'GLU', 'LYS', 'HIS']
    save_csv_path = '/home/huang/newdisk01/czt/data/pka_data_new/cphmd_csv/CpHMD_pka279.csv'
    # screen_convergent_residue_to_csv(total_dir=total_dir, last_sub_time=last_sub_time,
    #                                  choosed_ires_names=choosed_ires_names, save_csv_path=save_csv_path)

    # read csv file and copy the pdb files into one directory
    save_pdb_dir = '/home/huang/newdisk01/czt/data/pka_data_new/data_pdb_CpHMD279'
    copy_CpHMD_pdb(total_dir=total_dir, csv_path=save_csv_path, save_pdb_dir=save_pdb_dir)

    # fix pdb files
    fixed_pdb_dir = '/home/huang/newdisk01/czt/data/pka_data_new/data_pdb_CpHMD279_fixed'
    fix_pdb_batch(save_pdb_dir, fixed_pdb_dir)

    # crop WT protein pdb files with radii 10
    crop_pdb_directory = '/home/huang/newdisk01/czt/data/pka_data_new/data_pdb_CpHMD279_ires/'
    wrong_ires_info_file = '/home/huang/newdisk01/czt/data/pka_data_new/wrong_ires_CpHMD279.txt'
    center_coors_dir = '/home/huang/newdisk01/czt/data/pka_data_new/center_coors_csv/'
    crop_pdb_ires_v1(csv_filename=save_csv_path, pdb_directory=fixed_pdb_dir, center_coors_dir=center_coors_dir,
                     crop_pdb_directory=crop_pdb_directory, wrong_ires_info_file=wrong_ires_info_file, radii=radii)

    # protonate and charge titrable residues
    ires_mol2_dir = '/home/huang/newdisk01/czt/data/pka_data_new/data_pdb_CpHMD279_ires_mol2'
    protonated_and_charged(crop_pdb_directory, ires_mol2_dir)

    # devide train, val, test set.
    devide_save_dir = '/home/huang/newdisk01/czt/data/pka_data_new/CpHMD_ires_mol2_devide/'
    train_weight = 1
    val_weight = 0
    test_weight = 0
    devide_train_val_test_set(input_pdb_dir=ires_mol2_dir, output_pdb_dir=devide_save_dir,
                              train_weight=train_weight, val_weight=val_weight, test_weight=test_weight)

    # read csv file and crop pdb files caculate crop pretein features and saved as a csv file.
    protein_format = 'mol2'
    csv_directory = '../data/preprocess_temp_data/features_csv'    # contain residues data for DeepKa
    proteins = [os.path.join(devide_save_dir, 'train'),
                os.path.join(devide_save_dir, 'val'),
                os.path.join(devide_save_dir, 'test')]
    for protein in proteins:
        if os.path.exists(protein):
            save_protein_features_as_csv(protein=protein, protein_format=protein_format, output_dir=csv_directory,
                                         pka_csv=save_csv_path)

    # protonate and charge proteins
    protein_mol2_dir = '../data/preprocess_temp_data/CPHMD_temp_data/data_pdb_CpHMD274_fixed_mol2'
    protonated_and_charged(fixed_pdb_dir, protein_mol2_dir)

    # read fixed one chain pdb files caculate protein features and save as csv file.
    csv_directory = '../data/preprocess_temp_data/fixed_single_chain_features_csv'     # contain protein data for DeepKa
    save_protein_features_as_csv(protein=protein_mol2_dir, protein_format=protein_format, output_dir=csv_directory)
    print('finished process')


def preprocess_expt_data():
    """
    This function will pretreat process of experiment data.
    The process include:
    1.  download source protein pdb files.
    2.  crop protein pdb files with radii 10.
    3.  devide train, val, test set.
    4.  read csv file and crop pdb files and caculate crop pretein features and saved as a csv file.
    ......
    :return: None
    """
    # clean csv file
    #input_csv = '/home/huang/newdisk01/czt/data/pka_data_new/expt_input_csv/WT_pka.csv'
    #cleaned_csv = '../data/cleaned_source_data/expt_cleaned_csv/final_expt_pka.csv'
    cleaned_csv = '/data2/rymiao/propka/PKAD2.csv'
    # clean_csv(input_csv=input_csv, cleaned_csv=cleaned_csv)

    # download WT protein pdb files
    #save_pdb_directory = '../data/cleaned_source_data/data_pdb_WT_accurate'
    save_pdb_directory = '/data2/rymiao/propka/PKAD2_PDB2'
    download_pdb_from_csv(csv_filename=cleaned_csv, save_pdb_directory=save_pdb_directory)

    # extract single chain
    single_chain_pdb_dir = '/data2/rymiao/propka/pka_process/temp/data_pdb_expt_single_chain'
    extract_single_chain(csv_file=cleaned_csv, pdb_dir=save_pdb_directory, single_chain_pdb_dir=single_chain_pdb_dir)

    # fix pdb files (only can fix single chain pdb file)
    fixed_pdb_directory = '/data2/rymiao/propka/pka_process/temp/EXPT_temp_data/data_pdb_WT_fixed/'
    fix_pdb_batch(single_chain_pdb_dir, fixed_pdb_directory)

    # crop WT protein pdb files with radii 10
    #radii=10
    crop_pdb_directory = '/data2/rymiao/propka/pka_process/temp/EXPT_temp_data/data_pdb_WT_ires'
    wrong_ires_info_file = '/data2/rymiao/propka/pka_process/temp/EXPT_temp_data/wrong_ires_WT.txt'
    center_coors_dir = '/data2/rymiao/propka/pka_process/temp/center_coors_csv/'
    # crop WT protein pdb files with radii
    # crop_pdb_directory = '/home/huang/newdisk01/czt/data/pka_data_new/data_pdb_WT_ires_rd11/'
    # wrong_ires_info_file = '/home/huang/newdisk01/czt/data/pka_data_new/wrong_ires_WT.txt'
    # center_coors_dir = '/home/huang/newdisk01/czt/data/pka_data_new/center_coors_csv/'
    crop_pdb_ires_v1(csv_filename=cleaned_csv, pdb_directory=fixed_pdb_directory, center_coors_dir=center_coors_dir,
                     crop_pdb_directory=crop_pdb_directory, wrong_ires_info_file=wrong_ires_info_file)

    # protonate and charge titrable residues
    ires_mol2_dir = '/data2/rymiao/propka/pka_process/temp/EXPT_temp_data/data_pdb_WT_ires_mol2'
    protonated_and_charged(crop_pdb_directory, ires_mol2_dir)

    # devide train, val, test set.
    devide_save_dir = '/data2/rymiao/propka/pka_process/temp/EXPT_temp_data/WT_ires_mol2_devide/'
    train_weight = 8
    val_weight = 1
    test_weight = 1
    devide_train_val_test_set(input_pdb_dir=ires_mol2_dir, output_pdb_dir=devide_save_dir,
                              train_weight=train_weight, val_weight=val_weight, test_weight=test_weight)

    # read csv file and mol2 files caculate crop pretein features and saved as a hdf file.
    protein_format = 'mol2'
    csv_directory = '/data2/rymiao/propka/pka_process/temp/features_csv'
    proteins = [os.path.join(devide_save_dir, 'train'),
                os.path.join(devide_save_dir, 'val'),
                os.path.join(devide_save_dir, 'test')]
    for protein in proteins:
        if os.path.exists(protein):
            save_protein_features_as_csv(protein=protein, protein_format=protein_format, output_dir=csv_directory,
                                         pka_csv=cleaned_csv)

    # protonate and charge proteins
    protein_mol2_dir = '/data2/rymiao/propka/pka_process/temp/EXPT_temp_data/data_pdb_WT_fixed_mol2/'
    protonated_and_charged(fixed_pdb_directory, protein_mol2_dir)

    # read fixed one chain mol2 files calculate protein features and save as csv file.
    csv_directory = '/data2/rymiao/propka/pka_process/temp/fixed_single_chain_features_csv'
    save_protein_features_as_csv(protein=protein_mol2_dir, protein_format=protein_format, output_dir=csv_directory)
    print('finished process')


def preprocess_CpHMD_input():
    txt_path = '/home/czt/data/pka_data_new/12859_2018_2280_MOESM1_ESM.txt'
    save_dir_path = '/media/czt/My Passport/czt/data/pka_data_new/swissmodel/pdb_swiss'
    # download_pdb_from_txt(txt_path, save_dir_path)

    fix_constraint_pdb = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_swiss_single_chain'
    extract_first_fixed_constant_chain(save_dir_path, fix_constraint_pdb)


def preprocess_expt_CpHMD_input():
    csv_file_path = '/media/czt/My Passport/czt/data/pka_data_new/expt_cleaned_csv/WT_cleaned_pka2.csv'
    pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_WT_SSBOND'
    single_chain_pdb_dir = '/media/czt/My Passport/czt/data/pka_data_new/data_pdb_WT_SSBOND_single_chain'
    extract_single_chain(csv_file_path, pdb_dir, single_chain_pdb_dir)


def get_CpHMD_pka_on_WT_data():
    # screen convergent residue info and save it into csv file
    total_dir = '/media/czt/My Passport/czt/data/total_output_WT'
    last_sub_time = 0.5
    # choosed_ires_names = ['ASP', 'GLU', 'LYS', 'HIS', 'CYS']
    choosed_ires_names = ['ASP', 'GLU', 'LYS', 'HIS']
    save_csv_path = '/media/czt/My Passport/czt/data/pka_data_new/cphmd_csv/CpHMD_pka_WT69.csv'
    screen_convergent_residue_to_csv(total_dir=total_dir, last_sub_time=last_sub_time,
                                     choosed_ires_names=choosed_ires_names, save_csv_path=save_csv_path)


def preprocess_features():
    in_features_csv_file = '../data/preprocess_temp_data/features_csv/test.csv'
    out_features_csv_file = '../data/preprocess_temp_data/features_csv/test_cgdtb.csv'
    pre_trainsform_data(in_features_csv_file, out_features_csv_file, change_distribution=True, in_cphmd=False)


def main():
    # preprocess_CpHMD_data()
    # get_CpHMD_pka_on_WT_data()
    preprocess_expt_data()
    # preprocess_CpHMD_input()
    # preprocess_expt_CpHMD_input()
    # preprocess_features()


if __name__ == '__main__':
    main()
