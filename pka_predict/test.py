import torch.nn as nn
import torch
import pandas as pd
from pandas import DataFrame
import torch
import numpy as np
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
import random
from pka_dataset import PkaDatasetCSV, PkaDatasetHDF
from torch.utils.data.dataloader import DataLoader
from evaluate import eval_result, draw_predict_target_scatter


def the_loss(y_pre, y):
    return abs(y_pre - y)


def test_grad():
    x = torch.ones(2, 2, requires_grad=True)
    y = torch.ones(1) * 5
    n1 = torch.ones(2, 2)
    b = torch.ones(2, 2)

    print('x', x)
    print('y', y)
    print('n1', n1)
    print('b', b)

    for i in range(1000):
        z = x @ n1 + b
        y_pre = z.sum()
        loss = the_loss(y_pre, y)
        loss.backward()
        print('**************************')
        print('x.grad', x.grad)
        with torch.no_grad():
            x -= x.grad.clone() * loss * 0.01
        x.grad.zero_()
        print('loss', loss)
        print('x', x)
        print('y', y)
        print('y_pre', y_pre)
        print('n1', n1)
        print('b', b)


def get_result():
    path = '../../../science_work/data/DownloadPDB/data_pdb_WT_ires'
    for i in range(1000):
        yield torch.randn(23, 150).float()


def test_yield():
    i = 0
    temp = get_result()
    temp_list = []

    while True:
        try:
            i += 1
            print(i)
            result = next(temp)
            temp_list.append(result)
        except StopIteration:
            print('done')
            print(temp_list)
            break
    # print(len(temp_list))


def test_panda():
    temp = DataFrame([{'a': 1, 'b': 1.5, 'c': 7},
                      {'a': 1, 'b': 1.5, 'c': 7},
                      {'a': 5, 'b': 6, 'c': 1}])
    # temp = DataFrame([{'a': 1, 'b': 1.5, 'c': 'e'},
    #                   {'a': 1, 'b': 4.5, 'c': 't'},
    #                   {'a': 5, 'b': 6, 'c': 'q'}])
    # temp = DataFrame([{'a': 'a', 'b': 1.5, 'c': 'e'},
    #                   {'a': 'a', 'b': 4.5, 'c': 't'},
    #                   {'a': 'd', 'b': 6, 'c': 'q'}])
    print(temp)
    temp2 = temp.set_index(['a', 'c'])[['b']]
    print(temp2)
    # print(temp2.loc['a'])
    # print(temp2.loc['a'].loc['e'])
    print(temp2.loc[1].loc[7])
    print(temp.duplicated())
    print(temp.drop_duplicates())
    print(temp)


def test_torch():
    # n = np.ones(5)
    # t = torch.from_numpy(n)
    n = [0, 1, 2, 3]
    t = torch.tensor(n)
    np = t.numpy()

    print(n, t, np)
    t[0] = 100
    print(n, t, np)


def test_np():
    a = np.array([1, 2, 3])
    print(a[[0, 2]])


def test_rotation():
    def rotation_matrix(axis, theta):
        """Counterclockwise rotation about a given axis by theta radians"""

        try:
            axis = np.asarray(axis, dtype=np.float)
        except ValueError:
            raise ValueError('axis must be an array of floats of shape (3,)')

        if axis.shape != (3,):
            raise ValueError('axis must be an array of floats of shape (3,)')

        if not isinstance(theta, (float, int)):
            raise TypeError('theta must be a float')

        axis = axis / sqrt(np.dot(axis, axis))
        a = cos(theta / 2.0)
        b, c, d = -axis * sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # Create matrices for all possible 90* rotations of a box
    ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

    # about X, Y and Z - 9 rotations
    for a1 in range(3):
        for t in range(1, 4):
            axis = np.zeros(3)
            axis[a1] = 1
            theta = t * pi / 2.0
            print(axis, theta / pi * 180)
            ROTATIONS.append(rotation_matrix(axis, theta))

    # about each face diagonal - 6 rotations
    for (a1, a2) in combinations(range(3), 2):
        axis = np.zeros(3)
        axis[[a1, a2]] = 1.0
        theta = pi
        print(axis, theta / pi * 180)
        ROTATIONS.append(rotation_matrix(axis, theta))
        axis[a2] = -1.0
        print(axis, theta / pi * 180)
        ROTATIONS.append(rotation_matrix(axis, theta))

    # about each space diagonal - 8 rotations
    for t in [1, 2]:
        theta = t * 2 * pi / 3
        axis = np.ones(3)
        print(axis, theta / pi * 180)
        ROTATIONS.append(rotation_matrix(axis, theta))
        for a1 in range(3):
            axis = np.ones(3)
            axis[a1] = -1
            print(axis, theta / pi * 180)
            ROTATIONS.append(rotation_matrix(axis, theta))


def test_random():
    for i in range(10):
        print(random.randint(6, 8))


def test_dateset():
    data_path = '/media/czt/TOSHIBA SSD/science_work/data/DownloadPDB/hdf/test.hdf'
    is_rotate = False
    data_set = PkaDatasetHDF(data_path=data_path, is_rotate=is_rotate)
    data_loader = DataLoader(dataset=data_set, batch_size=1)
    while True:
        y_list = []
        for i, data in enumerate(data_loader):
            x, y = data
            y_list.append(y)
        print(y[0])
        print(len(y))
        if not data_set.is_empty():
            data_set.batch_load_data()
        else:
            break


def test_view():
    data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(data.view(-1))


def test_for_in():
    a = range(10)
    c = [(temp, temp * 10) for temp in a]

    # print(b)
    print(c)


def test_get_no_zero_index():
    a = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    print(a)
    b = a.nonzero()
    x, y = b
    print()


def test_equare():
    a = pd.DataFrame([[1, 2, 3, 4]])
    a['new'] = 10
    print(a)


def test_find_large_shift_asp():
    a = pd.read_csv('./train_info/model_bn3_f19_99_elu_z-score-all/test_r4_f19_z-score-all.csv')
    # a = a.loc[a['Res Name'] == 'ASP']
    a = a.loc[a['Target pKa shift'].abs() >= 2]
    print(a)


def test_numpy_min_max():
    a = np.array([1, 3, 9])
    b = np.array([2, 4, 6])
    c = np.vstack((a, b))
    # d = np.hstack((a, b))
    d = c.min(0)
    print(c)
    print(d)


def test_get_col_idx():
    a = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 6, 9], 'c': [9, 18, 27]})
    idx = a.columns.get_loc('b')
    print(idx)


def test_rot90():
    a = np.array(range(1, 9)).reshape((2, 2, 2))
    print('a:', a)
    b = np.rot90(a, k=1, axes=(0, 1))
    print('b:', b)


def test_mean_std():
    a = np.array(range(24)).reshape((2, 3, 4))
    print('*' * 30)
    print(a)
    print('*' * 30)
    print(a.mean())
    print('*' * 30)
    print(a.mean(0))
    print('*' * 30)
    print(a.mean(1))
    print('*' * 30)
    print(a.mean((0, 2)))
    print('*' * 30)
    print(a.mean(-1))


def test_fill():
    a = np.array([1, 2, 3])
    c = np.zeros((2, 3))
    b = c.fill(a)
    print(b)


def test_expand_dimension():
    a = np.array([1, 2, 3])
    b = np.expand_dims(a, (0, 2, 3, 4))
    b = b.repeat(2, axis=2)
    np.repeat()
    print(b)


def devide_zero():
    a = np.array([-1, 1, 1, 0])
    b = np.array([0, 1, 0, 0])
    c = a / b
    c[np.isinf(c)] = 0
    print(c)


def test_serise_loc():
    a = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    b = a.loc[['b', 'c']]
    print(b)


def test_np_flaot_type():
    a = np.array([-1., 1.], dtype=np.float16)
    b = np.array([-1, 1], dtype=np.float64)
    print(type(a), type(b))


def test_caculate_mae_muse_r2():
    target_pka = [3.9, 4.5, 6.5, 3.7, 3.7, 3.2, 4.5, 5.4,  # BBL   1W4H
                  3.0, 3.6, 3.1, 4.0, 4.2, 4.2,  # NTL9  1CQU
                  4.8, 4.4, 4.0, 3.8, 9.9, 4.1, 3.3, 5.3, 2.8, 4.2, 3.2, 4.9, 4.6, 3.7, 4.1, 3.9, 4.4,
                  # Thioredoxin   1ERT
                  3.1, 4.0, 3.5, 4.4,  # HP36  1VII
                  2.8, 4.0, 6.2, 2.0, 3.5, 6.0, 4.7, 3.9, 3.5, 4.1, 6.7, 3.5, 6.1, 3.1,  # RNase A   7RSA
                  3.0, 2.5, 4.6, 2.7, 3.2, 3.6, 6.5, 6.7,  # Xylanase  1BCX
                  2.7, 4.1, 3.2, 2.3, 4.8, 7.5,  # OMTKY 1OMU
                  4.5, 6.1, 3.6, 4.4, 3.2, 3.9, 7.0, 4.4, 2.6, 5.5, 3.2, 3.2, 4.1, 7.1, 7.9, 3.6, 4.3, 4.1, 4.3, 4.2,
                  4.4,  # RNase H   2RN2
                  5.2, 3.5,  # BACE1 1SGZ
                  2.6, 5.5, 2.8, 6.1, 1.4, 3.6, 1.2, 2.2, 4.5, 3.5,  # HEWL  2LZT
                  6.5, 2.8, 2.2, 6.5, 3.9, 4.3, 3.9, 3.5, 3.8, 3.3, 3.3, 2.2, 3.8, 5.2, 3.9, 3.8, 3.8  # SNASE 3BDC
                  ]
    target_pka2 = [3.9, 4.5, 6.5, 3.7, 3.7, 3.2, 4.5, 5.4,  # BBL   1W4H
                   3.0, 3.6, 3.1, 4.0, 4.2, 4.2,  # NTL9  1CQU
                   4.8, 4.4, 4.0, 3.8, 9.9, 4.1, 3.3, 5.3, 2.8, 4.2, 3.2, 4.9, 4.6, 3.7, 4.1, 3.9, 4.4,
                   # Thioredoxin   1ERT
                   3.1, 4.0, 3.5, 4.4,  # HP36  1VII
                   2.8, 4.0, 6.2, 2.0, 3.5, 6.0, 4.7, 3.9, 3.5, 4.1, 6.7, 3.5, 6.1, 3.1,  # RNase A   7RSA
                   3.0, 2.5, 4.6, 2.7, 3.2, 3.6, 6.5,  # Xylanase  1BCX
                   2.7, 4.1, 3.2, 2.3, 4.8, 7.5,  # OMTKY 1OMU
                   4.5, 6.1, 3.6, 4.4, 3.2, 3.9, 7.0, 4.4, 2.6, 5.5, 3.2, 3.2, 4.1, 7.1, 7.9, 3.6, 4.3, 4.1, 4.3, 4.2,
                   4.4,  # RNase H   2RN2
                   5.2, 3.5,  # BACE1 1SGZ
                   2.6, 5.5, 2.8, 6.1, 1.4, 3.6, 1.2, 2.2, 4.5, 3.5,  # HEWL  2LZT
                   6.5, 2.8, 2.2, 6.5, 3.9, 4.3, 3.9, 3.5, 3.8, 3.3, 3.3, 2.2, 3.8, 5.2, 3.9, 3.8, 3.8  # SNASE 3BDC
                   ]
    # others predict
    # MAE: 0.6345132743362834, MUSE: 0.8598723922614631, R2: 0.6140808189646363
    predict_pka = [2.8, 4.1, 6.9, 2.6, 3.3, 3.2, 4.0, 6.0,  # BBL   1W4H
                   2.1, 3.4, 2.9, 3.6, 3.8, 3.8,  # NTL9  1CQU
                   3.9, 4.4, 4.0, 2.9, 6.2, 4.3, 4.5, 3.8, 3.6, 4.6, 3.1, 4.3, 5.0, 3.8, 3.5, 3.9, 4.7,
                   # Thioredoxin   1ERT
                   2.1, 3.7, 3.6, 4.1,  # HP36  1VII
                   3.2, 3.4, 6.4, 2.4, 2.8, 7.2, 2.6, 4.3, 2.9, 3.5, 6.3, 3.5, 6.1, 3.5,  # RNase A   7RSA
                   3.3, 2.5, 5.1, 3.1, 4.0, 3.4, 7.3, 7.0,  # Xylanase  1BCX
                   2.5, 3.9, 3.7, 3.6, 4.6, 6.6,  # OMTKY 1OMU
                   4.3, 4.8, 3.2, 2.5, 4.1, 2.8, 6.9, 3.1, 2.6, 6.2, 3.2, 3.1, 3.9, 6.2, 6.6, 4.6, 4.3, 4.3, 4.2, 3.9,
                   3.8,  # RNase H   2RN2
                   4.2, 2.4,  # BACE1 1SGZ
                   3.5, 6.5, 1.1, 4.6, 1.8, 3.3, 2.1, 1.8, 4.8, 2.4,  # HEWL  2LZT
                   6.5, 3.7, 1.8, 4.1, 2.8, 3.7, 3.9, 3.4, 4.5, 3.9, 2.6, 4.3, 3.5, 6.8, 3.0, 4.5, 4.2  # SNASE 3BDC
                   ]
    # my predict
    predict_pka2 = [2.92, 4.12, 7.18, 2.57, 3.64, 2.42, 4.01, 5.88,  # BBL   1W4H
                    2.83, 3.41, 3.29, 3.84, 3.78, 4.01,  # NTL9  1CQU
                    4.63, 4.59, 3.93, 3.00, 7.64, 4.34, 3.98, 4.53, 2.72, 4.18, 2.75, 4.24, 5.03, 3.63, 3.99, 3.64,
                    4.66,
                    # Thioredoxin   1ERT
                    3.03, 3.86, 3.22, 4.00,  # HP36  1VII
                    3.45, 3.79, 7.16, 3.88, 2.75, 7.59, 2.73, 3.83, 2.77, 3.46, 6.59, 3.57, 5.82, 2.40,
                    # RNase A   7RSA
                    4.03, 2.99, 5.49, 3.22, 3.50, 3.41, 7.23,  # Xylanase  1BCX
                    2.88, 3.91, 3.59, 4.19, 4.50, 6.49,  # OMTKY 1OMU
                    5.05, 2.46, 3.22, 3.87, 4.72, 2.27, 6.87, 3.13, 5.47, 5.95, 2.79, 2.81, 3.83, 5.75, 6.42, 3.27,
                    4.54, 4.46, 4.12, 3.78,
                    3.61,  # RNase H   2RN2
                    6.13, 0.15,  # BACE1 1SGZ
                    3.59, 6.56, 1.48, 5.46, 1.72, 3.86, 2.40, 1.88, 4.47, 2.76,  # HEWL  2LZT
                    6.12, 4.04, 3.86, 2.68, 3.19, 3.59, 3.85, 3.62, 4.35, 3.76, 3.44, 4.63, 3.84, 6.69, 3.52, 4.15, 4.02
                    # SNASE 3BDC
                    ]

    df = pd.DataFrame({'Target pKa': target_pka2, 'Predict pKa': predict_pka2})
    MAE, MUSE, R2, R2_shift = eval_result(df)
    print('MAE: {}, MUSE: {}, R2: {}, R2_shift: {}'.format(MAE, MUSE, R2, R2_shift))


def test_find_same_residue():
    csv_test = '/home/czt/data/pka_data_new/features_csv/test_f19_r4.csv'
    csv_train = '/home/czt/data/pka_data_new/features_csv/train_n209_f19_r4.csv'
    test_df = pd.read_csv(csv_test)
    train_df = pd.read_csv(csv_train)
    test_df = test_df['file_name'].drop_duplicates().reset_index()
    train_df = train_df['file_name'].drop_duplicates().reset_index()
    test_df['file_name'] = test_df['file_name'].map(lambda x: x.split('_')[0])
    train_df['file_name'] = train_df['file_name'].map(lambda x: x.split('_')[0])
    test_df.drop_duplicates().reset_index()
    train_df.drop_duplicates().reset_index()
    merge_df = pd.merge(test_df, train_df)
    print('*' * 50)
    print(test_df)
    print('*' * 50)
    print(train_df)
    print('*' * 50)
    print(merge_df)


def test_draw_cphmd_model_scatter():
    model_predict_csv = '/home/czt/project/Predict_pKa/train_info/model_bn4_elu_n228_f19_r4_zscore_fcharge_adam/test_f19_r4_cgdtb_inSSbondCpHMD_42_29.csv'
    cphmd_predict_csv = '/home/czt/project/Predict_pKa/CpHMD_predict_info/CpHMD_predict_WT113.csv'
    model_df = pd.read_csv(model_predict_csv)
    cphmd_df = pd.read_csv(cphmd_predict_csv)
    model_df = model_df.loc[:,
               ['PDB ID', 'Res ID', 'Res Name', 'Predict pKa', 'Chain', 'model pKa', 'Predict pKa shift']]
    cphmd_df = cphmd_df.loc[:,
               ['PDB ID', 'Res ID', 'Res Name', 'Predict pKa', 'Chain', 'model pKa', 'Predict pKa shift']]
    cphmd_df.rename(columns={'Predict pKa': 'Target pKa', 'Predict pKa shift': 'Target pKa shift'}, inplace=True)
    merge_df = pd.merge(model_df, cphmd_df)
    choose_draw_res_names = ['ASP', 'HIS', 'GLU', 'LYS']
    save_dir = '/home/czt/project/Predict_pKa/train_info/model_bn4_elu_n228_f19_r4_zscore_fcharge_adam/'
    MAE, RMSE, R2, R2_shift = eval_result(merge_df)
    print('MAE: {}, RMSE: {}, R2: {}, R2_shift: {}'.format(MAE, RMSE, R2, R2_shift))
    draw_predict_target_scatter(choose_res_names=choose_draw_res_names, csv_df=merge_df, save_dir=save_dir,
                                save_name_suffix='_with_chpmd')


def data_work_on_test():
    '''
    This function integrates the model's predicted PKA values on the test set, calculated CPHMD values on the test set,
    and experimental values on the test set for analysis.
    '''
    model_csv = '../model_result/21.2/test_chimera_f20_r4_incphmd_undersample.csv'
    cphmd_csv = 'CpHMD_predict_info/test_chimera_f19_r4_incphmd_undersample.csv'
    model_df = pd.read_csv(model_csv)
    cphmd_df = pd.read_csv(cphmd_csv)

    model_df = model_df.rename(columns={'Predict pKa': 'Pre pKa', 'Predict pKa shift': 'Pre pKa shift',
                                        'Target pKa': 'Expt pKa', 'Target pKa shift': 'Expt pKa shift'})
    cphmd_df = cphmd_df.rename(columns={'Predict pKa': 'Cal pKa', 'Predict pKa shift': 'Cal pKa shift',
                                        'Target pKa': 'Expt pKa', 'Target pKa shift': 'Expt pKa shift'})
    cphmd_df = cphmd_df.loc[:, ['PDB ID', 'Chain', 'Res Name', 'Res ID', 'Cal pKa', 'Cal pKa shift']]
    merge_df = pd.merge(model_df, cphmd_df)
    merge_df['|Pre pKa shift|'] = merge_df['Pre pKa shift'].abs()
    merge_df['|Cal pKa shift|'] = merge_df['Cal pKa shift'].abs()
    merge_df['|Expt pKa shift|'] = merge_df['Expt pKa shift'].abs()
    merge_df['Pre - Expt pKa'] = merge_df['Pre pKa'] - merge_df['Expt pKa']
    merge_df['Cal - Expt pKa'] = merge_df['Cal pKa'] - merge_df['Expt pKa']
    merge_df['Pre - Cal pKa'] = merge_df['Pre pKa'] - merge_df['Cal pKa']
    merge_df['|Pre - Expt pKa|'] = merge_df['Pre - Expt pKa'].abs()
    merge_df['|Pre - Cal pKa|'] = merge_df['Pre - Cal pKa'].abs()
    merge_df['|Cal - Expt pKa|'] = merge_df['Cal - Expt pKa'].abs()
    merge_df = merge_df.loc[:, ['PDB ID', 'Chain', 'Res Name', 'Res ID', 'Pre pKa', 'Cal pKa', 'Expt pKa', 'model pKa',
                                'Pre pKa shift', 'Cal pKa shift', 'Expt pKa shift', '|Pre pKa shift|',
                                '|Cal pKa shift|',
                                '|Expt pKa shift|', 'Pre - Expt pKa', 'Cal - Expt pKa', 'Pre - Cal pKa',
                                '|Pre - Expt pKa|', '|Cal - Expt pKa|', '|Pre - Cal pKa|']]
    suffix = '_atom_charge_radii10'
    center = '_undersample'
    # center = ''
    merge_df.to_csv('test{}_result_all{}.csv'.format(center, suffix), index=False)
    merge_df1 = merge_df[merge_df['Res Name'] == 'ASP']
    merge_df2 = merge_df[merge_df['Res Name'] == 'GLU']
    merge_df3 = merge_df[merge_df['Res Name'] == 'HIS']
    merge_df4 = merge_df[merge_df['Res Name'] == 'LYS']
    merge_df1.to_csv('test{}_result_asp{}.csv'.format(center, suffix), index=False)
    merge_df2.to_csv('test{}_result_glu{}.csv'.format(center, suffix), index=False)
    merge_df3.to_csv('test{}_result_his{}.csv'.format(center, suffix), index=False)
    merge_df4.to_csv('test{}_result_lys{}.csv'.format(center, suffix), index=False)


def data_work_on_val():
    '''
    This function integrates the model's predicted PKA values on the test set and
    calculated CPHMD values on the test set for analysis.
    '''
    model_csv = '../model_result/21.2/val_n27_f20_n4.csv'
    model_df = pd.read_csv(model_csv)

    model_df = model_df.rename(columns={'Predict pKa': 'Pre pKa', 'Predict pKa shift': 'Pre pKa shift',
                                        'Target pKa': 'Cal pKa', 'Target pKa shift': 'Cal pKa shift'})

    model_df['|Pre pKa shift|'] = model_df['Pre pKa shift'].abs()
    model_df['|Cal pKa shift|'] = model_df['Cal pKa shift'].abs()
    model_df['Pre - Cal pKa'] = model_df['Pre pKa'] - model_df['Cal pKa']
    model_df['|Pre - Cal pKa|'] = model_df['Pre - Cal pKa'].abs()
    model_df = model_df.loc[:, ['PDB ID', 'Chain', 'Res Name', 'Res ID', 'Pre pKa', 'Cal pKa', 'model pKa',
                                'Pre pKa shift', 'Cal pKa shift', '|Pre pKa shift|', '|Cal pKa shift|',
                                'Pre - Cal pKa', '|Pre - Cal pKa|']]
    suffix = '_atom_charge_radii10'
    # center = '_undersample'
    center = ''
    model_df.to_csv('val{}_result_all{}.csv'.format(center, suffix), index=False)
    merge_df1 = model_df[model_df['Res Name'] == 'ASP']
    merge_df2 = model_df[model_df['Res Name'] == 'GLU']
    merge_df3 = model_df[model_df['Res Name'] == 'HIS']
    merge_df4 = model_df[model_df['Res Name'] == 'LYS']
    merge_df1.to_csv('val{}_result_asp{}.csv'.format(center, suffix), index=False)
    merge_df2.to_csv('val{}_result_glu{}.csv'.format(center, suffix), index=False)
    merge_df3.to_csv('val{}_result_his{}.csv'.format(center, suffix), index=False)
    merge_df4.to_csv('val{}_result_lys{}.csv'.format(center, suffix), index=False)


def test_astype():
    a = np.array([-0.5, -0.1, 0.4, 0.49, 0.5, 1.6])
    print(a.astype(int))
    print(a.round().astype(int))


if __name__ == '__main__':
    data_work_on_val()
