from torch.utils.data import Dataset, TensorDataset
from pka_utils import get_cube_rotations, make_grid, grid_features_mean, grid_features_std, box_features_mean, box_features_std, atom_features_mean, atom_features_std, features_name, crop_charge_grid, random_rotation, draw_grid, calculate_input_data_feature_mean_and_std
import numpy as np
import h5py
import pandas as pd
import random
import time
from spme_ch_grid import get_charge_grid
from fasta import load_fasta

class PkaDatasetHDF(Dataset):
    """
    (not use anymore!!!)
    Initialition data, preprocessing data and data augmuntation.
    """
    def __init__(self, data_path='hdf', is_rotate=False):
        # hdf_name_list = glob.glob(os.path.join(data_path, '*.hdf'))
        self.data_path = data_path
        self.names = []
        self.load_data_num = 0
        self.is_rotate = is_rotate
        with h5py.File(data_path, 'r') as f:
            for name in f:
                self.names.append(name)
        self.total_len = len(self.names)
        self.batch_load_data()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def batch_load_data(self, max_num=512):
        """
        load a batch of data from disk, every time will load new batch of data.
        This is done to reduce memory consumption.
        :param max_num: The max number of one batch data.
        :return: None
        """
        features = []
        coords = []
        pkas = []
        # Check how many data unload and get next batch of data names.
        if self.load_data_num + max_num <= self.total_len:
            self.batch_names = self.names[self.load_data_num:self.load_data_num + max_num]
            self.load_data_num += max_num
        else:
            self.batch_names = self.names[self.load_data_num:]
            self.load_data_num = self.total_len

        with h5py.File(self.data_path, 'r') as f:
            for idx, name in enumerate(self.batch_names):
                dataset = f[name]
                coords.append(dataset[:, :3])
                features.append(dataset[:, 3:])
                pkas.append([dataset.attrs['pka'], idx])

        # self normalization
        # for feature in features:
        #     feature[:, 9:] -= np.array(features_mean)[9:]
        #     feature[:, 9:] /= np.array(features_std)[9:]

        grids = []
        if self.is_rotate:
            rotations = get_cube_rotations()
        for idx in range(len(self.batch_names)):
            if self.is_rotate:
                # Rotate 24 angles to expand data set
                random_idx = random.randint(0, 23)
                matrix = rotations[random_idx]
                coords[idx] = np.dot(coords[idx], matrix)
            grid = make_grid(coords[idx], features[idx])
            grid = grid.transpose(0, 4, 1, 2, 3)  # shape[n, x, y, z, f] ->shape[n, f, x, y, z]
            grids.append(grid)
        self.len = len(grids)
        self.x_data = np.vstack(grids)
        self.y_data = np.vstack(pkas)

    def is_empty(self):
        """
        Check if there is any remaining batch data.
        :return: Bool. True means Yes, False mean No.
        """
        return self.load_data_num >= self.total_len

    def get_total_len(self):
        return self.total_len


class PkaDatasetCSV(Dataset):
    """
    Initialition data, preprocessing data and data augmuntation.
    """
    def __init__(self, data_path='xxxx.csv', is_rotate=False, rotate_angle=90, fill_charge='atom charge', normalize=False,
                 center_coors_path=None, proteins_features_path=None, radii=10, is_train=True, res_only=True):
        self.data_path = data_path
        self.names = []
        self.load_data_num = 0
        self.is_rotate = is_rotate
        self.rotate_angle = rotate_angle
        self.df = pd.read_csv(data_path)
        self.features_cols = self.df.columns[6:]
        self.charge_col_idx = self.features_cols.get_loc('partialcharge')
        self.names = self.df['file_name'].drop_duplicates().values.tolist()
        self.idxes = self.df['idx'].drop_duplicates().values.tolist()
        self.total_len = len(self.names)
        self.center_coors_path = center_coors_path
        self.proteins_features_path = proteins_features_path
        self.fill_charge = fill_charge
        self.normalize = normalize
        self.radii = radii
        
        self.res_only = res_only
        if not res_only:
            self.fasta_map = load_fasta(is_train=is_train)

        # load proteins' charge grids
        if self.fill_charge == 'grid charge':
            if self.center_coors_path is None:
                raise ValueError('If set fill_charge true, center_coors_path must be set.')
            if self.proteins_features_path is None:
                raise ValueError('If set fill_charge true, proteins_features_path must be set.')
            self.center_coors_pd = pd.read_csv(self.center_coors_path)
            self.proteins_grid = {}
            self.proteins_offset = {}
            proteins_features_pd = pd.read_csv(proteins_features_path)
            protein_idxes = proteins_features_pd['idx'].drop_duplicates().tolist()
            for idx in protein_idxes:
                one_protein_features = proteins_features_pd[proteins_features_pd['idx'] == idx]
                pdb_id_chain = one_protein_features['file_name'].iloc[0]
                protein_grid, protein_offset = get_charge_grid(4, data=one_protein_features)
                self.proteins_grid[pdb_id_chain] = protein_grid
                self.proteins_offset[pdb_id_chain] = protein_offset
        elif self.fill_charge == 'box charge':
            self.ires_grid_dict = {}
            self.ires_offset_dict = {}
            ires_idxes = self.df['idx'].drop_duplicates().tolist()
            for idx in ires_idxes:
                one_ires_features = self.df[self.df['idx'] == idx]
                file_name = one_ires_features['file_name'].iloc[0]
                ires_grid, ires_offset = get_charge_grid(4, data=one_ires_features)
                self.ires_grid_dict[file_name] = ires_grid
                self.ires_offset_dict[file_name] = ires_offset

        # choose std and mean number for z-score normalization according to input data features
        if self.normalize:
            if self.fill_charge == 'grid charge':
                self.features_mean = grid_features_mean
                self.features_std = grid_features_std
            elif self.fill_charge == 'box charge':
                self.features_mean = box_features_mean
                self.features_std = box_features_std
            else:
                self.features_mean = atom_features_mean
                self.features_std = atom_features_std
            mean_series = pd.Series(self.features_mean, index=features_name)
            std_series = pd.Series(self.features_std, index=features_name)
            # choose used features columns
            mean_series = mean_series.loc[self.features_cols]
            std_series = std_series[self.features_cols]
            self.features_mean = mean_series.to_numpy().astype(np.float32)
            self.features_std = std_series.to_numpy().astype(np.float32)

        self.batch_load_data()

    def __getitem__(self, index):
        if self.res_only:
            return self.x_data[index], self.y_data[index]
        else:
            return self.x_data[index], self.y_data[index], self.f_data[index]

    def __len__(self):
        return self.len

    def flash_batch_data(self):
        self.load_data_num = 0
        self.batch_load_data()

    def batch_load_data(self, max_num=20000):
        """
        load a batch of data from disk, every time will load new batch of data.
        This is done to reduce memory consumption.
        :param max_num: The max number of one batch data.
        :return: None
        """
        features = []
        coords = []
        pkas = []
        fastas = []
        # Check how many data unload and get next batch of data names.
        if self.load_data_num + max_num <= self.total_len:
            self.batch_names = self.names[self.load_data_num:self.load_data_num + max_num]
            self.batch_idxes = self.idxes[self.load_data_num:self.load_data_num + max_num]
            self.load_data_num += max_num
        else:
            self.batch_names = self.names[self.load_data_num:]
            self.batch_idxes = self.idxes[self.load_data_num:]
            self.load_data_num = self.total_len

        for idx, batch_idx in enumerate(self.batch_idxes):
            temp_df = self.df.loc[self.df['idx'] == batch_idx]
            pka = temp_df['pka'].iloc[0]
            if not self.res_only:
                filename = temp_df['file_name'].iloc[0]
                fasta_name = '_'.join(filename.split('_')[:2])
                fasta = self.fasta_map[fasta_name]
                fastas.append(fasta)
            dataset = temp_df.drop(columns=['idx', 'file_name', 'pka'], axis=1).values
            coords.append(dataset[:, :3])
            features.append(dataset[:, 3:])
            pkas.append([pka, idx])

        grids = []
        time1 = time.time()
        for idx in range(len(self.batch_names)):
            grid = make_grid(coords[idx], features[idx], max_dist=self.radii)
            grid = grid.transpose(0, 4, 1, 2, 3)  # shape[n, x, y, z, f] ->shape[n, f, x, y, z]
            # do fill charge
            if self.fill_charge == 'grid charge':
                name = self.batch_names[idx]
                pdb_id_chain = '{}_{}'.format(name.split('_')[0], name.split('_')[1])
                ires_coor = self.center_coors_pd[self.center_coors_pd['file_name'] == name].loc[:, ['x', 'y', 'z']].to_numpy()[0]
                protein_offset = self.proteins_offset[pdb_id_chain]
                charge_grid = crop_charge_grid(self.proteins_grid[pdb_id_chain], protein_offset, ires_coor, radii=self.radii)
                grid[0, self.charge_col_idx, :, :, :] = charge_grid
                # grid = grid.transpose(0, 2, 3, 4, 1)
                # draw_grid(grid, 'xx', 0)
            elif self.fill_charge == 'box charge':
                name = self.batch_names[idx]
                ires_coor = np.zeros((3,))
                box_charge = crop_charge_grid(self.ires_grid_dict[name], self.ires_offset_dict[name], ires_coor)
                grid[0, self.charge_col_idx, :, :, :] = box_charge
            # do rotation
            if self.is_rotate:
                grid = random_rotation(grid, self.rotate_angle)
            grids.append(grid)

        self.len = len(grids)
        self.x_data = np.vstack(grids)
        self.y_data = np.vstack(pkas)
        self.f_data = np.array(fastas)

        # do z-score normalize
        if self.normalize:
            self.x_data = self.x_data - np.expand_dims(self.features_mean, axis=(0, 2, 3, 4))
            self.x_data = self.x_data / np.expand_dims(self.features_std, axis=(0, 2, 3, 4))
            self.x_data[np.isinf(self.x_data)] = 0
            self.x_data[np.isnan(self.x_data)] = 0
        time2 = time.time()
        print('use time: {}'.format(time2 - time1))

    def is_empty(self):
        """
        Check if there is any remaining batch data.
        :return: Bool. True means Yes, False mean No.
        """
        return self.load_data_num >= self.total_len

    def get_total_len(self):
        return self.total_len


if __name__ == '__main__':
    data_path = '../data/model_input/final_train_data/train_n252_f20_n4.csv'
    is_rotate = True
    fill_charge = 'grid charge'
    normalize = False
    radii = 10
    center_coors_path = '../data/model_input/final_train_data/CpHMD_pka252_center_coors.csv'
    proteins_features_path = '../data/model_input/final_train_data/data_pdb_CpHMD252_fixed_mol2.csv'
    dataset = PkaDatasetCSV(data_path, is_rotate=is_rotate, fill_charge=fill_charge,
                            center_coors_path=center_coors_path, proteins_features_path=proteins_features_path,
                            radii=radii)
    print('test: finished')
    features_mean, features_std = calculate_input_data_feature_mean_and_std(dataset.x_data)
    print('charge type:', fill_charge)
    print('data num:', len(dataset.x_data))
    print('features_mean:', features_mean)
    print('features_std:', features_std)


