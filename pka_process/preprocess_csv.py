import numpy as np
import pandas as pd
import h5py
import pybel
from preprocess_bio import Featurizer
import os
import argparse
import re

model_pka = {
    'ASP': 3.7,
    'GLU': 4.2,
    'HIS': 6.34,
    'LYS': 10.4,
    'CYS': 8.5,
}


def input_file(path):
    """Check if input file exists."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path


def output_file(path):
    """Check if output file can be created."""
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path


def string_bool(s):
    s = s.lower()
    if s in ['true', 't', '1', 'yes', 'y']:
        return True
    elif s in ['false', 'f', '0', 'no', 'n']:
        return False
    else:
        raise IOError('%s cannot be interpreted as a boolean' % s)


def get_protein(protein, protein_format):
    """
    This function will read a pdb(or mol2) file or a directory contain pdb(or mol2) files,
    and caculate their 20 features, than return 3D coordinates , 20 features, and file names. The 20 features presents:
    ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge',
     'is_center_residue', 'res_type', 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
    :param protein: String, a pdb(or mol2) file or a directory contain pdb(or mol2) files.
    :param protein_format: String, 'pdb' presents pdb format,  'mol2' presents mol2 format.
    :return: Tuple(List, List, List), yield (protein_coords, protein_features, file_name)
    """
    featurizer = Featurizer()
    if os.path.isdir(protein):
        file_name_list = os.listdir(protein)
        for file_name in file_name_list:
            file_name = file_name.split('.')[0]
            file_path = os.path.join(protein, file_name + '.{}'.format(protein_format))
            protein_data = next(pybel.readfile(protein_format, file_path))
            protein_coords, protein_features = featurizer.get_features(protein_data, file_name)
            yield protein_coords, protein_features, file_name
    else:
        file_name = os.path.splitext(os.path.split(protein)[1])[0]
        protein_data = next(pybel.readfile(protein_format, protein))
        protein_coords, protein_features = featurizer.get_features(protein_data, file_name)
        yield protein_coords, protein_features, file_name


def clean_csv(input_csv, cleaned_csv):
    """
    this function will clean input_csv file, then save as cleaned_csv file.
    :param input_csv: String, the path of csv should be clean.
    :param cleaned_csv: String, the path of cleaned csv file.
    :return: None.
    """
    useful_res_name = ['ASP', 'GLU', 'HIS', 'LYS', 'CYS']
    csv_df = pd.read_csv(input_csv)
    # rename columns 'Expt. pKa' to 'pKa'
    csv_df = csv_df.rename(columns={'Expt. pKa': 'pKa'})
    # change lowercase to uppercase
    csv_df['Res Name'] = csv_df['Res Name'].apply(lambda x: x.upper())
    # change 'pKa' from type string type to float type
    csv_df['pKa choose'] = csv_df['pKa'].apply(lambda x: re.match('^\d+(\.\d+)?$', x) is not None)
    csv_df = csv_df.loc[csv_df['pKa choose']]
    csv_df['pKa'] = csv_df['pKa'].apply(lambda x: float(x))
    # choose pKa in Expt. pH range
    csv_df['min pH'] = csv_df['Expt. pH'].apply(lambda x: None if x is np.nan else float(x.split('-')[0]))
    csv_df['max pH'] = csv_df['Expt. pH'].apply(lambda x: None if x is np.nan else float(x.split('-')[1]))
    csv_df = csv_df.loc[(csv_df['min pH'] <= csv_df['pKa']) & (csv_df['pKa'] <= csv_df['max pH'])
                        | csv_df['min pH'].isna()]
    # choose Expt. Uncertainty is not N/A
    csv_df = csv_df.loc[~ csv_df['Expt. Uncertainty'].isna()]
    # choose necessary residue name
    csv_df['pKa choose'] = csv_df['Res Name'].apply(lambda x: x in useful_res_name)
    csv_df = csv_df.loc[csv_df['pKa choose']]
    # change 'Res ID' from type float to int type
    csv_df['Res ID'] = csv_df['Res ID'].apply(lambda x: int(x))
    # choose necessary columns, drop out duplicates lines, then save a new csv file
    csv_df = csv_df.loc[:, ['PDB ID', 'Res Name', 'Chain', 'Res ID', 'pKa']].drop_duplicates()
    csv_df.to_csv(cleaned_csv, index=False)


def save_protein_features_as_csv(protein, protein_format, output_dir, pka_csv=None):
    """
    This function will read a pdb(or mol2) file or a directory contain pdb(or mol2) files,
    and caculate their 20 features, then format the protein information as hdf:
    {
        name1 : dataset1{
             [[ x, y, z, 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', 'hyb',
                'heavyvalence', 'heterovalence', 'partialcharge','is_center_residue', 'res_type', 'hydrophobic',
                'aromatic', 'acceptor', 'donor', 'ring'], ...] -> features [n * 23]
            'pka' : float
        },
        ...
        namek : datasetk{
            ...
        }
    }
    name: string, protein_name, "XXXX", 'XXXX' must be format like
            "(pdb id)_(chain)_(residue id)_(residue name)_(new residue id)",
            this function will use the name to find pka value in file 'pka_csv'.
    features: 2D float array, shape[n * w], n present the protein contain how many heavy atoms.
                w present 23 features -> [ x, y, z, 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', 'hyb',
                'heavyvalence', 'heterovalence', 'partialcharge','is_center_residue', 'res_type', 'hydrophobic',
                'aromatic', 'acceptor', 'donor', 'ring'],
    pka: float, pka value.
    k : present how many proteins.
    :param protein: String, a pdb(or mol2) file or a directory contain pdb(or mol2) files.
    :param protein_format: String, 'pdb' presents pdb format,  'mol2' presents mol2 format.
    :param output_dir: String, the directory save hdf file.
    :param choosed_residue: List, the list must be sub set of [ 'ASP', 'GLU', 'LYS', 'HIS', 'CYS'].
    :param mini_shift: Float, the mini number of distance between model pKa an pKa, if residue's distance more than mini_shift,
                         the residues data will be choosed.
    :param pka_csv: String, CSV table with pka values. It must contain two columns: `name` which must be equal to protein's file
            name without extenstion, and `pka` which must contain floats'
    :param choose_rate: Float, how much rate of total data will be saved in hdf.
    :return: None.
    """
    print('start')
    global model_pka
    if protein[-1] == '/':
        protein = protein[0:-1]
    save_name = protein.split('/')[-1].split('.')[0]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_path = os.path.join(output_dir, save_name + '.csv')
    print(save_path)

    if pka_csv is not None:
        pka_dataframe = pd.read_csv(pka_csv)
        pka_dataframe = pka_dataframe.rename(columns={'Expt. pKa': 'pKa'})
        pka_dataframe = pka_dataframe.set_index(['PDB ID', 'Chain', 'Res ID', 'Res Name'])[['pKa']]
    else:
        pka_dataframe = None

    features_df = pd.DataFrame(columns=('idx', 'pka', 'file_name', 'x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se',
                                        'halogen', 'metal', 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge',
                                        'is_center_residue', 'res_type', 'hydrophobic', 'aromatic', 'acceptor', 'donor',
                                        'ring'))
    features_list = []
    protein_generator = get_protein(protein, protein_format)
    idx = 0
    while True:
        try:
            protein_coords, protein_features, file_name = next(protein_generator)
            name_split = file_name.split('_')
            # if ires name not be choosed, drump over.
            print('read file {}.'.format(file_name))
            pka_value = None
            if pka_dataframe is not None:
                try:
                    pka_value = pka_dataframe.loc[name_split[0]].loc[name_split[1]].loc[float(name_split[2])].loc[
                        name_split[3]].loc['pKa']
                except:
                    print('{} is not choosed.'.format(name_split))
                    continue
            length = len(protein_coords)
            for i in range(length):
                features_list.append({'idx': idx, 'pka': pka_value, 'file_name': file_name, 'x': protein_coords[i][0],
                                     'y': protein_coords[i][1], 'z': protein_coords[i][2], 'B': protein_features[i][0],
                                     'C': protein_features[i][1], 'N': protein_features[i][2], 'O': protein_features[i][3],
                                     'P': protein_features[i][4], 'S': protein_features[i][5], 'Se': protein_features[i][6],
                                     'halogen': protein_features[i][7], 'metal': protein_features[i][8], 'hyb': protein_features[i][9],
                                    'heavyvalence': protein_features[i][10], 'heterovalence': protein_features[i][11],
                                     'partialcharge': protein_features[i][12], 'is_center_residue': protein_features[i][13],
                                    'res_type': protein_features[i][14], 'hydrophobic': protein_features[i][15],
                                    'aromatic': protein_features[i][16], 'acceptor': protein_features[i][17],
                                     'donor': protein_features[i][18], 'ring': protein_features[i][19]})
            idx += 1
        except StopIteration:
            break
    features_df = features_df.append(features_list)
    features_df.to_csv(save_path, index=False)


def run():
    parser = argparse.ArgumentParser(
        description='Prepare molecular data for the network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=''' '''
    )

    parser.add_argument('--protein', '-p', required=True, type=str,
                        help='files or folder with protein\' structures')
    parser.add_argument('--protein_format', type=str, default='pdb',
                        help='file format for the protein,'
                             ' must be supported by openbabel')
    parser.add_argument('--output_dir', '-o', default='./hdf',
                        type=output_file,
                        help='name for the file with the prepared structures')
    parser.add_argument('--pka_csv', '-a', default=None, type=input_file,
                        help='CSV table with pka values.'
                             ' It must contain two columns: `name` which must be'
                             ' equal to protein\'s file name without extenstion,'
                             ' and `pka` which must contain floats')

    args = parser.parse_args()
    save_protein_features_as_csv(args.protein, args.protein_format, args.output_dir, args.pka_csv)


if __name__ == '__main__':
    run()

