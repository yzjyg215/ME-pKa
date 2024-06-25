import numpy as np
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
import random

grid_features_mean = [0.0000000e+00, 1.3736343e-02, 3.6577375e-03, 4.0088212e-03,
                      0.0000000e+00, 9.3589711e-05, 0.0000000e+00, 0.0000000e+00,
                      0.0000000e+00, 5.1632512e-02, 4.2728718e-02, 1.1211292e-02,
                      - 2.4629454e-03, 9.5343147e-04, 6.3717526e-01, 6.5841060e-03,
                      1.4606385e-03, 7.4423631e-03, 3.8711089e-03, 1.9136829e-03]

grid_features_std = [0., 0.11663348, 0.06041992, 0.06310891,
                     0., 0.00967377,0., 0.,
                     0., 0.35787693, 0.31008136, 0.13484533,
                     0.01620124, 0.03089078, 0.818935, 0.08094488,
                     0.03821403, 0.08607551, 0.06215977, 0.04373885]

box_features_mean = [0.0000000e+00, 1.3736343e-02, 3.6577375e-03, 4.0088212e-03,
                     0.0000000e+00, 9.3589711e-05, 0.0000000e+00, 0.0000000e+00,
                     0.0000000e+00, 5.1632512e-02, 4.2728718e-02, 1.1211292e-02,
                     -2.3283656e-03, 9.5343147e-04, 6.3717526e-01, 6.5841060e-03,
                     1.4606385e-03, 7.4423631e-03, 3.8711089e-03, 1.9136829e-03]

box_features_std = [0., 0.11663348, 0.06041992, 0.06310891,
                    0., 0.00967377, 0., 0.,
                    0., 0.35787693, 0.31008136, 0.13484533,
                    0.01622527, 0.03089078, 0.818935, 0.08094488,
                    0.03821403, 0.08607551, 0.06215977, 0.04373885]

atom_features_mean = [0.0000000e+00, 1.3736343e-02, 3.6577375e-03, 4.0088212e-03,
                      0.0000000e+00, 9.3589711e-05, 0.0000000e+00, 0.0000000e+00,
                      0.0000000e+00, 5.1632512e-02, 4.2728718e-02, 1.1211292e-02,
                      -2.5641348e-03, 9.5343147e-04, 6.3717526e-01, 6.5841060e-03,
                      1.4606385e-03, 7.4423631e-03, 3.8711089e-03, 1.9136829e-03]

atom_features_std = [0., 0.11663348, 0.06041992, 0.06310891,
                     0., 0.00967377, 0., 0.,
                     0., 0.35787693, 0.31008136, 0.13484533,
                     0.0644822, 0.03089078, 0.818935, 0.08094488,
                     0.03821403, 0.08607551, 0.06215977, 0.04373885]


features_name = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'is_center_residue',
                 'res_type', 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']


def crop_charge_grid(protein_grid, protein_offset, ires_center_coor, radii=10.0):
    """
    This function will corp protein charge grid according to titrable residue center coordinate and box radii.
    :param protein_grid: Numpy.NDArray, 3D Array contain single chain protein's interpolated partial charge.
    :param protein_offset: Numpy.NDArray, the offset of protein_grid array. shape like: [x, y, z]
    :param ires_center_coor: Numpy.NDArray, titrable residue center coordinate. shape like: [x, y, z]
    :param radii: float, the radii of crop box. The length of 3D box is (radii * 2 + 1)
    :return croped_charge_grid:Numpy.NDArray, croped charge grid. The shape is [radii * 2 + 1, radii * 2 + 1, radii * 2 + 1]
    """
    ires_center_coor = np.around(ires_center_coor)
    p1 = ires_center_coor - protein_offset - radii
    p2 = ires_center_coor - protein_offset + radii + 1
    p1 = p1.round().astype(int)
    p2 = p2.round().astype(int)
    protein_grid_shape = np.asarray(protein_grid.shape)
    overlap_p1 = np.vstack((p1, np.array([0, 0, 0]))).max(0).astype(int)
    overlap_p2 = np.vstack((p2, protein_grid_shape)).min(0).astype(int)
    crop_offset_p1 = overlap_p1 - p1
    crop_offset_p2 = overlap_p2 - p1
    box_len = int(radii * 2 + 1)
    croped_charge_grid = np.zeros((box_len, box_len, box_len))

    overlap_grid = protein_grid[overlap_p1[0]:overlap_p2[0], overlap_p1[1]:overlap_p2[1], overlap_p1[2]:overlap_p2[2]]
    croped_charge_grid[crop_offset_p1[0]:crop_offset_p2[0], crop_offset_p1[1]:crop_offset_p2[1],
    crop_offset_p1[2]:crop_offset_p2[2]] = overlap_grid
    return croped_charge_grid


def make_grid(coords, features, grid_resolution=1.0, max_dist=10.0):
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.

    Parameters
    ----------
    coords, features: array-likes, shape (N, 3) and (N, F)
        Arrays with coordinates and features for each atoms.
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.

    Returns
    -------
    coords: np.ndarray, shape = (M, M, M, F)
        4D array with atom properties distributed in 3D space. M is equal to
        2 * `max_dist` / `grid_resolution` + 1
    """
    try:
        coords = np.asarray(coords, dtype=np.float32)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    c_shape = coords.shape
    if len(c_shape) != 2 or c_shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')

    N = len(coords)
    try:
        features = np.asarray(features, dtype=np.float32)
    except ValueError:
        raise ValueError('features must be an array of floats of shape (N, 3)')
    f_shape = features.shape
    if len(f_shape) != 2 or f_shape[0] != N:
        raise ValueError('features must be an array of floats of shape (%s, 3)'
                         % N)

    if not isinstance(grid_resolution, (float, int)):
        raise TypeError('grid_resolution must be float')
    if grid_resolution <= 0:
        raise ValueError('grid_resolution must be positive')

    if not isinstance(max_dist, (float, int)):
        raise TypeError('max_dist must be float')
    if max_dist <= 0:
        raise ValueError('max_dist must be positive')

    num_features = f_shape[1]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)

    box_size = ceil(2 * max_dist / grid_resolution + 1)

    # move all atoms to the neares grid point
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    # print("in_box", in_box.shape, in_box)
    grid = np.zeros((1, box_size, box_size, box_size, num_features),
                    dtype=np.float32)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f
    # when the number of features is 20, means it's 15th feature represent residue type.
    if num_features == 20:
        ires_type_value = features[0][14]
        # Color the gird according to the residue type, try to tell the net witch the residue is.
        grid[0, :, :, :, 14] = ires_type_value
    return grid


def draw_grid(grid, res_name, pka):
    """
    This function will read grid then draw it with 3D scatter diagram.
    :param grid: Numpy.NDArray, it's shape is (1, 21, 21, 21, 18), it is 3D box
                    contains titriable residue structural information.
    :return: None.
    """
    # features_dict = {
    #     'names': ["B", "C", "N", "O", "P", "S", "Se", "halogen", "metal", "hyb", "heayvalence", "heterovalence",
    #               "partialcharge", "hydrophobic", "aromatic", "acceptor", "donor", "ring"],
    #     'color': ['purple', 'green', 'blue', 'pink', 'brown', 'red', 'teal', 'orange',
    #               'yellow', 'grey', 'lime green', 'tan', 'yellow', 'black', 'b', 'dark green',
    #               'turquoise', 'sky blue']
    # }
    features_dict = {
        'names': ["B", "C", "N", "O", "P", "S", "Se", "halogen", "metal", "hyb", "heayvalence", "heterovalence",
                  "partialcharge", "is_center_res", "residue_type", "hydrophobic", "aromatic", "acceptor",
                  "donor", "ring"],
        'color': ['purple', 'green', 'blue', 'pink', 'brown', 'red', 'teal', 'orange',
                  'yellow', 'grey', 'lime green', 'tan', 'yellow', 'black', 'b', 'cyan', 'dark green',
                  'turquoise', 'sky blue', 'light green']
    }

    res_name = '{}_{}'.format(res_name, pka)
    grid = grid[0]
    fig = plt.figure(res_name)
    ax3d = Axes3D(fig)
    sca_list = []
    label_list = []
    for i in range(0, 9):
        one_chanel_grid = grid[:, :, :, i]
        x, y, z = one_chanel_grid.nonzero()
        color = features_dict['color'][i]
        sca = ax3d.scatter(x, y, z, c=color, marker="o", s=10)
        sca_list.append(sca)
        label_list.append(features_dict['names'][i])
    for i in range(13, 14):
        one_chanel_grid = grid[:, :, :, i]
        x, y, z = one_chanel_grid.nonzero()
        color = features_dict['color'][i]
        sca = ax3d.scatter(x, y, z, c=color, marker="o", s=20)
        sca_list.append(sca)
        label_list.append(features_dict['names'][i])
    for i in range(12, 13):
        one_chanel_grid = grid[:, :, :, i]
        x, y, z = one_chanel_grid.nonzero()
        color = features_dict['color'][i]
        sca = ax3d.scatter(x, y, z, c=color, marker="o", s=0)
        sca_list.append(sca)
        label_list.append(features_dict['names'][i])
    print(res_name)
    save_path = os.path.join('/media/czt/My Passport/czt/img/3D/', '{}_atoms.jpg'.format(res_name))
    ax3d.set_xlim3d(0, 20)
    ax3d.set_ylim3d(0, 20)
    ax3d.set_zlim3d(0, 20)
    plt.legend(tuple(sca_list), tuple(label_list), loc='best')
    # plt.savefig(save_path)
    plt.show()


def rotation_matrix(axis, theta):
    """Counterclockwise rotation about a given axis by theta radians"""

    try:
        axis = np.asarray(axis, dtype=np.float32)
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


def get_cube_rotations():
    """
    This function will get cube rotation matrixes.
    :return: List[NDArray, ...] , a list of rotation matrix, the shape of each matrix is (3 , 3).
    """
    # Create matrices for all possible 90* rotations of a box
    rotations = [rotation_matrix([1, 1, 1], 0)]

    # about X, Y and Z - 9 rotations
    for a1 in range(3):
        for t in range(1, 4):
            axis = np.zeros(3)
            axis[a1] = 1
            theta = t * pi / 2.0
            rotations.append(rotation_matrix(axis, theta))

    # about each face diagonal - 6 rotations
    for (a1, a2) in combinations(range(3), 2):
        axis = np.zeros(3)
        axis[[a1, a2]] = 1.0
        theta = pi
        rotations.append(rotation_matrix(axis, theta))
        axis[a2] = -1.0
        rotations.append(rotation_matrix(axis, theta))

    # about each space diagonal - 8 rotations
    for t in [1, 2]:
        theta = t * 2 * pi / 3
        axis = np.ones(3)
        rotations.append(rotation_matrix(axis, theta))
        for a1 in range(3):
            axis = np.ones(3)
            axis[a1] = -1
            rotations.append(rotation_matrix(axis, theta))

    return rotations


def show_grid(data_path):
    coords = []
    features = []
    pkas = []
    names = []
    with h5py.File(data_path, 'r') as f:
        for name in f:
            dataset = f[name]
            names.append(name)
            coords.append(dataset[:, :3])
            features.append(dataset[:, 3:])
            pkas.append([dataset.attrs['pka']])

    for idx in range(len(coords)):
        grid = make_grid(coords[idx], features[idx])
        draw_grid(grid, names[idx], pkas[idx])


def calculate_features_average(data_path):
    coords = []
    features = []
    pkas = []
    names = []
    with h5py.File(data_path, 'r') as f:
        for name in f:
            dataset = f[name]
            names.append(name)
            coords.append(dataset[:, :3])
            features.append(dataset[:, 3:])
            pkas.append([dataset.attrs['pka']])

    stack_features = np.vstack(tuple(features))
    total_num = stack_features.shape[0]
    # print(total_num)
    # features = np.array(features)
    result_mean = stack_features.mean(0)
    result_max = stack_features.max(0)
    result_min = stack_features.min(0)
    result_scale = result_max - result_min
    result_sd = np.sqrt(((stack_features - result_mean) ** 2).mean(0))
    # result_sqrt = ((stack_features - result_mean)**2).sum(0).sqrt()
    # print(result_max)
    # print(result_min)
    print(result_mean)
    print(result_sd)
    # print(result_scale)
    # for feature in features:
    #     feature[:, 9:] /= np.array(features_scale)[9:]
    # print(feature[0])


def random_rotation(gird, rotate_angle):
    """
    This function will rotate cube with random direction, total is 24 different direction with rotate 90 degrees or 4
     different direction with rotate 180 degrees.
    :param gird: Numpy.NDArray, the gird should be rotated. Shape is (1, f, x, y, z)
    :return rotated_gird: Numpy.NDArray, rotated gird, shape is (1, f, x, y, z), , only rotate x, y, z dimensions.
    """
    if rotate_angle == 90:
        # put the six faces up
        first_rotate_axies = [(2, 3), (2, 3), (2, 3), (3, 2), (3, 4), (4, 3)]
        first_rotate_times = [0, 2, 1, 1, 1, 1]
        # Rotate in four directions along the vertical axis
        second_rotate_axies = [(2, 4), (2, 4), (2, 4), (2, 4)]
        second_rotate_times = [0, 1, 2, 3]
        # use random number to rotate random
        first_random = random.randint(0, 5)
        second_random = random.randint(0, 3)
        rotated_grid = np.rot90(gird, k=first_rotate_times[first_random], axes=first_rotate_axies[first_random])
        rotated_grid = np.rot90(rotated_grid, k=second_rotate_times[second_random], axes=second_rotate_axies[second_random])
    elif rotate_angle == 180:
        # put the six faces up
        first_rotate_axies = [(2, 3), (2, 3), (2, 4), (3, 4)]
        first_rotate_times = [0, 2, 2, 2]
        # use random number to rotate random
        first_random = random.randint(0, 3)
        rotated_grid = np.rot90(gird, k=first_rotate_times[first_random], axes=first_rotate_axies[first_random])
    else:
        raise ValueError('rotate angle only support 90 degrees and 180 degrees')
    return rotated_grid


def calculate_input_data_feature_mean_and_std(grids):
    """
    this function will calculate dataset's feature mean value and feature standard deviation.
    :param grids: Numpy.NDArray, input dataset, shape is (n, f, x, y, z)
    :return features_mean, features_std: List, List, mean value of features, standard deviation of features.
    """
    features_mean = grids.mean(axis=(0, 2, 3, 4))
    grids_deviation = grids - np.expand_dims(features_mean, axis=(0, 2, 3, 4))
    features_std2 = (grids_deviation ** 2).mean(axis=(0, 2, 3, 4))
    features_std = np.sqrt(features_std2)
    # print('features_mean', features_mean)
    # print('before:', grids[0, :, 0, 0, 0])
    # print('after:', grids_deviation[0, :, 0, 0, 0])
    # print('features_std2', features_std2)
    # print('features_std', features_std)
    # print(grids_deviation)
    return features_mean, features_std


