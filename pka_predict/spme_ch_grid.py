import math

import numpy as np
from numba import njit, float32, float64


# ================================================================
# API
# ================================================================
def get_charge_grid(order: int, csv_file=None, data=None):
    """
    Construct the grid charge array q[box_dim1][box_dim2][box_dim3] with the charges and the B-Spline coefficients,
    where box_dim refers to the size of particle mesh in Angstrom. And, the value of B-Spline coefficients depends on
    the location of the particles.
    :param order: interpolation order, an integer that determines the order of Spline interpolation,
                  value of 4 is typical, higher accuracy is around o=10.
                  The <order> must be an even number and at least 4.
    :param csv_file: optional(str or file), a csv-file containing particle information with column names
                    in 'x','y','z','partialcharge' format
    :param data: list, an array of the size of [n][4] with subscripts corresponding to 'x','y','z','partialcharge'
    :returns: ch_grid: ndarray, the grid charge array
              offset: ndarray, the offset of the index of ch_grid from the actual coordinates
    """
    if order < 4 or (order & 1) != 0:
        raise ValueError('the <order> must be an even number and at least 4')
    if csv_file is not None and data is not None:
        raise ValueError('only <csv_file> or the other <data> can be specified')
    if csv_file is not None:
        numatoms, coord, charge = get_from_file(csv_file)
    elif data is not None:
        numatoms, coord, charge = get_from_data(data)
    else:
        raise ValueError('<csv_file> or <data> should be specified')
    # get bspline coefficients
    theta1, theta2, theta3 = get_bspline_coeffs(numatoms, order, coord)
    # cal offset & box-dimension.
    # left offset = order-2. right offset = 2
    offset_f = np.floor(np.min(coord, axis=0)) - order + 2
    box_dim_f = np.max(coord, axis=0) + 2 - offset_f
    offset = offset_f.astype(int)
    box_dim = box_dim_f.astype(int)
    # get grid charge array
    ch_grid = fill_ch_grid(numatoms, order, coord, charge, theta1, theta2, theta3, offset, box_dim)
    # move to absolute coord
    offset += (order - 1) // 2
    return ch_grid, offset


def save_ch_grid2npz(fname, **kwargs):
    """
    Do get_charge_grid() function and save .npz file with dict.
    The format is the dictionary shown:
    {
        offset: offset (ndarray),
        grid_array: grid_array (ndarray)
    }
    Usage:
        save_ch_grid2npz(fname='grid', order=4, csv_file='particles.csv')
        dic = np.load('grid.npz', allow_pickle=True)
    :param fname：filename or file.
    """
    ch_grid, offset = get_charge_grid(**kwargs)
    grid_array = []
    for i in range(ch_grid.shape[0]):
        for j in range(ch_grid.shape[1]):
            for k in range(ch_grid.shape[2]):
                grid_array.append((i + offset[0], j + offset[1], k + offset[2], ch_grid[i][j][k]))
    grid_array = np.asarray(grid_array)
    np.savez(fname, offset=offset, grid_array=grid_array)


# ================================================================
# Backend
# ================================================================
def fill_ch_grid(numatoms, order, coord, charge, theta1, theta2, theta3, offset, box_dim):
    """
    Compute the grid charge array.
    For each charge, the counters i, j, and k are responsible to go through all the grid points
    that the charges have to interpolate to in the three dimensional space.
    Note: offset is used to adjust the index of the array, to ensure that the index is non-negative.
    :param numatoms: int, number of atoms
    :param order: int, the order of spline interpolation
    :param coord: ndarray, the coords of atoms
    :param charge: ndarray, the charges of atoms
    :param theta1: ndarray, the bspline coeffs array (x-coord)
    :param theta2: ndarray, the bspline coeffs array (y-coord)
    :param theta3: ndarray, the bspline coeffs array (z-coord)
    :param offset: ndarray, the offset of grid array
    :param box_dim: ndarray, the charge grid dimensions
    :return: the charge grid
    """
    charge_grid = np.zeros(box_dim)
    for n in range(numatoms):
        k0 = math.floor(coord[n][2]) - order
        for ith3 in range(order):
            k0 = k0 + 1
            k = k0 + 1
            j0 = math.floor(coord[n][1]) - order
            for ith2 in range(order):
                j0 = j0 + 1
                j = j0 + 1
                prod = multi3(theta2[n][ith2], theta3[n][ith3], charge[n][0])
                i0 = math.floor(coord[n][0]) - order
                for ith1 in range(order):
                    i0 = i0 + 1
                    i = i0 + 1
                    tmp = charge_grid[i - offset[0]][j - offset[1]][k - offset[2]]
                    charge_grid[i - offset[0]][j - offset[1]][k - offset[2]] = \
                        plus2(tmp, multi3(prod, theta1[n][ith1], 1))
    return charge_grid


def get_bspline_coeffs(numatoms, order, coord):
    """
    Compute the B-Spline coefficients.
    The actual calculations are done in fill_bspline() function.
    The B-Spline coefficients can be seen as influence/weights of a particle at the nearby grid points.
    :param numatoms: int, number of atoms
    :param order: int, the order of spline interpolation
    :param coord: ndarray, the coords of atoms
    :returns theta1,theta2,theta3: ndarray, the spline coefficient arrays
    """
    theta1 = np.empty((numatoms, order))
    theta2 = np.empty((numatoms, order))
    theta3 = np.empty((numatoms, order))
    for i in range(numatoms):
        w = get_w(coord[i][0])
        fill_bspline(w, order, theta1[i])
        w = get_w(coord[i][1])
        fill_bspline(w, order, theta2[i])
        w = get_w(coord[i][2])
        fill_bspline(w, order, theta3[i])
    return theta1, theta2, theta3


@njit(cache=True)
def fill_bspline(w, order, theta):
    """
    Compute the B-Spline coefficients for each particle.
    Furthermore: <@njit> decorator is used to compile a Python function into native code.
    """
    # do linear case
    init(theta, w, order)
    # compute standard b-spline recursion
    for k in range(3, order + 1):
        one_pass(theta, w, k)


@njit(cache=True)
def init(theta, x, order):
    theta[order - 1] = 0.
    theta[1] = x
    theta[0] = 1. - x


@njit(cache=True)
def one_pass(theta, x, k):
    div = 1. / (k - 1)
    theta[k - 1] = div * x * theta[k - 2]
    for j in range(1, k - 1):
        theta[k - j - 1] = div * ((x + j) * theta[k - j - 2] + (k - j - x) * theta[k - j - 1])
    theta[0] = div * (1 - x) * theta[0]


@njit(cache=True)
def get_w(coord):
    return coord - math.floor(coord)


def read_csv(csv_file):
    import pandas
    if isinstance(csv_file, str):
        file = open(csv_file, 'rt')
    else:
        assert hasattr(csv_file, 'read'), '<csv_file> expected file or str, got %s' % csv_file
        file = csv_file
    return pandas.read_csv(file, index_col=None, comment='#')


def get_from_file(csv_file):
    df = read_csv(csv_file)
    return get_from_data(df)


def get_from_data(df):
    numatoms = df.shape[0]
    coord = df.loc[:, ['x', 'y', 'z']].to_numpy()
    charge = df.loc[:, ['partialcharge']].to_numpy()
    return numatoms, coord, charge


@njit([float32(float32, float32, float32), float64(float64, float64, float64)], cache=True)
def multi3(a, b, c):
    return a * b * c


@njit([float32(float32, float32), float64(float64, float64)], cache=True)
def plus2(a, b):
    return a + b
