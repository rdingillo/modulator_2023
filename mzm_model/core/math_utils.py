import numpy as np
from numpy import log10, seterr, exp

seterr(all='ignore')

# Conversion functions


def db2lin(value):
    lin_value = 10 ** (value / 10)
    return lin_value

# to pass from dBm to Watts
def dbm2lin(value):
    lin_value = (10 ** (value / 10))*1e-3
    return lin_value

def lin2db(value):
    db_value = 10*np.log10(value)
    return db_value


def lin2dbm(value):
    dbm_value = 10*np.log10(value/0.001)
    return dbm_value


def alfa2lin(alfa_db):
    alfa_lin = alfa_db/1e3/(20*np.log10(np.e))
    return alfa_lin


# def lin2dBm(array):
#     return lin2db(array) + 30
#
#
# def dBm2lin(array):
#     return db2lin(array) * 1e-3
#
def normalize(min_val, max_val, data):
    """
    Normalize data with respect to a minimum and maximum range
    """

    # minimum and maximum values of original data
    min_data = min(data)
    max_data = max(data)

    # new range
    new_min = min_val
    new_max = max_val

    # convert data to new range
    norm_data = [((x - min_data) / (max_data - min_data)) * (new_max - new_min) + new_min for x in data]

    # norm_data = []
    # for val in data:
    #     norm_val = (val - min_val) / (max_val - min_val)
    #     norm_data.append(norm_val)
    return np.array(norm_data)
