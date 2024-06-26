import os
import numpy as np
from pprint import pprint
import shutil
from scripts.signal_manipulation import *


def draw_x_authorized(x):
    authorized = os.path.join('..', 'data', 'database', 'incoming_users', 'authorized_users')
    authorized_ids = os.listdir(authorized)
    if '.DS_Store' in authorized_ids:
        authorized_ids.remove('.DS_Store')

    authorized_wavs = []
    for i in authorized_ids:
        for j in os.listdir(os.path.join(authorized, i)):
            if j.endswith('.wav'):
                authorized_wavs.append(os.path.join(authorized, i, j))
    r_wavs = []
    for i in range(x):
        r_wavs.append(np.random.choice(authorized_wavs))
    # r_wavs = ['../data/database/incoming_users/authorized_users/id10004/lu_eVSfv3Tg_00034.wav',
    #           ... ]
    return r_wavs


def draw_x_unauthorized(x):
    unauthorized = os.path.join('..', 'data', 'database', 'incoming_users', 'unauthorized_users')
    unauthorized_ids = os.listdir(unauthorized)
    if '.DS_Store' in unauthorized_ids:
        unauthorized_ids.remove('.DS_Store')

    unauthorized_wavs = []
    for i in unauthorized_ids:
        for j in os.listdir(os.path.join(unauthorized, i)):
            if j.endswith('.wav'):
                unauthorized_wavs.append(os.path.join(unauthorized, i, j))
    r_wavs = []
    for i in range(x):
        r_wavs.append(np.random.choice(unauthorized_wavs))
    # r_wavs = ['../data/database/incoming_users/unauthorized_users/id10004/lu_eVSfv3Tg_00034.wav',
    #           ... ]
    return r_wavs


def copy_fies(r_wavs, exp_dir):
    for i in r_wavs:
        head = os.path.split(i)[0]
        user_id = os.path.split(head)[1]
        if not os.path.exists(os.path.join(exp_dir, user_id)):
            os.makedirs(os.path.join(exp_dir, user_id))
        shutil.copy(i, os.path.join(exp_dir, user_id))


def prep_exp1(dir_name, x):
    '''
    wylosuj x probek zautoryzowanych uzytkownikow
    '''
    exp_dir = os.path.join('..', 'data', dir_name)
    if not os.path.exists(os.path.join('..', 'data', dir_name)):
        os.mkdir(exp_dir)

    r_wavs = draw_x_authorized(x)
    copy_fies(r_wavs, exp_dir)


def prep_exp2(exp1dir_name, dir_name, x):
    '''na podstawie exp1 dodaj losowe przemnozenia amplitudy'''
    if not os.path.exists(os.path.join('..', 'data', dir_name)):
        os.mkdir(os.path.join('..', 'data', dir_name))
    prep_exp1(exp1dir_name, x)
    random_amplitude_multiply(os.path.join('..', 'data', 'exp1'), os.path.join('..', 'data', dir_name))


def prep_exp3(dir_name):
    '''dla 200 probek zmiejsz czestotliwocs probek'''

    resample_rates = [0.5, 0.2, 0.1]
    x_wavs_auth = draw_x_authorized(100)
    x_wavs_unauth = draw_x_unauthorized(100)
    r_wavs = x_wavs_auth + x_wavs_unauth

    copy_fies(r_wavs, os.path.join('..', 'data', dir_name + 'temp'))
    dirs_i = []
    for i in resample_rates:
        dir_i = dir_name + f'_sr_{i}'
        dirs_i.append(os.path.join('..', 'data', dir_i))
        if not os.path.exists(os.path.join('..', 'data', dir_i)):
            os.mkdir(os.path.join('..', 'data', dir_i))
        change_sampling_rate(os.path.join('..', 'data', dir_name + 'temp'), os.path.join('..', 'data', dir_i), i)
    shutil.rmtree(os.path.join('..', 'data', dir_name + 'temp'))  # remove temp directory
    return dirs_i


def prep_exp4(dir_name):
    '''do 100 probek dodaj szum'''
    x_wavs_auth = draw_x_authorized(50)
    x_wavs_unauth = draw_x_unauthorized(50)
    r_wavs = x_wavs_auth + x_wavs_unauth
    copy_fies(r_wavs, os.path.join('..', 'data', dir_name + 'temp'))
    scal = [(0, 1), (0, 10), (5, 1)]
    dirs_i = []
    for i in scal:
        dir_i = dir_name + f'_amp_{i[0]}_{i[1]}'
        dirs_i.append(os.path.join('..', 'data', dir_i))
        if not os.path.exists(os.path.join('..', 'data', dir_i)):
            os.mkdir(os.path.join('..', 'data', dir_i))
        add_gaussian_noise(os.path.join('..', 'data', dir_name + 'temp'), os.path.join('..', 'data', dir_i), i)
    shutil.rmtree(os.path.join('..', 'data', dir_name + 'temp'))  # remove temp directory
    return dirs_i


def prep_exp5(dir_name, sound_path: str = os.path.join("..", "data", "electric-saw-aka-pandemia.wav")):
    '''do 100 probek dodaj nieregularne zakłócenia'''
    if not os.path.exists(os.path.join('..', 'data', dir_name)):
        os.mkdir(os.path.join('..', 'data', dir_name))
    x_wavs_auth = draw_x_authorized(50)
    x_wavs_unauth = draw_x_unauthorized(50)
    r_wavs = x_wavs_auth + x_wavs_unauth
    copy_fies(r_wavs, os.path.join('..', 'data', dir_name + 'temp'))
    add_irregular_sound(os.path.join('..', 'data', dir_name + 'temp'), os.path.join('..', 'data', dir_name), sound_path)
    shutil.rmtree(os.path.join('..', 'data', dir_name + 'temp'))  # remove temp directory
