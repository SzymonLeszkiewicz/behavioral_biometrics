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

    return r_wavs


def copy_fies(r_wavs, exp_dir):
    for i in r_wavs:
        # copy file to exp_dir/authorized_users/user_id
        user_id = i.split('/')[-2]
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
    r_wavs = draw_x_authorized(200)
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
    r_wavs = draw_x_authorized(100)
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

def prep_exp5(dir_name, sound_path : str = os.path.join("..", "data", "electric-saw-aka-pandemia.wav")):
    '''do 100 probek dodaj nieregularne zakłócenia'''
    if not os.path.exists(os.path.join('..', 'data', dir_name)):
        os.mkdir(os.path.join('..', 'data', dir_name))
    r_wavs = draw_x_authorized(100)
    copy_fies(r_wavs, os.path.join('..', 'data', dir_name + 'temp'))
    add_irregular_sound(os.path.join('..', 'data', dir_name + 'temp'), os.path.join('..', 'data', dir_name), sound_path)
    shutil.rmtree(os.path.join('..', 'data', dir_name + 'temp'))  # remove temp directory
