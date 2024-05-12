import os
import shutil

import numpy as np

from scripts import signal_manipulation

np.random.seed(42)


def draw_number_of_users(users_path: str, number_of_users: int) -> list[str]:
    users_ids = os.listdir(users_path)
    if ".DS_Store" in users_ids:
        users_ids.remove(".DS_Store")

    audio_files = []
    for user_id in users_ids:
        for user_audio_file in os.listdir(os.path.join(users_path, user_id)):
            if user_audio_file.endswith(".wav"):
                audio_files.append(os.path.join(users_path, user_id, user_audio_file))
    random_audio_files = np.random.choice(
        audio_files, size=number_of_users, replace=False
    ).tolist()

    return random_audio_files


def copy_files(random_audio_files: list[str], experiment_directory: str):
    for audio_file in random_audio_files:
        user_id = os.path.dirname(audio_file).split(os.path.sep)[-1]
        if not os.path.exists(os.path.join(experiment_directory, user_id)):
            os.makedirs(os.path.join(experiment_directory, user_id), exist_ok=True)
        shutil.copy(audio_file, os.path.join(experiment_directory, user_id))


def prepare_experiment_1(
    experiment_path: str,
    authorized_users_path: str,
    unauthorized_users_path: str,
    number_of_users: int,
):
    """
    Draw fixed number of authorized and unauthorized users
    """
    experiment_authorized_path = os.path.join(experiment_path, "authorized")
    experiment_unauthorized_path = os.path.join(experiment_path, "unauthorized")

    if not os.path.exists(experiment_authorized_path):
        os.makedirs(experiment_authorized_path, exist_ok=True)
    if not os.path.exists(experiment_unauthorized_path):
        os.makedirs(experiment_unauthorized_path, exist_ok=True)

    random_authorized_audio_files = draw_number_of_users(
        authorized_users_path, number_of_users
    )
    random_unauthorized_audio_files = draw_number_of_users(
        unauthorized_users_path, number_of_users
    )

    copy_files(random_authorized_audio_files, experiment_authorized_path)
    copy_files(random_unauthorized_audio_files, experiment_unauthorized_path)


def prepare_experiment_2(
    experiment_1_path: str,
    experiment_2_path: str,
    authorized_users_path: str,
    unauthorized_users_path: str,
    number_of_users: int,
):
    """
    Based on Experiment 1, add random amplitude multiply
    """
    if not os.path.exists(experiment_1_path):
        prepare_experiment_1(
            experiment_1_path,
            authorized_users_path,
            unauthorized_users_path,
            number_of_users,
        )

    experiment_1_authorized_path = os.path.join(experiment_1_path, "authorized")
    experiment_1_unauthorized_path = os.path.join(experiment_1_path, "unauthorized")

    experiment_2_authorized_path = os.path.join(experiment_2_path, "authorized")
    experiment_2_unauthorized_path = os.path.join(experiment_2_path, "unauthorized")

    if not os.path.exists(experiment_2_authorized_path):
        os.makedirs(experiment_2_authorized_path, exist_ok=True)
    if not os.path.exists(experiment_2_unauthorized_path):
        os.makedirs(experiment_2_unauthorized_path, exist_ok=True)

    signal_manipulation.random_amplitude_multiply(
        experiment_1_authorized_path, experiment_2_authorized_path
    )
    signal_manipulation.random_amplitude_multiply(
        experiment_1_unauthorized_path, experiment_2_unauthorized_path
    )


def prepare_experiment_3(
    experiment_path: str,
    authorized_users_path: str,
    unauthorized_users_path: str,
    number_of_users: int = 200,
):
    """
    Reduce the sampling rate
    """
    sample_rates_reduction = [0.5, 0.2, 0.1]
    random_audio_files_authorized = draw_number_of_users(
        authorized_users_path, number_of_users
    )
    random_audio_files_unauthorized = draw_number_of_users(
        unauthorized_users_path, number_of_users
    )
    copy_files(
        random_audio_files_authorized, os.path.join(experiment_path, "authorized_temp")
    )
    copy_files(
        random_audio_files_unauthorized,
        os.path.join(experiment_path, "unauthorized_temp"),
    )

    for sample_rate in sample_rates_reduction:
        sample_rate_authorized_directory = os.path.join(
            experiment_path, f"sample_rate_{sample_rate}", "authorized"
        )
        sample_rate_unauthorized_directory = os.path.join(
            experiment_path, f"sample_rate_{sample_rate}", "unauthorized"
        )
        if not os.path.exists(sample_rate_authorized_directory):
            os.makedirs(sample_rate_authorized_directory, exist_ok=True)
        if not os.path.exists(sample_rate_unauthorized_directory):
            os.makedirs(sample_rate_unauthorized_directory, exist_ok=True)
        signal_manipulation.change_sampling_rate(
            os.path.join(experiment_path, "authorized_temp"),
            sample_rate_authorized_directory,
            sample_rate,
        )
        signal_manipulation.change_sampling_rate(
            os.path.join(experiment_path, "unauthorized_temp"),
            sample_rate_unauthorized_directory,
            sample_rate,
        )
    # remove temporary directories
    shutil.rmtree(os.path.join(experiment_path, "authorized_temp"))
    shutil.rmtree(os.path.join(experiment_path, "unauthorized_temp"))


# def prepare_experiment_4(
#     experiment_path: str, authorized_users_path: str, number_of_users: int = 100
# ):
#     """do 100 probek dodaj szum"""
#     random_audio_files = draw_number_of_users_authorized(
#         authorized_users_path, number_of_users
#     )
#     copy_files(random_audio_files, experiment_path + "_temp")
#     scal = [(0, 1), (0, 10), (5, 1)]
#     dirs_i = []
#     for i in scal:
#         dir_i = dir_name + f"_amp_{i[0]}_{i[1]}"
#         dirs_i.append(os.path.join("..", "data", dir_i))
#         if not os.path.exists(os.path.join("..", "data", dir_i)):
#             os.mkdir(os.path.join("..", "data", dir_i))
#         signal_manipulation.add_gaussian_noise(
#             os.path.join("..", "data", dir_name + "temp"),
#             os.path.join("..", "data", dir_i),
#             i,
#         )
#     shutil.rmtree(
#         os.path.join("..", "data", dir_name + "temp")
#     )  # remove temp directory
#     return dirs_i
#
#
# def prep_exp5(
#     dir_name,
#     sound_path: str = os.path.join("..", "data", "electric-saw-aka-pandemia.wav"),
# ):
#     """do 100 probek dodaj nieregularne zakłócenia"""
#     if not os.path.exists(os.path.join("..", "data", dir_name)):
#         os.mkdir(os.path.join("..", "data", dir_name))
#     r_wavs = draw_x_authorized(100)
#     copy_fies(r_wavs, os.path.join("..", "data", dir_name + "temp"))
#     signal_manipulation.add_irregular_sound(
#         os.path.join("..", "data", dir_name + "temp"),
#         os.path.join("..", "data", dir_name),
#         sound_path,
#     )
#     shutil.rmtree(
#         os.path.join("..", "data", dir_name + "temp")
#     )  # remove temp directory
