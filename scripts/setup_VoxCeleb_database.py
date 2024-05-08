import os
import random
import shutil

from tqdm import tqdm

"""
Authorized users:
* get first 100 sorted users
* for each user shuffle all his songs
    * 70% of songs go to authorized users
    * 30% of songs go to incoming authorized users
        * limited to 50 users

Unauthorized users:
* get 100-150 sorted users
* for each user shuffle all his songs
    * 30% of songs go incoming unauthorized users

Current database format
data/
    database/
            authorized_users/
                            /1
                                audio_1.jpg
                                audio_2.jpg
                            ...
                            /100
                                audio_1.jpg
                                audio_2.jpg
            incoming_users/
                            authorized_users/
                                            /1
                                                audio_3.jpg
                                                audio_4.jpg
                                            ...
                                            /50
                                                audio_3.jpg
                                                audio_4.jpg
                            unauthorized_users/
                                            /101
                                                audio_1.jpg
                                                audio_2.jpg
                                            ...
                                            /150
                                                audio_1.jpg
                                                audio_2.jpg
"""

parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_directory = os.path.join(parent_directory, "data")
database_directory = os.path.join(data_directory, "database")

VoxCeleb_subset_path = os.path.join(data_directory, "VoxCelebSubset")
VoxCeleb_path = os.path.join(data_directory, "VoxCeleb")

authorized_users_path = os.path.join(data_directory, "database", "authorized_users")
incoming_authorized_users_path = os.path.join(
    data_directory, "database", "incoming_users", "authorized_users"
)
incoming_unauthorized_users_path = os.path.join(
    data_directory, "database", "incoming_users", "unauthorized_users"
)


def setup_VoxCeleb_database():
    create_VoxCeleb_subset()
    rename_audio_files()
    setup_authorized_users()
    setup_unauthorized_users()


def create_VoxCeleb_subset():
    if not os.path.exists(VoxCeleb_subset_path):
        os.makedirs(VoxCeleb_subset_path)

        sorted_users = sorted(
            [
                user
                for user in os.listdir(VoxCeleb_path)
                if os.path.isdir(os.path.join(VoxCeleb_path, user))
            ]
        )

        selected_users = sorted_users[:150]

        for user in tqdm(selected_users, desc="Creating VoxCeleb subset"):
            source_path = os.path.join(VoxCeleb_path, user)
            destination_path = os.path.join(VoxCeleb_subset_path, user)
            shutil.copytree(source_path, destination_path)
    else:
        print("VoxCeleb subset already created!")


def rename_audio_files():
    # Iterate over each user directory
    for user in tqdm(os.listdir(VoxCeleb_subset_path), desc="Renaming audio files"):
        user_path = os.path.join(VoxCeleb_subset_path, user)
        if not os.path.isdir(user_path):
            continue

        # Iterate over each conversation directory
        for conversation in os.listdir(user_path):
            conversation_path = os.path.join(user_path, conversation)
            if not os.path.isdir(conversation_path):
                continue

            # Iterate over each audio file
            for audio_file in os.listdir(conversation_path):
                if audio_file.endswith(".wav"):
                    if audio_file.startswith(conversation):
                        print("Files already renamed!")
                        return
                    new_file_name = f"{conversation}_{audio_file}"
                    old_file_path = os.path.join(conversation_path, audio_file)
                    new_file_path = os.path.join(conversation_path, new_file_name)
                    os.rename(old_file_path, new_file_path)


def setup_authorized_users():
    if not os.path.exists(authorized_users_path) and not os.path.exists(
        incoming_authorized_users_path
    ):
        os.makedirs(authorized_users_path, exist_ok=True)
        os.makedirs(incoming_authorized_users_path, exist_ok=True)

        sorted_users = sorted(
            [
                user
                for user in os.listdir(VoxCeleb_subset_path)
                if os.path.isdir(os.path.join(VoxCeleb_subset_path, user))
            ]
        )
        authorized_users = sorted_users[:100]
        user_count = 0

        for user_id in tqdm(
            authorized_users, desc="Setting up authorized users dataset"
        ):
            user_directory = os.path.join(VoxCeleb_subset_path, user_id)
            user_audio_files = []
            for root, dirs, files in os.walk(user_directory):
                for file in files:
                    if file.endswith(".wav"):
                        user_audio_files.append(os.path.join(root, file))

            number_of_files = len(user_audio_files)
            number_of_files_authorized = int(0.7 * number_of_files)
            random.shuffle(user_audio_files)
            authorized_users_files = user_audio_files[:number_of_files_authorized]
            incoming_authorized_users_files = user_audio_files[
                number_of_files_authorized:
            ]

            os.makedirs(os.path.join(authorized_users_path, user_id), exist_ok=True)
            for file in authorized_users_files:
                shutil.copy(
                    file,
                    os.path.join(
                        authorized_users_path, user_id, os.path.basename(file)
                    ),
                )

            if user_count >= 50:
                continue

            os.makedirs(
                os.path.join(incoming_authorized_users_path, user_id), exist_ok=True
            )

            for file in incoming_authorized_users_files:
                shutil.copy(
                    file,
                    os.path.join(
                        incoming_authorized_users_path, user_id, os.path.basename(file)
                    ),
                )
            user_count += 1
    else:
        print("Authorized users already setup!")


def setup_unauthorized_users():
    if not os.path.exists(incoming_unauthorized_users_path):
        os.makedirs(incoming_unauthorized_users_path, exist_ok=True)

        sorted_users = sorted(
            [
                user
                for user in os.listdir(VoxCeleb_subset_path)
                if os.path.isdir(os.path.join(VoxCeleb_subset_path, user))
            ]
        )
        unauthorized_users = sorted_users[100:150]

        for user_id in tqdm(
            unauthorized_users, desc="Setting up unauthorized users dataset"
        ):
            user_directory = os.path.join(VoxCeleb_subset_path, user_id)
            user_audio_files = []
            for root, dirs, files in os.walk(user_directory):
                for file in files:
                    if file.endswith(".wav"):
                        user_audio_files.append(os.path.join(root, file))

            number_of_files = len(user_audio_files)
            number_of_files_unauthorized = int(0.3 * number_of_files)
            random.shuffle(user_audio_files)
            incoming_unauthorized_users_files = user_audio_files[
                :number_of_files_unauthorized
            ]

            os.makedirs(
                os.path.join(incoming_unauthorized_users_path, user_id), exist_ok=True
            )
            for file in incoming_unauthorized_users_files:
                shutil.copy(
                    file,
                    os.path.join(
                        incoming_unauthorized_users_path,
                        user_id,
                        os.path.basename(file),
                    ),
                )
    else:
        print("Unauthorized users already setup!")


setup_VoxCeleb_database()
