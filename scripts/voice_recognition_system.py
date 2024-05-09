import os

import pandas as pd
import wespeaker
from tqdm import tqdm


class VoiceRecognitionSystem:
    def __init__(
        self,
        database_path: str,
        database_embeddings_path: str = None,
        acceptance_threshold: float = 0.5,
        enable_multiple_audio_file_system: bool = False,
        allow_brute_force: bool = False,
    ):
        self.database_path = database_path
        self.database_embeddings_path = database_embeddings_path
        self.acceptance_threshold = acceptance_threshold
        self.enable_multiple_audio_file_system = enable_multiple_audio_file_system
        self.allow_brute_force = allow_brute_force

        self.authorized_users_path = os.path.join(
            self.database_path, "authorized_users"
        )

        self.model = self.initialize_model()
        self.initialize_database()

    def initialize_model(self):
        model = wespeaker.load_model("english")
        model.set_gpu(0)
        return model

    def initialize_database(self) -> None:
        if self.database_embeddings_path:
            # load embeddings from file
            raise NotImplementedError
        if self.enable_multiple_audio_file_system:
            # user can be defined by more than one audio file
            # user is represented as a mean of audio file embeddings
            raise NotImplementedError

        for user_folder in tqdm(
            os.listdir(self.authorized_users_path), desc="Initializing database"
        ):
            user_path = os.path.join(self.authorized_users_path, user_folder)
            if os.path.isdir(user_path):
                user_audio_files = [
                    audio_file
                    for audio_file in os.listdir(user_path)
                    if audio_file.endswith(".wav")
                ]
                if user_audio_files:
                    first_audio_file = os.path.join(user_path, user_audio_files[0])
                    self.model.register(name=user_folder, audio_path=first_audio_file)

    def verify_user(self, user_name: str, user_voice_path: str):
        if self.enable_multiple_audio_file_system:
            raise NotImplementedError

        predicted_user_name, confidence = self.model.recognize(
            audio_path=user_voice_path
        ).values()

        is_access_granted = user_name == predicted_user_name
        if self.allow_brute_force:
            system_user_names = os.listdir(self.authorized_users_path)
            is_access_granted = predicted_user_name in system_user_names

        return is_access_granted, confidence

    def verify_multiple_users(self, incoming_users_path: str):
        df_users = pd.DataFrame(
            columns=[
                "audio_path",
                "is_access_granted",
                "confidence",
            ]
        )

        for user_folder in tqdm(
            os.listdir(incoming_users_path), desc="Veryfing multiple users"
        ):
            user_path = os.path.join(incoming_users_path, user_folder)
            if os.path.isdir(user_path):
                user_audio_files = [
                    audio_file
                    for audio_file in os.listdir(user_path)
                    if audio_file.endswith(".wav")
                ]
                if user_audio_files:
                    for audio_file in user_audio_files:
                        is_access_granted, confidence = self.verify_user(
                            user_name=user_folder,
                            user_voice_path=os.path.join(user_path, audio_file),
                        )
                        df_user = pd.DataFrame(
                            {
                                "audio_path": os.path.join(user_path, audio_file),
                                "is_access_granted": is_access_granted,
                                "confidence": confidence,
                            },
                            index=[0],
                        )
                        df_users = pd.concat([df_users, df_user], ignore_index=True)
        return df_users
