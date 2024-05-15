import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.cuda
import wespeaker
from sklearn.metrics import roc_curve, confusion_matrix
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
        if not os.path.exists(self.authorized_users_path):
            os.makedirs(self.authorized_users_path, exist_ok=True)
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path, exist_ok=True)

        self.initialize_database()

    def initialize_model(self):
        model = wespeaker.load_model("english")
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
        model.set_gpu(device_id)
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

        is_access_granted = (
            user_name == predicted_user_name and confidence >= self.acceptance_threshold
        )
        if self.allow_brute_force:
            system_user_names = os.listdir(self.authorized_users_path)

            if user_name in system_user_names:
                system_user_names.remove(user_name)

            is_access_granted = (
                predicted_user_name in system_user_names
                and confidence >= self.acceptance_threshold
            )
            
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

    def calculate_ROC_curve(
        self,
        df_users_authorized: pd.DataFrame,
        df_users_unauthorized: pd.DataFrame,
        roc_curve_path: str = None,
    ) -> Tuple[int, int, int, int]:
        """
        Function to calculate ROC curve based on DFs with authorized and unauthorized users,
        based on changing threshold of confidence. Saves ROC curve if roc_curve_path is provided.
        :param df_users_authorized: DF with users in database
        :param df_users_unauthorized: DF with users that are not authorized in database
        :param roc_curve_path: path to save ROC curve plot
        :return: Tuple of TN, FP, FN, TP
        """
        df_concatenated = pd.concat(
            [df_users_authorized, df_users_unauthorized], axis=0
        )
        true_labels = [True] * len(df_users_authorized) + [False] * len(
            df_users_unauthorized
        )
        predicted_labels = df_concatenated["is_access_granted"].to_list()
        confidences = df_concatenated["confidence"]

        non_inf_values = (
            confidences.replace([np.inf, -np.inf], np.nan).dropna().unique()
        )
        non_inf_values_sorted = np.sort(non_inf_values)[::-1]
        second_max_value = (
            non_inf_values_sorted[1]
            if len(non_inf_values_sorted) > 1
            and len(non_inf_values) != len(confidences.unique())
            else non_inf_values_sorted[0]
        )  # get valid confidences from system
        confidences.replace(
            [np.inf], second_max_value, inplace=True
        )  # replace inf confidence with max available confidence, which is not np.inf -> system errored finding match
        probabilities = (
            second_max_value - confidences
        )  # analyze confidences as probabilities of accepting by system

        acceptance_threshold_scaled = (
            self.acceptance_threshold - probabilities.min()
        ) / (probabilities.max() - probabilities.min())
        scale_factor = (
            0.5 / acceptance_threshold_scaled
        )  # make sure that threshold is in the middle of probabilities
        probabilities_rescaled = (probabilities - probabilities.min()) * scale_factor

        if roc_curve_path is not None:
            roc_curve_directory = os.path.dirname(roc_curve_path)
            if not os.path.exists(roc_curve_directory):
                os.makedirs(roc_curve_directory, exist_ok=True)

            fpr, tpr, thresholds = roc_curve(true_labels, probabilities_rescaled)
            plt.figure(figsize=(16, 9), dpi=200)
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.tight_layout()
            plt.savefig(roc_curve_path, dpi=200)
            plt.close()

        tn, fp, fn, tp = confusion_matrix(
            y_true=true_labels, y_pred=predicted_labels
        ).ravel()
        return tn, fp, fn, tp

    @staticmethod
    def calculate_access_granted_rate(
        df_users: pd.DataFrame,
    ) -> float:
        return df_users["is_access_granted"].sum() / len(df_users)

    @staticmethod
    def calculate_far_frr(
        df_users_authorized: pd.DataFrame, df_users_unauthorized: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Function to calculate False Acceptance Rate, False Rejection Rate

        :param df_users_authorized: DF with users in database
        :param df_users_unauthorized: DF with users that are not authorized in database
        :return: False Acceptance Rate, False Rejection Rate
        """
        df_concatenated = pd.concat(
            [df_users_authorized, df_users_unauthorized], axis=0
        )
        true_labels = [True] * len(df_users_authorized) + [False] * len(
            df_users_unauthorized
        )
        predicted_labels = df_concatenated["is_access_granted"].to_list()
        tn, fp, fn, tp = confusion_matrix(
            y_true=true_labels, y_pred=predicted_labels
        ).ravel()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        frr = 1 - tpr
        far = fpr
        return far, frr
