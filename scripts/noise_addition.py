import os
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
import librosa


def add_gaussian_noise(
    input_dir: str, output_dir: str, noise_tuple: tuple[int, int | float]
):
    """
    Function walks through the input directory and adds gaussian noise according to passed tuple of loc, scale.
    Function writes .wav files into the output directory, creating the same directories structure and adding to original name "_loc_{loc}_scale_{scale}.wav"
    :param input_dir: Directory from which .wav files should be from
    :param output_dir: Directory to which noised .wav files should be written
    :param noise_tuple: Tuple of loc, scale Gaussian noise
    """
    loc, scale = noise_tuple
    for dirpath, dirnames, filenames in tqdm(
        os.walk(input_dir),
        desc="Users computed",
        colour="magenta",
        total=len(os.listdir(input_dir)),
        leave=False,
    ):
        structure = os.path.join(output_dir, os.path.relpath(dirpath, input_dir))
        if not os.path.isdir(structure):  # create the same structure
            os.mkdir(structure)
        for name in filenames:
            filename = str(os.path.join(dirpath, name))
            noise_filename = os.path.join(
                structure, f"{name[:-4:]}_loc_{loc}_scale_{scale}.wav"
            )
            signal, sample_rate = sf.read(filename)

            noise = np.random.normal(loc=loc, scale=scale, size=signal.shape[0])
            signal_noise = signal + noise
            sf.write(file=noise_filename, data=signal_noise, samplerate=sample_rate)


def random_amplitude_multiply(input_dir: str, output_dir: str):
    """
    Function walks through the input directory and multiplies amplitude according to random choice from factors {25, 1, 0.04}
    Function writes .wav files into the output directory, creating the same directories structure and adding to original name "_amp_{scale_factor}.wav", where scale_factor is chosen factor.
    :param input_dir: Directory from which .wav files should be read
    :param output_dir: Directory to which .wav files should be written
    """
    for dirpath, dirnames, filenames in tqdm(
        os.walk(input_dir),
        desc="Users computed",
        colour="magenta",
        total=len(os.listdir(input_dir)),
        leave=False,
    ):
        structure = os.path.join(output_dir, os.path.relpath(dirpath, input_dir))
        if not os.path.isdir(structure):  # create the same structure
            os.mkdir(structure)
        for name in filenames:
            filename = str(os.path.join(dirpath, name))
            scale_factor = np.random.choice([25, 1, 0.04])  # perform random choice
            noise_filename = os.path.join(
                structure, f"{name[:-4:]}_amp_{scale_factor}.wav"
            )

            signal, sample_rate = sf.read(filename)
            signal_multiplied = signal * scale_factor
            sf.write(
                file=noise_filename, data=signal_multiplied, samplerate=sample_rate
            )


def change_sampling_rate(
    input_dir: str, output_dir: str, sampling_rate_factor: float = 0.5
):
    """
    Function walks through the input directory and changes the sampling rate according to provided factor
    Function writes .wav files into the output directory, creating the same directories structure and adding to \
    original name "_sr_{sampling_rate_factor}.wav", where sampling_rate_factor is provided factor.
    :param input_dir: Directory from which .wav files should be read
    :param output_dir: Directory to which .wav files should be written
    :param sampling_rate_factor: Sampling rate that .wav file should be resampled into
    :return:
    """
    for dirpath, dirnames, filenames in tqdm(
        os.walk(input_dir),
        desc="Users computed",
        colour="magenta",
        total=len(os.listdir(input_dir)),
        leave=False,
    ):
        structure = os.path.join(output_dir, os.path.relpath(dirpath, input_dir))
        if not os.path.isdir(structure):  # create the same structure
            os.mkdir(structure)
        for name in filenames:
            filename = str(os.path.join(dirpath, name))
            noise_filename = os.path.join(
                structure, f"{name[:-4:]}_sr_{sampling_rate_factor}.wav"
            )

            signal, sample_rate = sf.read(filename)
            signal = signal.T
            target_samplerate = int(sample_rate * sampling_rate_factor)
            data_resampled = librosa.resample(
                signal, orig_sr=sample_rate, target_sr=target_samplerate
            )
            sf.write(
                file=noise_filename, data=data_resampled, samplerate=target_samplerate
            )
