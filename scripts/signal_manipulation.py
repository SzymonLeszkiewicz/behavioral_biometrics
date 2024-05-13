import os
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
import librosa
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def add_gaussian_noise(
    input_dir: str, output_dir: str, noise_tuple: tuple[int, int | float]
) -> None:
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


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def add_gaussian_noise_with_lowpass(
    input_dir: str, output_dir: str, noise_tuple: tuple[int, int | float]
) -> None:
    """
    Function walks through the input directory and adds gaussian noise according to passed tuple of loc, scale. \
    After that, it tries to filter out noise.
    Function writes .wav files into the output directory, creating the same directories structure and adding to original \
    name "_loc_{loc}_scale_{scale}_lowpass_filer.wav"
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
                structure, f"{name[:-4:]}_loc_{loc}_scale_{scale}_lowpass_filer.wav"
            )
            signal, sample_rate = sf.read(filename)

            noise = np.random.normal(loc=loc, scale=scale, size=signal.shape[0])
            signal_noise = signal + noise

            filtered_data = lowpass_filter(signal_noise, 1000, sample_rate)

            sf.write(file=noise_filename, data=filtered_data, samplerate=sample_rate)


def random_amplitude_multiply(input_dir: str, output_dir: str) -> None:
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
) -> None:
    """
    Function walks through the input directory and changes the sampling rate according to provided factor
    Function writes .wav files into the output directory, creating the same directories structure and adding to \
    original name "_sr_{sampling_rate_factor}.wav", where sampling_rate_factor is provided factor.
    :param input_dir: Directory from which .wav files should be read
    :param output_dir: Directory to which .wav files should be written
    :param sampling_rate_factor: Sampling rate that .wav file should be resampled into
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


def add_irregular_sound(
    input_dir: str,
    output_dir: str,
    disturbance_signal_path: str = os.path.join(
        "..", "data", "electric-saw-aka-pandemia.wav"
    ),
) -> None:
    """
    Function walks through the input directory and adds disturbance signal pandemic sound - electric saw while renovation by default.
    Function writes .wav files into the output directory, creating the same directories structure and adding to \
    original name "_irregular_signal.wav".
    :param input_dir: Directory from which .wav files should be read
    :param output_dir: Directory to which .wav files should be written
    :param disturbance_signal_path: Signal that should be added to .wav files as irregular sound
    """

    disturbance_signal, disturbance_sample_rate = sf.read(disturbance_signal_path)
    if len(disturbance_signal.shape) != 1:
        disturbance_signal = np.mean(disturbance_signal, axis=1)
    for dirpath, dirnames, filenames in os.walk(input_dir):
        structure = os.path.join(output_dir, os.path.relpath(dirpath, input_dir))
        if not os.path.isdir(structure):  # create the same structure
            os.mkdir(structure)
        for name in filenames:
            filename = str(os.path.join(dirpath, name))
            noise_filename = os.path.join(
                structure, f"{name[:-4:]}_irregular_signal.wav"
            )

            signal, sample_rate = sf.read(filename)

            disturbance_signal = (
                0.5 * np.max(signal) * disturbance_signal / np.max(disturbance_signal)
            )

            repeats = np.ceil(len(signal) / len(disturbance_signal)).astype(int)
            disturbance_signal = np.tile(disturbance_signal, repeats)
            disturbance_signal = disturbance_signal[: len(signal)]

            signal_modified = signal + disturbance_signal

            sf.write(file=noise_filename, data=signal_modified, samplerate=sample_rate)
