from pathlib import Path

from pydub import AudioSegment


def convert_to_wav(input_audio_file: str, output_wav_file: str) -> None:
    input_audio_file_format = Path(input_audio_file).suffix[1:]
    if input_audio_file_format in ["m4a", "mp4"]:
        audio = AudioSegment.from_file(
            file=input_audio_file, format=input_audio_file_format
        )
        audio.export(out_f=output_wav_file, format="wav")
    else:
        raise ValueError(f"Unsupported file format: {input_audio_file_format}")
