from .audio_loader import ArkiveAudioReader, LhotseAudioReader, SoundfileReader
from .dialogue_loader import DialogueReader
from .interleave_loader import InterleaveReader
from .text_loader import ArkiveTextReader, TextReader

ALL_DATA_LOADERS = {
    "lhotse_audio": LhotseAudioReader,
    "arkive_audio": ArkiveAudioReader,
    "soundfile": SoundfileReader,
    "text": TextReader,
    "arkive_text": ArkiveTextReader,
    "dialogue": DialogueReader,
    "interleave": InterleaveReader,
}

__all__ = [
    "ALL_DATA_LOADERS",
]
