
from pathlib import Path
import random
import argparse
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=Path, 
        help="Download datadir for LibriSpeech"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        help="Output data folder for training"
    )
    parser.add_argument(
        "--dset", 
        type=str, 
        help="Split for LibriSpeech"
    )
    parser.add_argument(
        "--dump_audio_dir", 
        type=Path, 
        help="Dump data folder for training"
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    dset = args.dset
    output_dir = args.output_dir / dset
    dump_audio_dir = args.dump_audio_dir

    dataset = DialogueDataset(task="audio_text_dialogue")
    with (
        open(data_dir / dset / "text", "r") as f,
        open(dump_audio_dir / dset / "wav.scp", "r") as r,
        open(data_dir / "asr_prompt_list", "r") as s,  
    ):

        prompts = s.readlines()
        prompts = [prompt.strip() for prompt in prompts]

        for line, line_r in zip(f, r):
            uttid, text = line.strip().split(" ", 1)
            uttid_1, wav_path = line_r.strip().split(" ", 1)
            assert uttid == uttid_1, f"uttid and uttid_1 are not the same: {uttid} {uttid_1}"
            dialogue = Dialogue(task="audio_text_dialogue")
            
            dialogue.add_segment("user", "speech_owsm_encoder", False, wav_path)
            dialogue.add_segment("user", "text_bpe", False, random.choice(prompts))
            dialogue.add_segment("assistant", "text_bpe", True, text)
            dataset.add_dialogue(line.strip().split(" ")[0], dialogue)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.dump_dataset(output_dir)

if __name__ == "__main__":
    main()
