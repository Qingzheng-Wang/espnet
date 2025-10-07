from pathlib import Path
import random
import argparse
import json
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=Path,
        default="data",
        help="The data dir that train, dev, test is in"
    )
    parser.add_argument(
        "--dump_dir",
        type=Path,
        default="dump",
        help="Dump dir for dialogue data"
    )
    parser.add_argument(
        "--dump_audio_dir", 
        type=Path, 
        default="dump_audio",
        help="In the sh file, we first dump audio to the dump_audio_dir in flac.ark format"
    ) # when generating dialogue data, we use the dumped flac.ark audio path
    parser.add_argument(
        "--dump_kaldi_dir",
        type=Path,
        default="dump_kaldi",
        help="Dump dir for kaldi data"
    )
    parser.add_argument(
        "--split",
        type=str, # a default value should be like datasetname_split
        help="The corresponding train, dev, test name of the dataset"
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        help="The name of the prompt json file" # like asr_prompt.json
    )

    return parser

def read_prompt_json(prompt_json_dir):
    with open(prompt_json_dir, "r") as f:
        prompt_json = json.load(f)
    
    prompt_list = []
    for key, value in prompt_json.items():
        prompt_list.extend(value)
    
    return prompt_list

def main():
    parser = get_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    dump_dir = args.dump_dir
    dump_audio_dir = args.dump_audio_dir
    dump_kaldi_dir = args.dump_kaldi_dir
    split = args.split
    prompt_json = args.prompt_json

    prompt_json_dir = data_dir / "prompts" / prompt_json
    prompt_list = read_prompt_json(prompt_json_dir)

    with (
        open(dump_kaldi_dir / split / "utt2spk", "r") as utt2spk_file,
        open(data_dir / split / "spk2gender", "r") as spk2gender_file,
        open(dump_audio_dir / f"audio_raw_audio_text_dialogue_{split}" / "wav.scp", "r") as dump_audio_file,
    ):
        spk2gender = {}
        for line_spk2gender in spk2gender_file:
            spk, gender = line_spk2gender.strip().split(" ", 1)
            spk2gender[spk] = gender

        dataset = DialogueDataset(task="audio_text_dialogue")
        
        for line_utt2spk, line_dump_audio in zip(utt2spk_file, dump_audio_file):
            uttid_spk, spk = line_utt2spk.strip().split(" ", 1)
            uttid, wav_path = line_dump_audio.strip().split(" ", 1)
            assert uttid_spk == uttid, f"uttid and uttid_spk are not the same: {uttid} {uttid_spk}"

            # Check if speaker gender is available, skip if not
            if spk not in spk2gender:
                print(f"Warning: Speaker {spk} not found in spk2gender, skipping utterance {uttid}")
                continue
                
            gender = spk2gender[spk]

            if gender == 'Unknown':
                continue

            dialogue = Dialogue(task="audio_text_dialogue")
            assistant_text = f"{gender}"
            
            dialogue.add_segment("user", "speech_owsm_encoder", False, wav_path)
            dialogue.add_segment("user", "text_bpe", False, random.choice(prompt_list))
            dialogue.add_segment("assistant", "text_bpe", True, assistant_text)
            dataset.add_dialogue(uttid, dialogue)

    dump_dialogue_dir = dump_dir / f"raw_audio_text_dialogue_{split}"
    dump_dialogue_dir.mkdir(parents=True, exist_ok=True)
    dataset.dump_dataset(dump_dialogue_dir)

if __name__ == "__main__":
    main()
