from pathlib import Path
import random
import argparse
import json
import os
import re
import iso639
from tqdm import tqdm
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
    
    prompt_list_without_source_lang = []
    prompt_list_with_source_lang = []
    for key, value in prompt_json["without_source_lang"].items():
        prompt_list_without_source_lang.extend(value)
    for key, value in prompt_json["with_source_lang"].items():
        prompt_list_with_source_lang.extend(value)
    
    return prompt_list_without_source_lang, prompt_list_with_source_lang

def clean_timestamp(text):
    """
    Remove all tags and timestamps from text, keeping only the actual content.
    
    Input example: <eng><st_deu><0.00> Also drei sehr schwierige Probleme...<6.12><6.80> Die letzten drei...
    Output example: Also drei sehr schwierige Probleme... Die letzten drei...
    """
    # Remove all tags like <eng>, <st_deu>, etc.
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove timestamps like <0.00>, <6.12>, etc. (numbers with dots)
    text = re.sub(r'<[\d.]+\d+>', '', text)
    
    # Clean up multiple spaces and strip whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    

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
    prompt_list_without_source_lang, prompt_list_with_source_lang = read_prompt_json(prompt_json_dir)

    total_lines = os.system(f"wc -l {dump_kaldi_dir / split / 'text.ctc'}")

    # Currently, first support the language smollm3 support
    support_lang = [
        "por", "ita", "deu", "spa", "fra", "eng", "zho", "ara", "rus"
    ]

    random.seed(2025)

    # NOTE: for st, the original language text is in text.ctc, the target language
    # text is in text
    with (
        open(dump_kaldi_dir / split / "text.ctc", "r") as text_ctc_file,
        open(dump_kaldi_dir / split / "text", "r") as text_file,
        open(dump_audio_dir / f"audio_raw_audio_text_dialogue_{split}" / "wav.scp", "r") as dump_audio_file,
    ):

        dataset = DialogueDataset(task="audio_text_dialogue")
        
        for line_text_ctc, line_text, line_dump_audio in tqdm(zip(text_ctc_file, text_file, dump_audio_file), total=total_lines):
            uttid_text_ctc, text_ctc = line_text_ctc.strip().split(" ", 1)
            uttid_text, text = line_text.strip().split(" ", 1)
            uttid, wav_path = line_dump_audio.strip().split(" ", 1)
            assert uttid_text_ctc == uttid and uttid_text == uttid, f"uttid and uttid_text_ctc are not the same: {uttid} {uttid_text_ctc} {uttid_text}"

            # text is like <task><lang><0.00> dddd
            # parse the text to get the task, lang, and text
            pattern = r'<([^>]+)><([^>]+)><([^>]+)>\s*(.*)'
            match = re.match(pattern, text)
            if match:
                lang_origin, task, timestamp, actual_text = match.groups()
                # task: the task type
                # lang: the language code  
                # timestamp: the timestamp (like 0.00)
                # actual_text: the remaining text content
            else:
                # fallback if pattern doesn't match
                lang_origin, task, timestamp, actual_text = None, None, None, text
            
            assert lang_origin is not None, f"lang_origin is None: {uttid} {text}"

            lang_origin_name = iso639.Language.from_part3(lang_origin).name

            if "st" in task:
                clean_text = clean_timestamp(text)
                
                lang_target = task.replace("st_", "")
                # if both lang_target and lang_origin are not in support_lang, skip
                # because, the smollm3 does not corresponding language's knowledge
                # so the translation cannot be done
                if lang_target not in support_lang and lang_origin not in support_lang:
                    continue

                lang_target_name = iso639.Language.from_part3(lang_target).name

                dialogue = Dialogue(task="audio_text_dialogue")
                assistant_text = f"{clean_text}"

                if random.random() < 0.5:
                    prompt_list = prompt_list_without_source_lang
                    prompt_template = random.choice(prompt_list)
                    prompt = prompt_template.format(target_lang=lang_target_name)
                else:
                    prompt_list = prompt_list_with_source_lang
                    prompt_template = random.choice(prompt_list)
                    prompt = prompt_template.format(source_lang=lang_origin_name, target_lang=lang_target_name)
                
                dialogue.add_segment("user", "speech_owsm_encoder", False, wav_path)
                dialogue.add_segment("user", "text_bpe", False, prompt)
                dialogue.add_segment("assistant", "text_bpe", True, assistant_text)
                dataset.add_dialogue(uttid, dialogue)

    dump_dialogue_dir = dump_dir / f"raw_audio_text_dialogue_{split}_st"
    dump_dialogue_dir.mkdir(parents=True, exist_ok=True)
    dataset.dump_dataset(dump_dialogue_dir)

if __name__ == "__main__":
    main()
