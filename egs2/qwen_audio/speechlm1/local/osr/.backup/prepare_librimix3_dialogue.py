from pathlib import Path
import random
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        "--split",
        type=str, # a default value should be like datasetname_split
        help="The corresponding train, dev, test name of the dataset"
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        help="The name of the prompt json file" # like asr_prompt.json
    )
    parser.add_argument(
        "--use_punctuation_restoration",
        action="store_true",
        help="Whether to use Qwen3 model for punctuation and case restoration"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Hugging Face model name for punctuation restoration"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the model on (auto, cpu, cuda, cuda:0, etc.)"
    )

    return parser

def read_prompt_json(prompt_json_dir):
    with open(prompt_json_dir, "r") as f:
        prompt_json = json.load(f)
    
    prompt_list = []
    for key, value in prompt_json.items():
        prompt_list.extend(value)
    
    return prompt_list

class PunctuationRestorer:
    def __init__(self, model_name, device="auto"):
        """
        Initialize the punctuation restoration model.
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device if self.device != "auto" else None,
                trust_remote_code=True
            )
            
            if self.device == "auto" and torch.cuda.is_available():
                self.model = self.model.cuda()
            elif self.device != "auto":
                self.model = self.model.to(self.device)
                
            print(f"Model loaded successfully on device: {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def restore_punctuation_and_case(self, text, max_retries=3):
        """
        Use Qwen3 model to restore punctuation and case sensitivity.
        
        Args:
            text (str): Input text without proper punctuation and case
            max_retries (int): Maximum number of retry attempts
        
        Returns:
            str: Text with restored punctuation and case, or original text if model fails
        """
        if self.model is None or self.tokenizer is None:
            print("Model not loaded, returning original text")
            return text
            
        prompt = f"For the given English sentence, restore the upper-case characters (if applicable) and add punctuation WITHOUT CHANGING ANY WORDS. Answer in English without any explanation. Here is the sentence: {text}. Here is the output:"
        
        for attempt in range(max_retries):
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                
                # Move to same device as model
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,  # Increased from 100
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Add repetition penalty
                        no_repeat_ngram_size=3  # Prevent repeating n-grams
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the restored text more carefully
                if "Here is the output:" in response:
                    # Split by "Here is the output:" and take everything after it
                    parts = response.split("Here is the output:")
                    if len(parts) > 1:
                        restored_text = parts[-1].strip()
                    else:
                        restored_text = text
                else:
                    # If the format is different, try to extract the last meaningful line
                    lines = response.strip().split('\n')
                    # Find the last line that doesn't contain the prompt
                    restored_text = text
                    for line in reversed(lines):
                        line = line.strip()
                        if line and not any(keyword in line.lower() for keyword in ['human:', 'here is the sentence:', 'here is the output:', 'for the given english']):
                            restored_text = line
                            break
                
                # Clean up the text - remove any remaining prompt fragments
                restored_text = restored_text.replace(prompt, "").strip()
                
                # Remove any remaining prompt-like text
                prompt_keywords = ['Human:', 'Here is the sentence:', 'Here is the output:', 'For the given English sentence']
                for keyword in prompt_keywords:
                    if keyword in restored_text:
                        restored_text = restored_text.split(keyword)[0].strip()
                
                # Validate the result
                if restored_text and restored_text != text and len(restored_text) > len(text) * 0.5:
                    print(f"Original text: {text}")
                    print(f"Restored text: {restored_text}")
                    return restored_text
                else:
                    print(f"Warning: Model returned invalid text, using original")
                    return text
                    
            except Exception as e:
                print(f"Model inference failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"All retry attempts failed. Using original text: {text}")
                    return text
        
        return text

def main():
    parser = get_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    dump_dir = args.dump_dir
    dump_audio_dir = args.dump_audio_dir
    split = args.split
    prompt_json = args.prompt_json
    use_punctuation_restoration = args.use_punctuation_restoration
    model_name = args.model_name
    device = args.device

    prompt_json_dir = data_dir / "prompts" / prompt_json
    prompt_list = read_prompt_json(prompt_json_dir)

    # Initialize punctuation restorer if needed
    restorer = None
    if use_punctuation_restoration:
        try:
            restorer = PunctuationRestorer(model_name, device)
            print("Punctuation restoration model loaded successfully")
        except Exception as e:
            print(f"Failed to load punctuation restoration model: {e}")
            print("Continuing without punctuation restoration...")
            use_punctuation_restoration = False

    dataset = DialogueDataset(task="audio_text_dialogue")
    with (
        open(data_dir / split / "text_spk1", "r") as text_spk1_file,
        open(data_dir / split / "text_spk2", "r") as text_spk2_file,
        open(data_dir / split / "text_spk3", "r") as text_spk3_file,  # Added third speaker text file
        open(dump_audio_dir / f"audio_raw_audio_text_dialogue_{split}" / "wav.scp", "r") as dump_audio_file,
    ):

        for line_text_spk1, line_text_spk2, line_text_spk3, line_dump_audio in zip(text_spk1_file, text_spk2_file, text_spk3_file, dump_audio_file):
            # Parse text_spk1
            parts_spk1 = line_text_spk1.strip().split(" ", 1)
            if len(parts_spk1) != 2:
                print(f"Warning: Invalid format in text_spk1: {line_text_spk1.strip()}")
                continue
            uttid_spk1, text_spk1 = parts_spk1
            
            # Parse text_spk2
            parts_spk2 = line_text_spk2.strip().split(" ", 1)
            if len(parts_spk2) != 2:
                print(f"Warning: Invalid format in text_spk2: {line_text_spk2.strip()}")
                continue
            uttid_spk2, text_spk2 = parts_spk2
            
            # Parse text_spk3
            parts_spk3 = line_text_spk3.strip().split(" ", 1)
            if len(parts_spk3) != 2:
                print(f"Warning: Invalid format in text_spk3: {line_text_spk3.strip()}")
                continue
            uttid_spk3, text_spk3 = parts_spk3
            
            # Parse dump_audio
            parts_dump = line_dump_audio.strip().split(" ", 1)
            if len(parts_dump) != 2:
                print(f"Warning: Invalid format in dump_audio: {line_dump_audio.strip()}")
                continue
            uttid, wav_path = parts_dump
            
            assert uttid_spk1 == uttid and uttid_spk2 == uttid and uttid_spk3 == uttid, f"uttid and uttid_spk1, uttid_spk2, uttid_spk3 are not the same: {uttid} {uttid_spk1} {uttid_spk2} {uttid_spk3}"

            # Apply punctuation and case restoration if enabled
            if use_punctuation_restoration and restorer is not None:
                print(f"Restoring punctuation for utterance: {uttid}")
                text_spk1 = restorer.restore_punctuation_and_case(text_spk1)
                text_spk2 = restorer.restore_punctuation_and_case(text_spk2)
                text_spk3 = restorer.restore_punctuation_and_case(text_spk3)

            dialogue = Dialogue(task="audio_text_dialogue")
            assistant_text = f"Speaker 1: {text_spk1} Speaker 2: {text_spk2} Speaker 3: {text_spk3}"  # Updated to include third speaker
            
            dialogue.add_segment("user", "speech_owsm_encoder", False, wav_path)
            dialogue.add_segment("user", "text_bpe", False, random.choice(prompt_list))
            dialogue.add_segment("assistant", "text_bpe", True, assistant_text)
            dataset.add_dialogue(uttid, dialogue)

    dump_dialogue_dir = dump_dir / f"raw_audio_text_dialogue_{split}"
    dump_dialogue_dir.mkdir(parents=True, exist_ok=True)
    dataset.dump_dataset(dump_dialogue_dir)

if __name__ == "__main__":
    main()