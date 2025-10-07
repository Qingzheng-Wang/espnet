from pathlib import Path
import random
import argparse
import json
import torch
import requests
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset
from tqdm import tqdm

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
    parser.add_argument(
        "--use_sglang",
        action="store_true",
        help="Whether to use SGLang API instead of local model"
    )
    parser.add_argument(
        "--sglang_url",
        type=str,
        default="http://localhost:30000",
        help="SGLang server URL for API calls"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for SGLang API calls (only effective with --use_sglang)"
    )

    return parser

def read_prompt_json(prompt_json_dir):
    with open(prompt_json_dir, "r") as f:
        prompt_json = json.load(f)
    
    prompt_list = []
    for key, value in prompt_json.items():
        prompt_list.extend(value)
    
    return prompt_list



# TODO: SGLang currently cannot handle no_thinking, to be fixed

class SGLangClient:
    def __init__(self, sglang_url):
        """
        Initialize SGLang client.
        
        Args:
            sglang_url (str): SGLang server URL
        """
        self.sglang_url = sglang_url.rstrip('/')
        self.generate_url = f"{self.sglang_url}/v1/chat/completions"
    
    def generate(self, messages, max_new_tokens=200, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0):
        """
        Generate text using SGLang API.
        
        Args:
            messages (list): List of message dictionaries
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            top_k (int): Top-k sampling parameter
            min_p (float): Min-p sampling parameter
            
        Returns:
            str: Generated text
        """
        payload = {
            "model": "default",  # SGLang uses "default" as model name
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "stream": False,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}  # Always disable thinking
            }
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # 60 second timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"SGLang API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"SGLang API response parsing failed: {e}")
    
    def generate_batch(self, messages_list, max_new_tokens=200, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0):
        """
        Generate text for multiple messages using SGLang API.
        
        Args:
            messages_list (list): List of message lists
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            top_k (int): Top-k sampling parameter
            min_p (float): Min-p sampling parameter
            
        Returns:
            list: List of generated texts
        """
        responses = []
        
        for messages in messages_list:
            try:
                response = self.generate(messages, max_new_tokens, temperature, top_p, top_k, min_p)
                responses.append(response)
            except Exception as e:
                print(f"Batch generation failed for one item: {e}")
                # Return original text or empty string as fallback
                responses.append("")
        
        return responses

class PunctuationRestorer:
    def __init__(self, model_name, device="auto", use_sglang=False, sglang_url=None):
        """
        Initialize the punctuation restoration model.
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Device to run the model on
            use_sglang (bool): Whether to use SGLang API
            sglang_url (str): SGLang server URL if using SGLang
        """
        self.model_name = model_name
        self.device = device
        self.use_sglang = use_sglang
        self.sglang_url = sglang_url
        self.tokenizer = None
        self.model = None
        self.sglang_client = None
        
        if self.use_sglang:
            self._load_sglang_client()
        else:
            self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            if self.device == "auto" and torch.cuda.is_available():
                self.model = self.model.cuda()
            elif self.device != "auto":
                self.model = self.model.to(self.device)
                
            print(f"Model loaded successfully on device: {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def _load_sglang_client(self):
        """Load SGLang client."""
        try:
            print(f"Connecting to SGLang server: {self.sglang_url}")
            self.sglang_client = SGLangClient(self.sglang_url)
            # Test connection
            test_messages = [{"role": "user", "content": "Hello"}]
            self.sglang_client.generate(test_messages, max_new_tokens=5)
            print("SGLang client connected successfully")
        except Exception as e:
            print(f"Error connecting to SGLang server: {e}")
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
        if self.use_sglang:
            if self.sglang_client is None:
                print("SGLang client not loaded, returning original text")
                return text
        else:
            if self.model is None or self.tokenizer is None:
                print("Model not loaded, returning original text")
                return text
            
        prompt = f"""You are a text restoration assistant. Your task is to restore proper capitalization and punctuation to the given sentence. Keep all words unchanged. Output only the corrected sentence with no explanation.

                Examples

                Input:
                BUT THE DISTINCTION OF DAYS AND YEARS HAVING DEPENDED ON THE MOTION OF THE SUN IT HAS BROUGHT THIS MISTAKE WITH IT THAT IT HAS BEEN THOUGHT THAT MOTION AND DURATION WERE THE MEASURE ONE OF ANOTHER

                Output:
                But the distinction of days and years, having depended on the motion of the sun, it has brought this mistake with it, that it has been thought that motion and duration were the measure one of another.

                ⸻

                Input:
                WITH STRONGER THEMES PRACTICAL PEACEFUL LIFE THE PEOPLE'S LIFE THE PEOPLE THEMSELVES LIFTED ILLUMIN’D BATHED IN PEACE ELATE SECURE IN PEACE

                Output:
                With stronger themes, practical, peaceful life, the people's life, the people themselves lifted, illumin’d, bathed in peace, elate, secure in peace.

                ⸻

                Input:
                AND THAT GIVES US AN HOUR WHILE MOTHER GETS BREAKFAST YOU KNOW BOYS AND GIRLS OF TODAY MAY PITY THOSE POOR PIONEER CHILDREN WHO HAD TO GET UP SO EARLY

                Output:
                And that gives us an hour, while mother gets breakfast. You know, boys and girls of today may pity those poor pioneer children who had to get up so early.

                ⸻

                Now continue:

                Input:
                {text}

                Output:
                """
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(max_retries):
            try:
                if self.use_sglang:
                    # Use SGLang API
                    response = self.sglang_client.generate(
                        messages,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        min_p=0.0
                    )
                    return response
                else:
                    # Use local model
                    # Tokenize input
                    text_input = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=200,  # Increased from 100
                            temperature=0.7,
                            top_p=0.8,
                            top_k=20,
                            min_p=0.0,
                        )
                    
                    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist() 
                    
                    # Decode response
                    response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

                    return response
                    
            except Exception as e:
                print(f"Model inference failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"All retry attempts failed. Using original text: {text}")
                    return text
        
        return text
    
    def restore_punctuation_and_case_batch(self, texts, max_retries=3):
        """
        Restore punctuation and case for multiple texts in batch.
        
        Args:
            texts (list): List of input texts
            max_retries (int): Maximum number of retry attempts
        
        Returns:
            list: List of restored texts
        """
        if not texts:
            return []
            
        if self.use_sglang:
            if self.sglang_client is None:
                print("SGLang client not loaded, returning original texts")
                return texts
        else:
            if self.model is None or self.tokenizer is None:
                print("Model not loaded, returning original texts")
                return texts
        
        # Prepare prompts for all texts
        messages_list = []
        for text in texts:
            prompt = f"""You are a text restoration assistant. Your task is to restore proper capitalization and punctuation to the given sentence. Keep all words unchanged. Output only the corrected sentence with no explanation.

                Examples

                Input:
                BUT THE DISTINCTION OF DAYS AND YEARS HAVING DEPENDED ON THE MOTION OF THE SUN IT HAS BROUGHT THIS MISTAKE WITH IT THAT IT HAS BEEN THOUGHT THAT MOTION AND DURATION WERE THE MEASURE ONE OF ANOTHER

                Output:
                But the distinction of days and years, having depended on the motion of the sun, it has brought this mistake with it, that it has been thought that motion and duration were the measure one of another.

                ⸻

                Input:
                WITH STRONGER THEMES PRACTICAL PEACEFUL LIFE THE PEOPLE'S LIFE THE PEOPLE THEMSELVES LIFTED ILLUMIN'D BATHED IN PEACE ELATE SECURE IN PEACE

                Output:
                With stronger themes, practical, peaceful life, the people's life, the people themselves lifted, illumin'd, bathed in peace, elate, secure in peace.

                ⸻

                Now continue:

                Input:
                {text}

                Output:
                """
            messages_list.append([{"role": "user", "content": prompt}])
        
        for attempt in range(max_retries):
            try:
                if self.use_sglang:
                    # Use SGLang API batch processing
                    responses = self.sglang_client.generate_batch(
                        messages_list,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        min_p=0.0
                    )
                    return responses
                else:
                    # For local model, process sequentially (could be optimized later)
                    responses = []
                    for messages in messages_list:
                        text_input = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=200,
                                temperature=0.7,
                                top_p=0.8,
                                top_k=20,
                                min_p=0.0,
                            )
                        
                        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                        responses.append(response)
                    
                    return responses
                    
            except Exception as e:
                print(f"Batch model inference failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"All batch retry attempts failed. Using original texts")
                    return texts
        
        return texts

def main():
    parser = get_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    dump_dir = args.dump_dir
    dump_audio_dir = args.dump_audio_dir
    dump_kaldi_dir = args.dump_kaldi_dir
    split = args.split
    prompt_json = args.prompt_json
    use_punctuation_restoration = args.use_punctuation_restoration
    model_name = args.model_name
    device = args.device
    use_sglang = args.use_sglang
    sglang_url = args.sglang_url
    batch_size = args.batch_size

    prompt_json_dir = data_dir / "prompts" / prompt_json
    prompt_list = read_prompt_json(prompt_json_dir)

    # Create dump directory if it doesn't exist
    dump_text_dir = dump_dir / f"raw_audio_text_dialogue_{split}"
    dump_text_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing files and load processed uttids
    processed_uttids_spk1 = set()
    processed_uttids_spk2 = set()
    processed_uttids = set()
    text_spk1_file_path = dump_text_dir / "text_spk1"
    text_spk2_file_path = dump_text_dir / "text_spk2"
    
    if text_spk1_file_path.exists() and text_spk2_file_path.exists():
        print("Found existing output files. Loading processed uttids for resume...")
        # Load processed uttids from existing files
        with open(text_spk1_file_path, "r") as f:
            for line in f:
                if line.strip():
                    uttid = line.strip().split(" ", 1)[0]
                    processed_uttids_spk1.add(uttid)
        
        with open(text_spk2_file_path, "r") as f:
            for line in f:
                if line.strip():
                    uttid = line.strip().split(" ", 1)[0]
                    processed_uttids_spk2.add(uttid)
                    
        processed_uttids = processed_uttids_spk1.union(processed_uttids_spk2)
        
        print(f"Found {len(processed_uttids)} already processed utterances. Will resume from where left off.")
    else:
        print("No existing output files found. Starting fresh.")
    
    dump_text_spk1_file = open(text_spk1_file_path, "a")
    dump_text_spk2_file = open(text_spk2_file_path, "a")

    processed_uttid2text_spk1 = {}
    processed_uttid2text_spk2 = {}
    with open(text_spk1_file_path, "r") as f:
        for line in f:
            if line.strip():
                uttid, text = line.strip().split(" ", 1)
                processed_uttid2text_spk1[uttid] = text
    with open(text_spk2_file_path, "r") as f:
        for line in f:
            if line.strip():
                uttid, text = line.strip().split(" ", 1)
                processed_uttid2text_spk2[uttid] = text

    # Initialize punctuation restorer if needed
    restorer = None
    if use_punctuation_restoration:
        if use_sglang:
            print(f"Initializing punctuation restorer with SGLang: {sglang_url}")
        else:
            print(f"Initializing punctuation restorer with model: {model_name}")
        try:
            restorer = PunctuationRestorer(
                model_name=model_name, 
                device=device, 
                use_sglang=use_sglang, 
                sglang_url=sglang_url
            )
            if use_sglang:
                print("SGLang punctuation restoration client loaded successfully")
            else:
                print("Punctuation restoration model loaded successfully")
        except Exception as e:
            print(f"Failed to load punctuation restoration: {e}")
            print("Continuing without punctuation restoration...")
            use_punctuation_restoration = False

    dataset = DialogueDataset(task="audio_text_dialogue")
    # First, count total lines for progress bar
    with open(dump_kaldi_dir / split / "text_spk1", "r") as f:
        total_lines = sum(1 for _ in f)

    print(f"Total utterances to process: {total_lines}")

    text_spk1_file = dump_kaldi_dir / split / "text_spk1"
    text_spk2_file = dump_kaldi_dir / split / "text_spk2"

    with (
        open(text_spk1_file, "r") as text_spk1_file,
        open(text_spk2_file, "r") as text_spk2_file,
        open(dump_audio_dir / f"audio_raw_audio_text_dialogue_{split}" / "wav.scp", "r") as dump_audio_file,
    ):

        # Collect data for batch processing
        data_batch = []
        batch_count = 0
        processed_items = 0
        skipped_items = 0
        
        # Create progress bar
        pbar = tqdm(
            total=total_lines,
            desc=f"Processing {split} dialogue data",
            unit="utterance",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for line_text_spk1, line_text_spk2, line_dump_audio in zip(text_spk1_file, text_spk2_file, dump_audio_file):
            uttid_spk1, text_spk1 = line_text_spk1.strip().split(" ", 1)
            uttid_spk2, text_spk2 = line_text_spk2.strip().split(" ", 1)
            uttid, wav_path = line_dump_audio.strip().split(" ", 1)
            assert uttid_spk1 == uttid and uttid_spk2 == uttid, f"uttid and uttid_spk1 and uttid_spk2 are not the same: {uttid} {uttid_spk1} {uttid_spk2}"

            data_batch.append({
                'uttid': uttid,
                'uttid_spk1': uttid_spk1,
                'uttid_spk2': uttid_spk2,
                'text_spk1': text_spk1,
                'text_spk2': text_spk2,
                'wav_path': wav_path
            })
            
            # Process batch when it reaches batch_size or at the end
            if len(data_batch) >= batch_size or (len(data_batch) > 0 and 
                line_text_spk1 == "" and line_text_spk2 == "" and line_dump_audio == ""):
                
                batch_count += 1
                pbar.set_description(f"Processing {split} dialogue data (batch {batch_count})")
                
                # Apply punctuation and case restoration if enabled
                if use_punctuation_restoration and restorer is not None:

                    # Collect all texts for batch processing
                    all_texts_spk1 = [item['text_spk1'] for item in data_batch]
                    all_texts_spk2 = [item['text_spk2'] for item in data_batch]
                    
                    # Process in batch if using SGLang and batch_size > 1
                    if use_sglang and batch_size > 1:
                        restored_texts_spk1 = restorer.restore_punctuation_and_case_batch(all_texts_spk1)
                        restored_texts_spk2 = restorer.restore_punctuation_and_case_batch(all_texts_spk2)
                    else:
                        restored_texts_spk1 = []
                        restored_texts_spk2 = []
                        for item in data_batch:
                            if item["uttid"] in processed_uttids:
                                restored_texts_spk1.append(processed_uttid2text_spk1[item["uttid"]])
                                restored_texts_spk2.append(processed_uttid2text_spk2[item["uttid"]])
                            else:
                                restored_texts_spk1.append(restorer.restore_punctuation_and_case(item["text_spk1"]))
                                restored_texts_spk2.append(restorer.restore_punctuation_and_case(item["text_spk2"]))
                    
                    # Update the batch data with restored texts
                    for i, item in enumerate(data_batch):
                        item['text_spk1'] = restored_texts_spk1[i]
                        item['text_spk2'] = restored_texts_spk2[i]
                        if item["uttid"] not in processed_uttids:
                            dump_text_spk1_file.write(f"{item['uttid_spk1']} {item['text_spk1']}\n")
                            dump_text_spk2_file.write(f"{item['uttid_spk2']} {item['text_spk2']}\n")
                
                # Create dialogues for this batch
                for item in data_batch:
                    dialogue = Dialogue(task="audio_text_dialogue")
                    assistant_text = f"Speaker 1: {item['text_spk1']} Speaker 2: {item['text_spk2']}"
                    
                    dialogue.add_segment("user", "speech_owsm_encoder", False, item['wav_path'])
                    dialogue.add_segment("user", "text_bpe", False, random.choice(prompt_list))
                    dialogue.add_segment("assistant", "text_bpe", True, assistant_text)
                    dataset.add_dialogue(item['uttid'], dialogue)
                
                # Clear the batch and update progress
                processed_items += len(data_batch)
                pbar.update(len(data_batch))
                data_batch = []
        
        # Process any remaining items
        if data_batch:
            batch_count += 1
            pbar.set_description(f"Processing {split} dialogue data (final batch {batch_count})")
            
            if use_punctuation_restoration and restorer is not None:
                all_texts_spk1 = [item['text_spk1'] for item in data_batch]
                all_texts_spk2 = [item['text_spk2'] for item in data_batch]
                
                if use_sglang and batch_size > 1:
                    restored_texts_spk1 = restorer.restore_punctuation_and_case_batch(all_texts_spk1)
                    restored_texts_spk2 = restorer.restore_punctuation_and_case_batch(all_texts_spk2)
                else:
                    restored_texts_spk1 = []
                    restored_texts_spk2 = []
                    for item in data_batch:
                        if item["uttid"] in processed_uttids:
                            restored_texts_spk1.append(processed_uttid2text_spk1[item["uttid"]])
                            restored_texts_spk2.append(processed_uttid2text_spk2[item["uttid"]])
                        else:
                            restored_texts_spk1.append(restorer.restore_punctuation_and_case(item["text_spk1"]))
                            restored_texts_spk2.append(restorer.restore_punctuation_and_case(item["text_spk2"]))

                for i, item in enumerate(data_batch):
                    item['text_spk1'] = restored_texts_spk1[i]
                    item['text_spk2'] = restored_texts_spk2[i]
                    if item["uttid"] not in processed_uttids:
                        dump_text_spk1_file.write(f"{item['uttid_spk1']} {item['text_spk1']}\n")
                        dump_text_spk2_file.write(f"{item['uttid_spk2']} {item['text_spk2']}\n")
            
            for item in data_batch:
                dialogue = Dialogue(task="audio_text_dialogue")
                assistant_text = f"Speaker 1: {item['text_spk1']} Speaker 2: {item['text_spk2']}"
                
                dialogue.add_segment("user", "speech_owsm_encoder", False, item['wav_path'])
                dialogue.add_segment("user", "text_bpe", False, random.choice(prompt_list))
                dialogue.add_segment("assistant", "text_bpe", True, assistant_text)
                dataset.add_dialogue(item['uttid'], dialogue)
            
            # Update progress for final batch
            processed_items += len(data_batch)
            pbar.update(len(data_batch))
        
        # Close progress bar
        pbar.close()

    # Close files
    dump_text_spk1_file.close()
    dump_text_spk2_file.close()
    
    print(f"\nProcessing completed for {split} split")
    print(f"Total dialogues processed: {len(dataset.dialogues)}")
    if skipped_items > 0:
        print(f"Skipped already processed items: {skipped_items}")
    print(f"New items processed: {len(dataset.dialogues) - skipped_items}")
    
    # Save dataset
    dump_dialogue_dir = dump_dir / f"raw_audio_text_dialogue_{split}"
    dump_dialogue_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dumping dataset to: {dump_dialogue_dir}")
    dataset.dump_dataset(dump_dialogue_dir)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    main()
