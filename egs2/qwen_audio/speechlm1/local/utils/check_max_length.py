
utt2num_samples_path = "/work/nvme/bbjs/qwang20/espnet_speechlm3_jinchuan/egs2/qwen_audio/speechlm1/dump_audio/audio_raw_audio_text_dialogue_voxceleb1_dev/utt2num_samples_full"

max_num_samples = 0
with open(utt2num_samples_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                utterance_id = parts[0]
                num_samples = int(parts[1])
                if num_samples > max_num_samples:
                    max_num_samples = num_samples

print(max_num_samples / 16000)

