

text_all = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/data/test_librispeech/text"

uttid_all = set()
with open(text_all, "r") as f:
    lines = f.readlines()
    for line in lines:
        uttid = line.split()[0]
        uttid_all.add(uttid)

text_all_root = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/dump/raw/test_librispeech"
text_all_files = [
    "audio_format",
    "feats_type",
    "spk2utt",
    "text",
    "text.ctc",
    "text.prev",
    "utt2num_samples",
    "utt2spk",
    "wav.scp",
]

text_clean = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/data/test_librispeech_clean/text"
uttid_clean = set()
with open(text_clean, "r") as f:
    lines = f.readlines()
    for line in lines:
        uttid = line.split()[0]
        uttid_clean.add(uttid)

text_other = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/data/test_librispeech_other/text"
uttid_other = set()
with open(text_other, "r") as f:
    lines = f.readlines()
    for line in lines:
        uttid = line.split()[0]
        uttid_other.add(uttid)

assert uttid_all == (uttid_clean | uttid_other), "Mismatch in utterance IDs between all, clean, and other sets."

test_clean_path = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/dump/raw/test_librispeech_clean"
test_other_path = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/dump/raw/test_librispeech_other"

for file in text_all_files:
    with open(f"{text_all_root}/{file}", "r") as f:
        lines = f.readlines()
        if len(lines) <= 1:
            with open(f"{test_clean_path}/{file}", "w") as f_clean, \
                 open(f"{test_other_path}/{file}", "w") as f_other:
                for line in lines:
                    f_clean.write(line)
                    f_other.write(line)
        else:
            for line in lines:
                uttid = line.split()[0]
                if uttid in uttid_clean:
                    with open(f"{test_clean_path}/{file}", "a") as f_clean:
                        f_clean.write(line)
                elif uttid in uttid_other:
                    with open(f"{test_other_path}/{file}", "a") as f_other:
                        f_other.write(line)
                else:
                    print(f"Warning: {uttid} not found in clean or other set.")

