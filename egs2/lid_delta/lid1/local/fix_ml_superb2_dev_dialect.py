indian_langs = ["tam", "tel", "guj"]

with open(f"data/dev_dialect_ml_superb2_lang/utt2spk", "r") as utt2spk_fp:
    utt2spk_lines = utt2spk_fp.readlines()

filtered_lines = []
for line in utt2spk_lines:
    uttid, lang = line.strip().split()
    if lang == "hin":
        for hin_lang in indian_langs:
            if hin_lang in uttid:
                lang = hin_lang
                break
        else:
            raise ValueError(f"{indian_langs} not in {uttid}")
    filtered_lines.append(f"{uttid} {lang}\n")

with open(f"data/dev_dialect_ml_superb2_lang/utt2spk", "w") as utt2spk_fp:
    utt2spk_fp.writelines(sorted(filtered_lines))
    
