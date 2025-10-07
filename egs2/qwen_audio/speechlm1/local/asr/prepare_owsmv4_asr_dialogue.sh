#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=3
stop_stage=100

# The original data dir structure is like this:
# data/
#  ├── librimix2_train
#  ├── librimix2_dev
#  └── librimix2_test
# it's data/split two level structure, rather than data/set/split three level structure
# ‼️‼️ We use one level name, not set_name/split_name,
# but both the set_name and split_name are represented as one level name
splits="owsmv4_dev owsmv4_train" 

data_dir="data"
dumpdir="dump" # this is for dump dialogue data
dumpdir_audio="dump_audio" # this is for dump audio data
dumpdir_kaldi="dump_kaldi" # sometimes we need to modify the kaldi data, e.g. filter out utts with lenghty audio

# ‼️‼️ Configure for each dataset according to its task
# ‼️‼️ Please do not contain wav.scp, here only need other necessary kaldi files
# ‼️‼️ Please do not add spk2* files, because when construct dialogue, we seldomly use it
#  when use it, the filtering on utt2spk will not have a impact on the effectiveness
# on spk2utt
kaldi_files="text.ctc"

nj=16
audio_format="flac.ark" # flac无损压缩，相比于wav.ark, 节省空间
fs=16000
min_wav_duration=0.1       # Minimum duration in second.
max_wav_duration=30        # Restrict to 30s as OWSM encoder only supports 30s

llm_token_list="TOBEADDED" # to be added after determine specific model 
# NOTE: must add it to the vocabularies of each data.json before merge, and run stage 6

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

_min_length=$(python3 -c "print(int(${min_wav_duration} * ${fs}))")
_max_length=$(python3 -c "print(int(${max_wav_duration} * ${fs}))")

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    :
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    :
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Prepare Dialogue Data"
    for split in ${splits}; do
        log "Processing ${split}"
        python local/asr/prepare_owsmv4_asr_dialogue.py \
            --data_dir ${data_dir} \
            --dump_dir ${dumpdir} \
            --dump_audio_dir ${dumpdir_audio} \
            --dump_kaldi_dir ${dumpdir_kaldi} \
            --split ${split} \
            --prompt_json asr_prompt.json \
        
        cp ${dumpdir}/raw_audio_text_dialogue_${split}_en_asr/data/dialogue.1 ${dumpdir}/raw_audio_text_dialogue_${split}_en_asr/dialogue

        python pyscripts/utils/make_speechlm_json.py \
            --output_json ${dumpdir}/raw_audio_text_dialogue_${split}_en_asr/data.json \
            --task audio_text_dialogue \
            --file_modality_type ${dumpdir}/raw_audio_text_dialogue_${split}_en_asr/dialogue,dialogue,dialogue_json \
            --token_list ${llm_token_list}
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
