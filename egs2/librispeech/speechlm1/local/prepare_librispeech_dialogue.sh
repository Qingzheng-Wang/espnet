#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100

data_dir="data"
dset="train_clean_100"
dumpdir="dump"
output_dir="dump/raw_audio_text_dialogue_librispeech"

nj=16
audio_format="flac.ark"
fs=16000
min_wav_duration=0.1       # Minimum duration in second.
max_wav_duration=120       # Maximum duration in second.

llm_token_list="/work/nvme/bbjs/qwang20/espnet_speechlm3_jinchuan/egs2/librispeech/speechlm1/data/token_list/smollm_1.7b_vocab/token_list"

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

data_audio="${dumpdir}/audio_raw_audio_text_dialogue_librispeech"
_min_length=$(python3 -c "print(int(${min_wav_duration} * ${fs}))")
_max_length=$(python3 -c "print(int(${max_wav_duration} * ${fs}))")

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # first format all audio files
    _opts=
    if [ -e "${data_dir}"/"${dset}"/segments ]; then
        _opts+="--segments "${data_dir}"/${dset}/segments "
    fi

    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
        --audio-format "${audio_format}" --fs "${fs}" \
        --out_filename wav.scp ${_opts} \
        "data/${dset}/wav.scp" "${data_audio}/${dset}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Filter Length
    # Only filter non-test sets; check if "test" is not a substring of $dset
    if [[ "${dset}" != *test* ]]; then
        awk -v min_len="${_min_length}" -v max_len="${_max_length}" '
        FNR==NR { lengths[$1]=$2; next }
        ($1 in lengths) && (lengths[$1] >= min_len) && (lengths[$1] <= max_len) { print $0 }
        ' ${data_audio}/${dset}/utt2num_samples ${data_audio}/${dset}/wav.scp \
        > ${data_audio}/${dset}/wav.scp.tmp
        mv ${data_audio}/${dset}/wav.scp.tmp ${data_audio}/${dset}/wav.scp
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python local/prepare_librispeech_dialogue.py \
        --data_dir ${data_dir} \
        --output_dir ${output_dir} \
        --dset ${dset} \
        --dump_audio_dir ${data_audio}

    # 这里其实应该是merge不同rank的dialogue.rank文件到dialogue，但是
    # 现在暂时不distributed, 所以暂时不考虑，先简单复制
    cp ${output_dir}/${dset}/data/dialogue.1 ${output_dir}/${dset}/dialogue

    python pyscripts/utils/make_speechlm_json.py \
        --output_json ${output_dir}/${dset}/data.json \
        --task audio_text_dialogue \
        --file_modality_type ${output_dir}/${dset}/dialogue,dialogue,dialogue_json \
        --token_list ${llm_token_list}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
