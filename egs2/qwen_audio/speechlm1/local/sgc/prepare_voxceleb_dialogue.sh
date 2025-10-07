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
stop_stage=3

# The original data dir structure is like this:
# data/
#  ├── librimix2_train
#  ├── librimix2_dev
#  └── librimix2_test
# it's data/split two level structure, rather than data/set/split three level structure
# 理论上每生成一个新脚本，只要改这里就行了
# ONLY voxceleb1 has gender information
splits="voxceleb1_test voxceleb1_dev" 

data_dir="data"
dumpdir="dump" # this is for dump dialogue data
dumpdir_audio="dump_audio" # this is for dump audio data
dumpdir_kaldi="dump_kaldi" # sometimes we need to modify the kaldi data, e.g. filter out utts with lenghty audio

# ‼️‼️ Configure for each dataset according to its task
# ‼️‼️ Please do not contain wav.scp, here only need other necessary kaldi files
# ‼️‼️ Please do not add spk2* files, because when construct dialogue, we seldomly use it
#  when use it, the filtering on utt2spk will not have a impact on the effectiveness
# on spk2utt
kaldi_files="utt2spk"

vox1_meta_file="/work/hdd/bbjs/shared/corpora/voxcelebs/voxceleb1/vox1_meta.csv"

nj=16
audio_format="flac.ark" # flac.ark saves more space than wav.ark
fs=16000
min_wav_duration=0.1       # Minimum duration in second.
max_wav_duration=30        #‼️‼️ Restrict to 30s as OWSM encoder only supports 30s

llm_token_list="TOBEADDED" # to be added after determine specific model 
# NOTE: must add it to the vocabularies of each data.json before merge, and run stage 6

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

_min_length=$(python3 -c "print(int(${min_wav_duration} * ${fs}))")
_max_length=$(python3 -c "print(int(${max_wav_duration} * ${fs}))")

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: VoxCeleb Data Preparation"
    bash local/sgc/voxceleb_spk1_local/data.sh
    python local/sgc/voxceleb_spk1_local/generate_speaker_info.py \
        --data_dir ${data_dir} \
        --vox1_meta_file "${vox1_meta_file}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Format Wav Scp"
    for split in ${splits}; do
        log "Processing ${split}"
        output_audio_dir="${dumpdir_audio}/audio_raw_audio_text_dialogue_${split}"
        _opts=
        if [ -e "${data_dir}"/"${split}"/segments ]; then
            _opts+="--segments "${data_dir}"/${split}/segments "
        fi

        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" \
            --out_filename wav.scp ${_opts} \
            "data/${split}/wav.scp" "${output_audio_dir}"
    done

    # Function to filter kaldi files by audio length
    filter_kaldi_file_by_length() {
        local utt2num_samples_file="$1"
        local input_file="$2"
        local min_length="$3"
        local max_length="$4"
        local output_file="$5"
        local filename="$6"
        
        log "Filtering ${filename}: min_length=${min_length}, max_length=${max_length}"
        
        # Check if input file exists
        if [ ! -f "${input_file}" ]; then
            log "Warning: ${filename} not found at ${input_file}, skipping..."
            return 0
        fi
        
        awk -v min_len="${min_length}" -v max_len="${max_length}" '
        FNR==NR { lengths[$1]=$2; next }
        ($1 in lengths) && (lengths[$1] >= min_len) && (lengths[$1] <= max_len) { print $0 }
        ' "${utt2num_samples_file}" "${input_file}" > "${output_file}"
    }

    # Filter Length
    # Only filter non-test sets; check if "test" is not a substring of $split
    for split in ${splits}; do
        output_audio_dir="${dumpdir_audio}/audio_raw_audio_text_dialogue_${split}"
        output_kaldi_dir="${dumpdir_kaldi}/${split}"
        if [[ "${split}" != *test* ]]; then
            mkdir -p ${output_kaldi_dir}
            # Filter wav.scp
            filter_kaldi_file_by_length \
                "${output_audio_dir}/utt2num_samples" \
                "${output_audio_dir}/wav.scp" \
                "${_min_length}" \
                "${_max_length}" \
                "${output_audio_dir}/wav.scp.tmp" \
                "wav.scp"
            
            mv ${output_audio_dir}/wav.scp.tmp ${output_audio_dir}/wav.scp
            cp ${output_audio_dir}/wav.scp ${output_kaldi_dir}/wav.scp
            
            # Filter other kaldi files
            mkdir -p ${output_kaldi_dir}
            for kaldi_file in ${kaldi_files}; do
                if [ -f "${data_dir}/${split}/${kaldi_file}" ]; then
                    filter_kaldi_file_by_length \
                        "${output_audio_dir}/utt2num_samples" \
                        "${data_dir}/${split}/${kaldi_file}" \
                        "${_min_length}" \
                        "${_max_length}" \
                        "${output_kaldi_dir}/${kaldi_file}.tmp" \
                        "${kaldi_file}"
                    
                    mv "${output_kaldi_dir}/${kaldi_file}.tmp" "${output_kaldi_dir}/${kaldi_file}"
                fi
            done

            cp "${output_audio_dir}/utt2num_samples" "${output_audio_dir}/utt2num_samples.tmp"
            filter_kaldi_file_by_length \
                "${output_audio_dir}/utt2num_samples.tmp" \
                "${output_audio_dir}/utt2num_samples" \
                "${_min_length}" \
                "${_max_length}" \
                "${output_audio_dir}/utt2num_samples.filtered" \
                "utt2num_samples"
            rm "${output_audio_dir}/utt2num_samples.tmp"
            mv "${output_audio_dir}/utt2num_samples" "${output_audio_dir}/utt2num_samples_full"
            mv "${output_audio_dir}/utt2num_samples.filtered" "${output_audio_dir}/utt2num_samples"
            cp "${output_audio_dir}/utt2num_samples" "${output_kaldi_dir}/utt2num_samples"
        else
            mkdir -p ${output_kaldi_dir}
            cp ${data_dir}/${split}/wav.scp ${output_kaldi_dir}/wav.scp
            for kaldi_file in ${kaldi_files}; do
                cp ${data_dir}/${split}/${kaldi_file} ${output_kaldi_dir}/${kaldi_file}
            done
            cp "${output_audio_dir}/utt2num_samples" "${output_kaldi_dir}/utt2num_samples"
        fi

    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Prepare Dialogue Data"
    for split in ${splits}; do
        log "Processing ${split}"
        python local/sgc/prepare_voxceleb_dialogue.py \
            --data_dir ${data_dir} \
            --dump_dir ${dumpdir} \
            --dump_audio_dir ${dumpdir_audio} \
            --split ${split} \
            --prompt_json sgc_prompt.json
        
        cp ${dumpdir}/raw_audio_text_dialogue_${split}/data/dialogue.1 ${dumpdir}/raw_audio_text_dialogue_${split}/dialogue

        python pyscripts/utils/make_speechlm_json.py \
            --output_json ${dumpdir}/raw_audio_text_dialogue_${split}/data.json \
            --task audio_text_dialogue \
            --file_modality_type ${dumpdir}/raw_audio_text_dialogue_${split}/dialogue,dialogue,dialogue_json \
            --token_list ${llm_token_list}
    done

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
