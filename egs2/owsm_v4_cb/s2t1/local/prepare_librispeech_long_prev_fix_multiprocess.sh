#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Default values
data_dir=downloads/LibriSpeech
prefix=LibriSpeech
max_prev_words=500
splits="dev-clean dev-other train-clean-100 train-clean-360 train-other-500 test-clean test-other"
max_workers=8
stage=1
stop_stage=100

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [options]
Options:
    --data_dir <dir>          Path to LibriSpeech data (default: downloads/LibriSpeech)
    --prefix <str>            Prefix for output (default: LibriSpeech)
    --splits <str>            Data splits to process (default: dev-clean dev-other train-clean-100 train-clean-360 train-other-500 test-clean test-other)
    --max_prev_words <int>    Maximum words in previous context (default: 500)
    --max_workers <int>       Number of worker threads (default: 8)
    --stage <int>             Stage to start from (default: 1)
    --stop_stage <int>        Stage to stop at (default: 100)
EOF
)

# Save command line args for logging
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

. ./path.sh || exit 1;
. ./db.sh || exit 1;

# Copied from utils/fix_data_dir.sh
function check_sorted {
  file=$1
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
}

SECONDS=0
output_dir="data/LibriSpeech_long_prev_${max_prev_words}words"

log "$0 $*"
log "${run_args}"

log "Configuration:"
log "  data_dir: ${data_dir}"
log "  prefix: ${prefix}"
log "  max_prev_words: ${max_prev_words}"
log "  max_workers: ${max_workers}"
log "  output_dir: ${output_dir}"
log "  splits: ${splits}"
log "  stage: ${stage}"
log "  stop_stage: ${stop_stage}"

train_sets="data/LibriSpeech_long_prev_${max_prev_words}words/train-clean-100 \
            data/LibriSpeech_long_prev_${max_prev_words}words/train-clean-360 \
            data/LibriSpeech_long_prev_${max_prev_words}words/train-other-500"
dev_sets="data/LibriSpeech_long_prev_${max_prev_words}words/dev-clean \
          data/LibriSpeech_long_prev_${max_prev_words}words/dev-other"

train_out="data/train_librispeech_long_prev_${max_prev_words}words"
dev_out="data/dev_librispeech_long_prev_${max_prev_words}words"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation"
    python local/prepare_librispeech_long_prev_fix_multiprocess.py \
        --data_dir="${data_dir}" \
        --prefix="${prefix}" \
        --output_dir="${output_dir}" \
        --splits ${splits} \
        --max_prev_words ${max_prev_words} \
        --max_workers ${max_workers}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Data validation and fixing"
    utt_extra_files="text.prev text.ctc"
    for x in ${splits}; do
        if [ -d "${output_dir}/${x}" ]; then
            log "Processing ${output_dir}/${x}"
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${output_dir}/${x} || exit 1;
            utils/validate_data_dir.sh --no-feats --non-print ${output_dir}/${x} || exit 1;
        else
            log "Warning: ${output_dir}/${x} does not exist, skipping"
        fi
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Data combination"
    utt_extra_files="text.prev text.ctc"
    
    # Combine train
    train_sets_exist=""
    for train_set in data/LibriSpeech_long_prev_${max_prev_words}words/train-clean-100 \
                     data/LibriSpeech_long_prev_${max_prev_words}words/train-clean-360 \
                     data/LibriSpeech_long_prev_${max_prev_words}words/train-other-500; do
        if [ -d "${train_set}" ]; then
            train_sets_exist="${train_sets_exist} ${train_set}"
        fi
    done
    
    if [ -n "${train_sets_exist}" ]; then
        log "Combining train sets: ${train_sets_exist}"
        utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
            ${train_out} ${train_sets_exist} || exit 1;
        # NOTE(yifan): extra text files must be sorted and unique
        for f in ${utt_extra_files}; do
            check_sorted ${train_out}/${f}
        done
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${train_out} || exit 1;
        utils/validate_data_dir.sh --no-feats --non-print ${train_out} || exit 1;
    else
        log "Warning: No train sets found to combine"
    fi
    
    # Combine dev
    dev_sets_exist=""
    for dev_set in data/LibriSpeech_long_prev_${max_prev_words}words/dev-clean \
                   data/LibriSpeech_long_prev_${max_prev_words}words/dev-other; do
        if [ -d "${dev_set}" ]; then
            dev_sets_exist="${dev_sets_exist} ${dev_set}"
        fi
    done
    
    if [ -n "${dev_sets_exist}" ]; then
        log "Combining dev sets: ${dev_sets_exist}"
        utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
            ${dev_out} ${dev_sets_exist} || exit 1;
        # NOTE(yifan): extra text files must be sorted and unique
        for f in ${utt_extra_files}; do
            check_sorted ${dev_out}/${f}
        done
        utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${dev_out} || exit 1;
        utils/validate_data_dir.sh --no-feats --non-print ${dev_out} || exit 1;
    else
        log "Warning: No dev sets found to combine"
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]" 
