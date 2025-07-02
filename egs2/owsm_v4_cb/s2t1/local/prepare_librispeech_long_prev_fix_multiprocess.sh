#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

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

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

data_dir=downloads/LibriSpeech
prefix=LibriSpeech
output_dir=data/${prefix}_long_prev_fix_multiprocess
splits="dev-clean"
max_prev_words=500
max_workers=8

log "data preparation started"

if [ ! -e "${data_dir}" ]; then
    log "LibriSpeech data not found at ${data_dir}. Please download it first."
    exit 1;
fi

# Prepare LibriSpeech data using multiprocess version
python local/prepare_librispeech_long_prev_fix_multiprocess.py \
    --data_dir="${data_dir}" \
    --prefix="${prefix}" \
    --output_dir="${output_dir}" \
    --splits ${splits} \
    --max_prev_words ${max_prev_words} \
    --max_workers ${max_workers}

for x in ${splits}; do
    log "fix_data_dir.sh ${output_dir}/${x}"
    utils/fix_data_dir.sh "${output_dir}/${x}"
    
    log "validate_data_dir.sh ${output_dir}/${x}"
    utils/validate_data_dir.sh --no-feats "${output_dir}/${x}"
done

# Combine all data
if [ -d "${output_dir}/train-clean-100" ] && [ -d "${output_dir}/dev-clean" ]; then
    log "combine_data.sh ${output_dir}/train_dev ${output_dir}/train-clean-100 ${output_dir}/dev-clean"
    utils/combine_data.sh "${output_dir}/train_dev" "${output_dir}/train-clean-100" "${output_dir}/dev-clean"
    
    log "fix_data_dir.sh ${output_dir}/train_dev"
    utils/fix_data_dir.sh "${output_dir}/train_dev"
    
    log "validate_data_dir.sh ${output_dir}/train_dev"
    utils/validate_data_dir.sh --no-feats "${output_dir}/train_dev"
elif [ -d "${output_dir}/dev-clean" ]; then
    log "Only dev-clean found, skipping train_dev combination"
else
    log "No valid data directories found for combination"
fi

log "Successfully finished. [elapsed=${SECONDS}s]" 