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
output_dir=data/LibriSpeech_long_prev_fix
splits="dev-clean dev-other train-clean-100 train-clean-360 train-other-500 test-clean test-other"
max_prev_words=500

train_sets="data/LibriSpeech_long_prev_fix/train-clean-100 \
            data/LibriSpeech_long_prev_fix/train-clean-360 \
            data/LibriSpeech_long_prev_fix/train-other-500"
dev_sets="data/LibriSpeech_long_prev_fix/dev-clean \
          data/LibriSpeech_long_prev_fix/dev-other"

train_out=data/train_librispeech_long_prev_fix
dev_out=data/dev_librispeech_long_prev_fix

python local/prepare_librispeech_long_prev_fix.py \
    --data_dir ${data_dir} \
    --prefix ${prefix} \
    --output_dir ${output_dir} \
    --splits ${splits} \
    --max_prev_words ${max_prev_words} || exit 1;

utt_extra_files="text.prev text.ctc"
for x in ${splits}; do
    utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${output_dir}/${x} || exit 1;
    utils/validate_data_dir.sh --no-feats --non-print ${output_dir}/${x} || exit 1;
done

# Combine train
utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
    ${train_out} ${train_sets} || exit 1;
# NOTE(yifan): extra text files must be sorted and unique
for f in ${utt_extra_files}; do
    check_sorted ${train_out}/${f}
done
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${train_out} || exit 1;
utils/validate_data_dir.sh --no-feats --non-print ${train_out} || exit 1;

# Combine dev
utils/combine_data.sh --skip_fix true --extra-files "${utt_extra_files}" \
    ${dev_out} ${dev_sets} || exit 1;
# NOTE(yifan): extra text files must be sorted and unique
for f in ${utt_extra_files}; do
    check_sorted ${dev_out}/${f}
done
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${dev_out} || exit 1;
utils/validate_data_dir.sh --no-feats --non-print ${dev_out} || exit 1;

log "Successfully finished. [elapsed=${SECONDS}s]"
