#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# parse args
dump_dir=dump/raw
train_set=
test_sets=

. utils/parse_options.sh || exit 1;

if [ -z "${dump_dir}" ] || [ -z "${train_set}" ] || [ -z "${test_sets}" ]; then
    echo "Usage: $0 --dump_dir <dump_dir> --train_set <train_set> --test_sets <test_sets>"
    exit 1
fi

# check if the dump_dir exists
if [ ! -d "${dump_dir}" ]; then
    echo "Error: dump_dir ${dump_dir} does not exist."
    exit 1
fi
# check if the train_set exists
if [ ! -d "${dump_dir}/${train_set}" ]; then
    echo "Error: train_set ${train_set} does not exist in ${dump_dir}."
    exit 1
fi
# check if the test_sets exists
for test_set in ${test_sets}; do
    if [ ! -d "${dump_dir}/${test_set}" ]; then
        echo "Error: test_set ${test_set} does not exist in ${dump_dir}."
        exit 1
    fi
done

python local/prepare_ood_test.py \
    --dump_dir ${dump_dir} \
    --train_set ${train_set} \
    --test_sets "${test_sets}" \
    "$@"

cross_sets=""
for test_set in ${test_sets}; do
    cross_set="${test_set}_cross_${train_set}"
    cross_sets="${cross_sets} ${cross_set}"
done
for cross_set in ${cross_sets}; do
    if [ ! -d "${dump_dir}/${cross_set}" ]; then
        if [ -f "${dump_dir}/${cross_set}/utt2spk" ]; then
            ./utils/utt2spk_to_spk2utt.pl ${dump_dir}/${cross_set}/utt2spk > ${dump_dir}/${cross_set}/spk2utt
            cp ${dump_dir}/${cross_set}/spk2utt ${dump_dir}/${cross_set}/category2utt 
        else
            echo "Error: File ${dump_dir}/${cross_set}/utt2spk does not exist."
            exit 1
        fi
    fi
done

echo "Successfully prepared OOD test sets to ${cross_sets}."
