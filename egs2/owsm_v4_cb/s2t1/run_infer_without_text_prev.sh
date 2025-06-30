#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_librispeech
valid_set=dev_librispeech
test_sets="test_librispeech_clean test_librispeech_other"

nbpe=50000
s2t_config=conf/train_conv2d8_size1024_e18_d18_mel128.yaml
inference_config=conf/decode_s2t.yaml

# score cleaner
cleaner=whisper_en
hyp_cleaner=whisper_en

./s2t_infer_without_text_prev.sh \
    --stage 1 \
    --stop_stage 13 \
    --use_lm false \
    --num_nodes 16 \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 16 \
    --num_splits_s2t 12 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" \
    --cleaner "${cleaner}" \
    --hyp_cleaner "${hyp_cleaner}" \
    --inference_s2t_model "valid.total_count.ave.pth" "$@"
