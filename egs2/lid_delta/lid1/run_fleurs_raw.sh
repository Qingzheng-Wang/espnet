#!/usr/bin/env bash
set -e
set -u
set -o pipefail

train_set="train_fleurs_lang"
valid_set="dev_fleurs_lang"
test_sets="dev_fleurs_lang"
tsne_set="train_fleurs_lang"
feats_type="raw"
exp_dir="exp_fleurs_raw"
inference_model="valid.accuracy.best.pth"

./lid.sh \
    --feats_type ${feats_type} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --tsne_set "${tsne_set}" \
    --inference_model ${inference_model} \
    --inference_batch_size 8 \
    --extract_embd false \
    --save_every 1000 \
    --nj 8 \
    --ngpu 1 \
    --spk_args "
        --use_wandb true 
        --wandb_project lid 
        --wandb_entity qingzhew-carnegie-mellon-university
        " \
    --expdir "${exp_dir}" \
    "$@"
