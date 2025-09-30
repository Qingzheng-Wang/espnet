#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

task=audio_text_dialogue

train_jsons="dump/raw_audio_text_dialogue_librispeech/train_clean_100/data.json "
valid_jsons="dump/raw_audio_text_dialogue_librispeech/train_clean_100/data.json "
test_jsons="dump/raw_audio_text_dialogue_librispeech/train_clean_100/data.json "

train_config=conf/train_parallel_smollm_1.7b.yaml
inference_config=conf/decode_general.yaml

token_list_dir=data/token_list/llm_vocab_smollm # use lllm vocab
# bpe_opts="--subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B"


subword_choice=huggingface
subword_model=HuggingFaceTB/SmolLM-1.7B

data_name=librispeech # this is just for finding the data.json file, and find the corresponding vocabulary list not real data use for train or other things

./speechlm.sh \
    --skip_data_prep true \
    --data_combo_name ls960_mlsen \
    --fs 16000 \
    --ngpu 2 \
    --nj 2 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference true \
    --token_list_dir ${token_list_dir} \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_jsons "${train_jsons}" \
    --valid_jsons "${valid_jsons}" \
    --test_jsons "${test_jsons}" \
    --dumpdir dump \
    --subword_choice ${subword_choice} \
    --subword_model ${subword_model} \
    --task ${task} \
    --data_name ${data_name} \
    "$@"
