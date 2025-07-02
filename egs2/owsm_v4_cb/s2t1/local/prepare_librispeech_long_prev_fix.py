"""Prepare LibriSpeech data for English ASR with long previous context."""

import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
from tqdm import tqdm

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    Utterance,
    merge_short_utterances_long_prev,
    generate_long_utterances_long_prev
)


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration using librosa."""
    try:
        duration = librosa.get_duration(filename=audio_path)
        return duration
    except Exception as e:
        logging.warning(f"Cannot get duration for {audio_path}: {e}")
        return 0.0


def collect_data(
    data_dir: Union[Path, str], split: str, prefix: str, max_prev_words: int = 500
) -> Tuple[Dict, Dict]:
    """Collect utterances from processed LibriSpeech data and organize by chapter."""
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    # 存储每个chapter的utterances
    chapter_2_utterances = defaultdict(list)
    
    # 读取所有speaker目录
    for speaker_dir in sorted(split_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        
        # 读取每个chapter目录
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
                
            chapter_id = chapter_dir.name
            chapter_key = f"{speaker_id}_{chapter_id}"
            
            # 读取转录文件
            trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if not trans_file.exists():
                logging.warning(f"Transcription file not found: {trans_file}")
                continue
                
            # 解析转录文件
            with open(trans_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                        
                    utt_id, text = parts
                    text = text.lower()
                    
                    # 音频文件路径
                    audio_file = chapter_dir / f"{utt_id}.flac"
                    if not audio_file.exists():
                        logging.warning(f"Audio file not found: {audio_file}")
                        continue
                    
                    # 获取音频时长
                    duration = get_audio_duration(str(audio_file))
                    if duration == 0.0:
                        continue
                    
                    # 创建Utterance对象
                    speaker_id_int = int(speaker_id)
                    chapter_id_int = int(chapter_id)
                    utt = Utterance(
                        utt_id=utt_id,
                        wav_id=f"{speaker_id_int:04d}_{chapter_id_int:06d}",
                        wav_path=f"sox {str(audio_file)} -t wav -c 1 -r 16000 - |",
                        start_time=0.0,  # 单个utterance从0开始
                        end_time=duration,
                        lang="<eng>",
                        task="<asr>",
                        text=text,
                        asr_text=text,
                        speaker_id=speaker_id,
                    )
                    
                    chapter_2_utterances[chapter_key].append(utt)
    
    # 按时间排序每个chapter的utterances（通过utt_id中的序号）
    for chapter_key in chapter_2_utterances:
        chapter_2_utterances[chapter_key].sort(key=lambda x: x.utt_id)
    
    # 为每个chapter分配前文和ASR文本
    chapter_2_prev_text_utts = {}
    chapter_2_asr_text_talks = {}
    
    for chapter_key, utterances in chapter_2_utterances.items():
        if len(utterances) < 2:  # 至少需要2个utterance才能分配前文和ASR
            continue
            
        # 计算累积单词数来分配前文
        prev_utts = []
        asr_utts = []
        cumulative_words = 0
        
        for i, utt in enumerate(utterances):
            utt_word_count = len(utt.text.split())
            
            if cumulative_words + utt_word_count <= max_prev_words:
                prev_utts.append(utt)
                cumulative_words += utt_word_count
            else:
                # 剩余的utterances作为ASR文本
                if cumulative_words == 0:  # 至少要有一个utterance作为前文
                    prev_utts.append(utt)
                    asr_utts = utterances[i+1:]
                else:
                    asr_utts = utterances[i:]
                break
        else:
            # 如果所有utterances都能作为前文，则最后一个作为ASR
            if len(prev_utts) > 1:
                asr_utts = [prev_utts.pop()]
        
        if prev_utts and asr_utts:
            chapter_2_prev_text_utts[chapter_key] = prev_utts
            chapter_2_asr_text_talks[chapter_key] = [asr_utts]  # 包装成list of talks
    
    logging.info(f"Chapters with prev text: {len(chapter_2_prev_text_utts)}")
    logging.info(f"Chapters with asr text: {len(chapter_2_asr_text_talks)}")
    
    return chapter_2_prev_text_utts, chapter_2_asr_text_talks


def create_combined_wav_id_and_path(utterances: List[Utterance]) -> Tuple[str, str]:
    """Create a combined wav_id and wav_path for multiple utterances from same chapter."""
    if len(utterances) == 1:
        return utterances[0].wav_id, utterances[0].wav_path
    
    # 对于多个utterances，我们需要从同一个chapter的音频文件创建连续的音频
    # 假设所有utterances来自同一个chapter，我们可以使用第一个的wav_id和wav_path
    # 在实际应用中，通过segments文件来处理时间偏移
    first_utt = utterances[0]
    return first_utt.wav_id, first_utt.wav_path


def transform_utterances_to_talks(utterances: List[Utterance]) -> List[List[Utterance]]:
    """Transform individual utterances into talks by combining them based on 30s duration."""
    if not utterances:
        return []
    
    talks = []
    current_talk = []
    current_duration = 0.0
    target_duration = 30.0
    
    # LibriSpeech的utterances已经是按顺序排列的，我们只需要按时长组合
    for utt in utterances:
        utt_duration = utt.end_time - utt.start_time
        
        if current_duration + utt_duration <= target_duration:
            current_talk.append(utt)
            current_duration += utt_duration
        else:
            # 如果当前talk不为空，保存它
            if current_talk:
                talks.append(current_talk)
            
            # 开始新的talk
            current_talk = [utt]
            current_duration = utt_duration
    
    # 添加最后一个talk（如果不为空）
    if current_talk:
        talks.append(current_talk)
    
    # 为每个talk创建新的wav_id和更新时间戳
    processed_talks = []
    for talk_idx, talk in enumerate(talks):
        if not talk:
            continue
            
        # 创建新的wav_id
        base_wav_id = talk[0].wav_id
        new_wav_id = f"{base_wav_id}_talk_{talk_idx:03d}"
        
        # 创建连续的音频文件路径（使用sox连接音频）
        audio_files = []
        for utt in talk:
            # 从sox命令中提取音频文件路径
            if "sox" in utt.wav_path and "|" in utt.wav_path:
                parts = utt.wav_path.split()
                if len(parts) >= 2:
                    audio_files.append(parts[1])
            else:
                audio_files.append(utt.wav_path)
        
        if audio_files:
            combined_wav_path = f"sox {' '.join(audio_files)} -t wav -c 1 -r 16000 - |"
        else:
            combined_wav_path = talk[0].wav_path
        
        # 更新talk中每个utterance的时间戳以形成连续音频
        cumulative_time = 0.0
        updated_talk = []
        for utt in talk:
            utt_duration = utt.end_time - utt.start_time
            new_utt = Utterance(
                utt_id=utt.utt_id,
                wav_id=new_wav_id,
                wav_path=combined_wav_path,
                start_time=cumulative_time,
                end_time=cumulative_time + utt_duration,
                lang=utt.lang,
                task=utt.task,
                text=utt.text,
                asr_text=utt.asr_text,
                speaker_id=utt.speaker_id,
            )
            updated_talk.append(new_utt)
            cumulative_time += utt_duration
        
        processed_talks.append(updated_talk)
    
    return processed_talks


def parse_args():
    parser = ArgumentParser(description="Prepare LibriSpeech data with long previous context.")
    parser.add_argument("--data_dir", type=Path, help="Path to processed LibriSpeech data.")
    parser.add_argument(
        "--prefix", type=str, help="Prefix that will be added to utt id."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to save the output data.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["dev-clean", "train-clean-100"],
        help="Data splits to prepare.",
    )
    parser.add_argument(
        "--max_prev_words",
        type=int,
        default=500,
        help="Maximum number of words in previous context (default: 500).",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        logging.info(f"Processing split: {split}")
        write_dir = args.output_dir / split
        write_dir.mkdir(parents=True, exist_ok=True)

        chapter_2_prev_text_utts, chapter_2_asr_text_talks = collect_data(
            data_dir=args.data_dir,
            split=split,
            prefix=args.prefix,
            max_prev_words=args.max_prev_words,
        )

        # 处理每个chapter的数据
        wav_scp_lines = []
        segments_lines = []
        text_lines = []
        textprev_lines = []
        textctc_lines = []
        utt2spk_lines = []
        
        for chapter_key, asr_text_talks in chapter_2_asr_text_talks.items():
            if len(asr_text_talks) > 0:
                # 获取对应chapter的prev text
                prev_text_utts = chapter_2_prev_text_utts.get(chapter_key, [])
                
                # 合并prev text utterances
                if prev_text_utts:
                    merged_prev_text_utt = merge_short_utterances_long_prev(prev_text_utts)
                    prev_text = merged_prev_text_utt
                else:
                    prev_text = None
                    logging.warning(f"No prev text for chapter: {chapter_key}")
                
                # 将ASR utterances转换为talks（30秒片段）
                for asr_utterances in asr_text_talks:
                    talks = transform_utterances_to_talks(asr_utterances)
                    
                    for talk in talks:
                        # 为每个talk生成长utterances
                        for u in generate_long_utterances_long_prev(talk, prev_text):
                            wav_scp_lines.append(f"{u.wav_id} {u.wav_path}\n")
                            segments_lines.append(
                                f"{u.utt_id} {u.wav_id} {u.start_time:.2f} {u.end_time:.2f}\n"
                            )
                            text_lines.append(f"{u.utt_id} {u.lang}{u.task}{u.text_with_time}\n")
                            textprev_lines.append(f"{u.utt_id} {u.prev_text}\n")
                            textctc_lines.append(f"{u.utt_id} {u.asr_text}\n")
                            utt2spk_lines.append(f"{u.utt_id} {u.speaker_id}\n")

        # 按utt-id排序
        wav_scp_lines = sorted(wav_scp_lines, key=lambda x: x.split()[0])
        segments_lines = sorted(segments_lines, key=lambda x: x.split()[0])
        text_lines = sorted(text_lines, key=lambda x: x.split()[0])
        textprev_lines = sorted(textprev_lines, key=lambda x: x.split()[0])
        textctc_lines = sorted(textctc_lines, key=lambda x: x.split()[0])
        utt2spk_lines = sorted(utt2spk_lines, key=lambda x: x.split()[0])

        # 写入文件
        with open(write_dir / "wav.scp", "w") as fp:
            fp.writelines(wav_scp_lines)
        with open(write_dir / "segments", "w") as fp:
            fp.writelines(segments_lines)
        with open(write_dir / "text", "w") as fp:
            fp.writelines(text_lines)
        with open(write_dir / "text.prev", "w") as fp:
            fp.writelines(textprev_lines)
        with open(write_dir / "text.ctc", "w") as fp:
            fp.writelines(textctc_lines)
        with open(write_dir / "utt2spk", "w") as fp:
            fp.writelines(utt2spk_lines)

        logging.info(f"Finished processing {split}: {len(text_lines)} utterances")

    # 创建特殊符号文件
    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        "<eng>",
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")

    logging.info("Data preparation completed!")
