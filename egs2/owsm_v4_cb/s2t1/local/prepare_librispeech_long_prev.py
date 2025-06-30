"""Prepare LibriSpeech data for English ASR."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import logging

from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    Utterance,
    merge_short_utterances_long_prev,
    generate_long_utterances_long_prev
)

def parse_chapters_file(file_path):
    """
    解析CHAPTER.TXT文件，生成book2chapters字典
    
    Args:
        file_path (str): CHAPTER.TXT文件路径
    
    Returns:
        dict: book_id -> list of chapter info
    """
    book2chapters = {}
    chapter2book = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过注释行和空行
            line = line.strip()
            if not line or line.startswith(';'):
                continue
                
            # 按管道符分割字段
            fields = [field.strip() for field in line.split('|')]
            
            # 确保有足够的字段
            if len(fields) < 8:
                continue
                
            try:
                chapter_id = int(fields[0])
                book_id = int(fields[5])
                
                
                # 添加到book2chapters字典
                if book_id not in book2chapters:
                    book2chapters[book_id] = []
                book2chapters[book_id].append(chapter_id)
                chapter2book[chapter_id] = book_id
                
            except (ValueError, IndexError) as e:
                # 跳过格式有问题的行
                print(f"跳过格式错误的行: {line}")
                continue
    
    return book2chapters, chapter2book


def split_prev_asr_text(data_dir):
    chapter_txt = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/downloads/LibriSpeech/CHAPTERS.TXT"
    book2chapters, chapter2book = parse_chapters_file(chapter_txt)

    book2chapters_split = defaultdict(list)
    chapter2book_split = {}
    speakers = [d.name for d in (data_dir / split).iterdir() if d.is_dir()]
    for speaker in speakers:
        for chapter in (data_dir / "mp3" / speaker).iterdir():
            if chapter.is_dir():
                chapter_id = int(chapter.name)
                book_id = chapter2book.get(chapter_id, None)
                chapter2book_split[chapter] = book_id
                book2chapters_split[book_id].append(chapter_id)

    book_2_prev_text_chapters = {}
    book_2_asr_text_chapters = {}
    for book_id, chapters in book2chapters_split.items():
        num_chapters = len(chapters)
        chapters = sorted(chapters)
        prev_text_chapters = chapters[:num_chapters // 2]
        asr_text_chapters = chapters[num_chapters // 2:]
        book_2_prev_text_chapters.update({book_id: prev_text_chapters})
        book_2_asr_text_chapters.update({book_id: asr_text_chapters})

    return book_2_prev_text_chapters, book_2_asr_text_chapters



def collect_data(
    data_dir: Union[Path, str], split: str, prefix: str
) -> List[List[Utterance]]:
    """Collect utterances in each long talk."""
    data_dir = Path(data_dir)
    speakers = [d.name for d in (data_dir / split).iterdir() if d.is_dir()]

    chapter_txt = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/downloads/LibriSpeech/CHAPTERS.TXT"
    book2chapters, chapter2book = parse_chapters_file(chapter_txt)


    book_2_prev_text_chapters, book_2_asr_text_chapters = split_prev_asr_text(data_dir)

    book_2_prev_text_utts = defaultdict(list)
    book_2_asr_text_talks = defaultdict(list)
    for speaker in speakers:
        for chapter in (data_dir / "mp3" / speaker).iterdir():
            if chapter.is_dir():
                chapter_id = int(chapter.name)
                book_id = chapter2book.get(chapter_id, None)
            
                if book_id is None or book_id in [124, 246]:
                    # the chapter id cannot found in CHAPTER.TXT, denotes this chapter is not included in the dataset
                    # book 124, 246 only has one setence labelled in the mp3 directory
                    # since book 124 belong to train-clean-360, pass it
                    continue

                utts = []
                audio = str((chapter / f"{chapter.name}.mp3").resolve())
                with open(
                    chapter / f"{speaker}-{chapter.name}.sents.seg.txt", "r"
                ) as seg_f, open(
                    chapter / f"{speaker}-{chapter.name}.sents.trans.txt", "r"
                ) as trans_f:
                    seg_lines = [line.strip() for line in seg_f.readlines()]
                    trans_lines = [line.strip() for line in trans_f.readlines()]
                    assert len(seg_lines) == len(trans_lines)
                    for seg, trans in zip(seg_lines, trans_lines):
                        utt_id, start_time, end_time = seg.split()
                        assert utt_id == trans.split(maxsplit=1)[0]
                        text = trans.split(maxsplit=1)[1].lower()
                        utts.append(
                            Utterance(
                                utt_id=f"{prefix}_{utt_id}",
                                wav_id=f"{prefix}_{speaker}_{chapter.name}",
                                wav_path=f"sox {audio} -t wav -c 1 -r 16000 - |",
                                start_time=float(start_time),
                                end_time=float(end_time),
                                lang="<eng>",
                                task="<asr>",
                                text=text,
                                asr_text=text,
                            )
                        )
                prev_text_chapters = book_2_prev_text_chapters[book_id]
                if not prev_text_chapters:
                    # If prev_text_chapters is empty (i.e. only one chapter total),
                    # split all utterances into two halves
                    prev_text_utts = utts[:len(utts) // 2]
                    asr_text_utts = utts[len(utts) // 2:]
                    book_2_prev_text_utts[book_id].extend(prev_text_utts)
                    book_2_asr_text_talks[book_id].append(asr_text_utts)
                else:
                    # If multiple chapters, use the first half chapters for prev text
                    if chapter_id in prev_text_chapters:
                        book_2_prev_text_utts[book_id].extend(utts)
                    elif chapter_id in book_2_asr_text_chapters.get(book_id, []):
                        book_2_asr_text_talks[book_id].append(utts)
    book_2_prev_text_utts_keys = set(book_2_prev_text_utts.keys())
    book_2_asr_text_talks_keys = set(book_2_asr_text_talks.keys())
    print(f"diff: {book_2_prev_text_utts_keys - book_2_asr_text_talks_keys}")
    assert len(book_2_prev_text_utts) == len(book_2_asr_text_talks), f"book_2_prev_text_utts: {book_2_prev_text_utts.keys()}, book_2_asr_text_talks: {book_2_asr_text_talks.keys()}"
    return book_2_prev_text_utts, book_2_asr_text_talks


def parse_args():
    parser = ArgumentParser(description="Prepare data.")
    parser.add_argument("--data_dir", type=Path, help="Path to raw data.")
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
        default=["dev", "train"],
        help="Data splits to prepare.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        write_dir = args.output_dir / split
        write_dir.mkdir(parents=True, exist_ok=True)

        wavscp_fp = open(write_dir / "wav.scp", "w")  # wav-id wav-path
        segments_fp = open(
            write_dir / "segments", "w"
        )  # utt-id wav-id start-time end-time
        text_fp = open(write_dir / "text", "w")  # utt-id transcript
        textprev_fp = open(write_dir / "text.prev", "w")
        textctc_fp = open(
            write_dir / "text.ctc", "w"
        )  # text for ASR CTC w/o special tokens
        utt2spk_fp = open(write_dir / "utt2spk", "w")

        book_2_prev_text_utts, book_2_asr_text_talks = collect_data(
            data_dir=args.data_dir,
            split=split,
            prefix=args.prefix,
        )

        book_2_merge_prev_text_utts = {}
        for book_id, prev_text_utts in book_2_prev_text_utts.items():
            if len(prev_text_utts) > 0:
                merged_prev_text_utt = merge_short_utterances_long_prev(
                    prev_text_utts
                )
                book_2_merge_prev_text_utts[book_id] = merged_prev_text_utt
            else:
                print(f"book id: {book_id}")

        for book_id, asr_text_talks in book_2_asr_text_talks.items():
            if len(asr_text_talks) > 0:
                prev_text = book_2_merge_prev_text_utts[book_id]
                for asr_text_talk in asr_text_talks:
                    for u in generate_long_utterances_long_prev(asr_text_talk, prev_text):
                        wavscp_fp.write(f"{u.wav_id} {u.wav_path}\n")
                        segments_fp.write(
                            f"{u.utt_id} {u.wav_id} {u.start_time:.2f} {u.end_time:.2f}\n"
                        )
                        text_fp.write(f"{u.utt_id} {u.lang}{u.task}{u.text_with_time}\n")
                        textprev_fp.write(f"{u.utt_id} {u.prev_text}\n")
                        textctc_fp.write(f"{u.utt_id} {u.asr_text}\n")
                        utt2spk_fp.write(f"{u.utt_id} {u.utt_id}\n")


        wavscp_fp.close()
        segments_fp.close()
        text_fp.close()
        textprev_fp.close()
        textctc_fp.close()
        utt2spk_fp.close()

    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        "<en>",
        "<asr>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
