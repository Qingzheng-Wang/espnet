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

# 使用示例
if __name__ == "__main__":
    chapter_txt = "/ocean/projects/cis210027p/qwang20/espnet_whisper/egs2/owsm_v4_cb/s2t1/downloads/LibriSpeech/CHAPTERS.TXT"

    # 解析文件
    book2chapters, chapter2book = parse_chapters_file(chapter_txt)

    for book_id, chapters in book2chapters.items():
        if len(chapters) == 1:
            print(f"Book {book_id} has only one chapter: {chapters[0]}")
