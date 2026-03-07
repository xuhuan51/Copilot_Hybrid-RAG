"""
递归分块器 - 按Markdown标题层级进行语义分块
利用Docling输出的Markdown结构，保证每个chunk语义完整
"""
import os
import json
import re
from pathlib import Path
from typing import Optional


def count_tokens(text: str) -> int:
    """简单按字符估算token数（中文1字≈1.5token，英文1词≈1token）"""
    # 粗略估算，后续可换成tiktoken精确计算
    return len(text) // 3


def split_by_headers(markdown_text: str) -> list[dict]:
    """
    按Markdown标题切分，保留层级关系
    返回: [{"level": 2, "title": "xxx", "content": "xxx"}, ...]
    """
    # 匹配 ## 标题
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    sections = []
    last_end = 0
    matches = list(header_pattern.finditer(markdown_text))

    for i, match in enumerate(matches):
        # 上一段的内容
        if i == 0 and match.start() > 0:
            # 标题前的内容（如文档开头）
            pre_content = markdown_text[:match.start()].strip()
            if pre_content:
                sections.append({
                    "level": 0,
                    "title": "文档开头",
                    "content": pre_content,
                    "header_path": []
                })

        # 当前section的内容范围
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[content_start:content_end].strip()

        level = len(match.group(1))  # #的数量
        title = match.group(2).strip()

        sections.append({
            "level": level,
            "title": title,
            "content": content,
            "header_path": []
        })

    # 如果没有标题，整个文档作为一个section
    if not sections:
        sections.append({
            "level": 0,
            "title": "全文",
            "content": markdown_text.strip(),
            "header_path": []
        })

    # 构建标题路径（如 "3. 事务隔离级别 > 3.1 READ UNCOMMITTED"）
    title_stack = {}
    for section in sections:
        level = section["level"]
        title_stack[level] = section["title"]
        # 清除更深层级的标题
        for l in list(title_stack.keys()):
            if l > level:
                del title_stack[l]
        section["header_path"] = [title_stack[l] for l in sorted(title_stack.keys()) if l <= level]

    return sections


def merge_small_sections(sections: list[dict], min_tokens: int = 100) -> list[dict]:
    """
    合并过小的section到前一个section
    避免出现只有标题没有内容的chunk
    """
    merged = []
    for section in sections:
        token_count = count_tokens(section["content"])
        if token_count < min_tokens and merged:
            # 合并到前一个
            header = "#" * section["level"]
            merged[-1]["content"] += f"\n\n{header} {section['title']}\n{section['content']}"
        else:
            merged.append(section)
    return merged


def split_long_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    对超长section按段落边界切分
    尽量在段落（空行）处切分，而不是硬截断
    """
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        test_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if count_tokens(test_chunk) > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            # overlap: 保留当前chunk末尾的一部分
            overlap_text = current_chunk[-overlap_tokens * 3:] if overlap_tokens > 0 else ""
            current_chunk = overlap_text + "\n\n" + para if overlap_text else para
        else:
            current_chunk = test_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def chunk_document(
        markdown_text: str,
        source_file: str,
        chunk_size: int = 768,
        chunk_overlap: int = 150,
        min_chunk_tokens: int = 50
) -> list[dict]:
    """
    对单个Markdown文档进行分块

    策略：
    1. 按标题层级切分成section
    2. 合并过小的section
    3. 对超长section按段落边界二次切分
    4. 添加元数据（来源文件、标题路径、chunk_id）
    """
    # Step 1: 按标题切分
    sections = split_by_headers(markdown_text)

    # Step 2: 合并过小的section
    sections = merge_small_sections(sections, min_tokens=min_chunk_tokens)

    # Step 3: 生成最终chunks
    chunks = []
    chunk_id = 0

    for section in sections:
        header_context = " > ".join(section["header_path"]) if section["header_path"] else ""
        full_content = f"[{header_context}]\n{section['content']}" if header_context else section["content"]

        token_count = count_tokens(full_content)

        # 🔴 新增：用正则提取图片路径
        # 匹配 doc_parser 生成的 ![image](路径)
        img_match = re.search(r'!\[image\]\((.*?)\)', full_content)
        image_path = img_match.group(1) if img_match else ""

        if token_count <= chunk_size:
            if token_count >= min_chunk_tokens:
                chunks.append({
                    "chunk_id": f"{source_file}_chunk_{chunk_id}",
                    "source_file": source_file,
                    "header_path": header_context,
                    "title": section["title"],
                    "content": full_content,
                    "token_count": token_count,
                    "image_path": image_path,  # 🔴 挂载字段
                })
                chunk_id += 1
        else:
            # 二次切分逻辑同理，也需要把 image_path 带上
            sub_chunks = split_long_section(full_content, chunk_size, chunk_overlap)
            for j, sub_content in enumerate(sub_chunks):
                # 检查切分后的子段落是否包含图片
                sub_img_match = re.search(r'!\[image\]\((.*?)\)', sub_content)
                sub_image_path = sub_img_match.group(1) if sub_img_match else ""

                tc = count_tokens(sub_content)
                if tc >= min_chunk_tokens:
                    chunks.append({
                        "chunk_id": f"{source_file}_chunk_{chunk_id}",
                        "source_file": source_file,
                        "header_path": header_context,
                        "title": f"{section['title']} (part {j + 1})",
                        "content": sub_content,
                        "token_count": tc,
                        "image_path": sub_image_path,  # 🔴 挂载字段
                    })
                    chunk_id += 1
    return chunks


def chunk_all_documents(parsed_dir: str, output_dir: str, chunk_size: int = 768, chunk_overlap: int = 150):
    """
    对parsed_docs下所有Markdown文件进行分块
    """
    parsed_path = Path(parsed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    md_files = [f for f in parsed_path.glob("*.md") if not f.name.startswith("_")]

    if not md_files:
        print(f"在 {parsed_dir} 下没有找到Markdown文件")
        return

    print(f"找到 {len(md_files)} 个Markdown文件，开始分块...")
    print(f"参数: chunk_size={chunk_size}, overlap={chunk_overlap}")

    all_chunks = []

    for md_file in md_files:
        print(f"\n  -> 分块: {md_file.name}")
        markdown_text = md_file.read_text(encoding="utf-8")

        chunks = chunk_document(
            markdown_text=markdown_text,
            source_file=md_file.stem,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(chunks)
        print(f"     生成 {len(chunks)} 个chunks")

    # 保存所有chunks
    chunks_file = output_path / "all_chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # 打印统计
    token_counts = [c["token_count"] for c in all_chunks]
    print(f"\n{'=' * 50}")
    print(f"分块完成!")
    print(f"  总chunk数: {len(all_chunks)}")
    print(f"  平均token数: {sum(token_counts) // len(token_counts)}")
    print(f"  最大token数: {max(token_counts)}")
    print(f"  最小token数: {min(token_counts)}")
    print(f"  输出文件: {chunks_file}")

    # 按来源统计
    print(f"\n各文档chunk分布:")
    from collections import Counter
    source_counts = Counter(c["source_file"] for c in all_chunks)
    for source, count in source_counts.most_common():
        print(f"  {source}: {count} chunks")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chunk_all_documents(
        "data/parsed_docs",
        "data/chunks",
        chunk_size=512,
        chunk_overlap=100
    )