import os
import base64
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from docling_core.types.doc import TextItem, TableItem, PictureItem
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import InputFormat

# ────────────────────── 配置区 ──────────────────────

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 关键目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
IMAGE_SAVE_DIR = DATA_DIR / "images"
OUTPUT_MD_DIR = DATA_DIR / "parsed_docs"

# 创建目录
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = "http://localhost:11434"
VL_MODEL = "qwen2.5vl:32b"

# ────────────────────── 视觉模型处理 ──────────────────────

def process_vision_element(image: Image.Image, element_id: str) -> str:
    """
    保存原图并调用 Qwen-VL 生成摘要
    返回: 直接返回拼装好的 Markdown 格式文本（含图片路径和摘要）
    """
    image_path = IMAGE_SAVE_DIR / f"vision_{element_id}.png"
    image.save(image_path, format="PNG")

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    vl_prompt = """你是一个资深的架构专家。请详细解析这张系统架构图。
要求：识别组件、描述数据流向和调用关系，直接输出技术解析内容。"""

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": VL_MODEL,
                "prompt": vl_prompt,
                "stream": False,
                "images": [img_base64],
                "options": {"temperature": 0.1, "num_predict": 300}
            },
            timeout=120,
        )
        resp.raise_for_status()
        summary = resp.json()["response"].strip()

        # 🟢 返回符合协议的 Markdown 片段
        return f"\n\n![image]({image_path})\n> **[图表语义摘要]**\n> {summary}\n\n"

    except Exception as e:
        print(f"  [VL Error] 视觉解析失败: {e}")
        return f"\n\n![image]({image_path})\n> **[图表解析失败，仅保留原图]**\n\n"

# ────────────────────── 文档解析主流程 ──────────────────────

def parse_pdf_multimodal(pdf_path: str):
    """解析 PDF 并生成完整的 Markdown 文件"""
    print(f"\n🚀 开始解析文档: {pdf_path}")

    # 🔴 修复：初始化完整内容变量
    full_markdown_content = ""
    source_name = os.path.basename(pdf_path)

    # 配置 Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    try:
        doc = converter.convert(pdf_path).document
        items = list(doc.iterate_items())

        for item, level in tqdm(items, desc=f"解析 {source_name}"):
            # 1. 处理标题
            if isinstance(item, TextItem) and item.label.name in ["title", "section_header"]:
                prefix = "#" * (level + 1)
                full_markdown_content += f"\n\n{prefix} {item.text}\n"
                continue

            # 2. 处理文本和表格
            if isinstance(item, (TextItem, TableItem)):
                text = item.text if isinstance(item, TextItem) else item.export_to_markdown()
                full_markdown_content += text + "\n"

            # 3. 处理图片
            elif isinstance(item, PictureItem):
                img = item.get_image(doc)
                if img and img.width > 150 and img.height > 150:
                    img_id = f"{source_name}_p{item.prov[0].page_no}_{id(item)}"
                    # 获取视觉解析后的 Markdown 片段
                    full_markdown_content += process_vision_element(img, element_id=img_id)

        # 🔴 修复：保存为独立 Markdown 文件
        output_file = OUTPUT_MD_DIR / f"{source_name}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_markdown_content)

        print(f"✅ 解析完成: {output_file}")

    except Exception as e:
        print(f"❌ 解析文档 {source_name} 时出错: {e}")

# ────────────────────── 批量运行入口 ──────────────────────

if __name__ == "__main__":
    pdf_files = list(RAW_DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"请将 PDF 文件放入: {RAW_DOCS_DIR}")
    else:
        print(f"共发现 {len(pdf_files)} 个文档，开始执行多模态解析...")
        for pdf in pdf_files:
            parse_pdf_multimodal(str(pdf))