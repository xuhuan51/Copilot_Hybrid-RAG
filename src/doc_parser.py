"""
文档解析器 - 使用 Docling 将 PDF 转换为 Markdown
"""
import os
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
except ImportError:
    print("请先安装 docling: pip install docling")
    exit(1)


def parse_documents(input_dir: str, output_dir: str) -> list[dict]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported = {".pdf", ".docx", ".doc"}
    files = [f for f in input_path.iterdir() if f.suffix.lower() in supported]

    if not files:
        print(f"在 {input_dir} 下没有找到PDF/Word文档")
        return []

    print(f"找到 {len(files)} 个文档，开始解析...")

    # 使用GPU加速版面分析
    accelerator = AcceleratorOptions(device="cuda")
    pipeline_options = PdfPipelineOptions(accelerator_options=accelerator)

    # 修改这里的 FormatOption 为 PdfFormatOption
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    results = []

    for file in tqdm(files, desc="解析文档"):
        try:
            print(f"\n  -> 正在解析: {file.name}")
            result = converter.convert(str(file))
            markdown_content = result.document.export_to_markdown()

            md_filename = file.stem + ".md"
            md_path = output_path / md_filename
            md_path.write_text(markdown_content, encoding="utf-8")

            doc_meta = {
                "source_file": file.name,
                "markdown_path": str(md_path),
                "file_size_kb": round(file.stat().st_size / 1024, 1),
                "markdown_length": len(markdown_content),
                "parsed_at": datetime.now().isoformat(),
            }
            results.append(doc_meta)
            print(f"  完成: {md_filename} ({len(markdown_content)} chars)")

        except Exception as e:
            print(f"  解析失败 {file.name}: {e}")
            results.append({
                "source_file": file.name,
                "error": str(e),
            })

    report_path = output_path / "_parse_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"\n{'='*50}")
    print(f"解析完成: {len(success)} 成功, {len(failed)} 失败")
    print(f"Markdown输出目录: {output_path}")

    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parse_documents("data/raw_docs", "data/parsed_docs")