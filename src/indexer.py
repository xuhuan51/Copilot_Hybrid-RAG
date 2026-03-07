"""
向量化 & Milvus入库
使用BGE-M3生成Dense+Sparse向量，写入Milvus
"""
import os
import json
from pathlib import Path
from tqdm import tqdm
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from FlagEmbedding import BGEM3FlagModel
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType
)


def load_chunks(chunks_path: str) -> list[dict]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"加载了 {len(chunks)} 个chunks")
    return chunks


def create_collection(collection_name: str, dense_dim: int = 1024):
    """创建Milvus Collection，支持Dense+Sparse混合检索"""

    # 如果已存在，先删除重建
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} 已存在，删除重建...")
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="header_path", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),

        # 🔴 新增图片路径字段 (可以为空字符串)
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1000),

        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields=fields, description="Hybrid RAG Document Chunks")
    collection = Collection(name=collection_name, schema=schema)

    # 创建索引
    # Dense向量索引
    collection.create_index(
        field_name="dense_vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
    )

    # Sparse向量索引
    collection.create_index(
        field_name="sparse_vector",
        index_params={
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP"
        }
    )

    print(f"Collection {collection_name} 创建成功")
    print(f"  Dense向量维度: {dense_dim}")
    print(f"  索引类型: IVF_FLAT (Dense) + SPARSE_INVERTED_INDEX (Sparse)")

    return collection


def embed_and_insert(
        chunks: list[dict],
        collection: Collection,
        model: BGEM3FlagModel,
        batch_size: int = 32
):
    """分批向量化并插入Milvus"""

    total = len(chunks)
    print(f"\n开始向量化并入库，共 {total} 个chunks，batch_size={batch_size}")

    for i in tqdm(range(0, total, batch_size), desc="向量化&入库"):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]

        # BGE-M3 同时生成 Dense + Sparse 向量
        embeddings = model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        dense_vectors = embeddings["dense_vecs"].tolist()

        # 转换sparse向量格式为Milvus要求的dict格式
        sparse_vectors = []
        for sparse_vec in embeddings["lexical_weights"]:
            # BGE-M3返回的是dict {token_id: weight}
            if isinstance(sparse_vec, dict):
                sparse_vectors.append(sparse_vec)
            else:
                sparse_vectors.append(dict(sparse_vec))

        # 准备插入数据
        insert_data = [
            [c["chunk_id"] for c in batch],
            [c["source_file"] for c in batch],
            [c.get("title", "")[:500] for c in batch],
            [c["header_path"][:500] for c in batch],
            [c["content"][:65535] for c in batch],

            # 🔴 提取图片路径，如果没有则为空字符串
            [c.get("image_path", "")[:1000] for c in batch],

            dense_vectors,
            sparse_vectors,
        ]

        collection.insert(insert_data)

    # flush确保数据持久化
    collection.flush()
    print(f"\n入库完成! 共插入 {collection.num_entities} 条记录")


def build_index(
        chunks_path: str = "data/chunks/all_chunks.json",
        collection_name: str = "hybrid_rag_docs",
        batch_size: int = 32
):
    """完整的入库流程"""

    # 1. 连接Milvus
    print("连接Milvus...")
    connections.connect(host="localhost", port="19530")

    # 2. 加载chunks
    chunks = load_chunks(chunks_path)

    # 3. 加载BGE-M3模型
    local_model_path = "/home/liuguangli/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

    print(f"加载BGE-M3模型(本地): {local_model_path}")
    model = BGEM3FlagModel(local_model_path, use_fp16=True, device="cuda")

    # 快速测试
    test_result = model.encode(["测试"], return_dense=True, return_sparse=True)
    dense_dim = len(test_result["dense_vecs"][0])
    print(f"  Dense向量维度: {dense_dim}")

    # 4. 创建Collection
    collection = create_collection(collection_name, dense_dim=dense_dim)

    # 5. 向量化并插入
    embed_and_insert(chunks, collection, model, batch_size=batch_size)

    # 6. 加载Collection到内存（检索前必须）
    collection.load()
    print(f"Collection已加载到内存，可以开始检索")

    # 断开连接
    connections.disconnect("default")
    print("完成!")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_index()