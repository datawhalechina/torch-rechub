"""Unified HLLM data preprocessing script.

This script combines text extraction and item embedding generation into a single pipeline:
1. Extract movie text information from MovieLens-1M (movies.dat)
2. Generate item embeddings using TinyLlama or Baichuan2
3. Save all necessary output files

Usage:
    python preprocess_hllm_data.py --model_type tinyllama --device cuda
    python preprocess_hllm_data.py --model_type baichuan2 --device cuda
"""

import os
import pickle
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import tqdm

# Get the directory where this script is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = _SCRIPT_DIR
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "processed")

# Official ByteDance HLLM item prompt
ITEM_PROMPT = "Compress the following sentence into embedding: "
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_ARCHIVE = "ml-1m.zip"
MOVIELENS_REQUIRED_FILES = ("ratings.dat", "movies.dat", "users.dat")


def download_file(url, output_path, overwrite=False):
    """Download a file with a progress bar."""
    if os.path.exists(output_path) and not overwrite:
        print(f"文件已存在，跳过下载: {output_path}")
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    progress_bar = None

    print(f"下载: {url}")
    print(f"保存到: {output_path}")

    def _progress_hook(block_num, block_size, total_size):
        nonlocal progress_bar
        if progress_bar is None:
            total = total_size if total_size > 0 else None
            progress_bar = tqdm.tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(output_path),
            )
        downloaded = block_num * block_size
        if progress_bar.total is not None:
            downloaded = min(downloaded, progress_bar.total)
        progress_bar.update(downloaded - progress_bar.n)

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress_hook)
        os.replace(tmp_path, output_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"下载失败: {exc}")
        return False
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(f"下载完成: {output_path}")
    return True


def extract_movielens_archive(archive_path, data_dir, overwrite=False):
    """Extract MovieLens-1M files from the official zip archive."""
    print(f"解压 MovieLens-1M 数据: {archive_path}")
    os.makedirs(data_dir, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        for filename in MOVIELENS_REQUIRED_FILES:
            member = f"ml-1m/{filename}"
            output_path = os.path.join(data_dir, filename)
            if member not in names:
                print(f"压缩包中缺少文件: {member}")
                return False
            if os.path.exists(output_path) and not overwrite:
                print(f"文件已存在，跳过解压: {output_path}")
                continue
            with archive.open(member) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
            print(f"已解压: {output_path}")

    return True


def ensure_movielens_data(data_dir, download=True, overwrite=False):
    """Download/extract MovieLens-1M data or validate existing local files."""
    required_paths = [os.path.join(data_dir, filename) for filename in MOVIELENS_REQUIRED_FILES]
    if not download:
        print("已禁用下载，将使用本地已有 MovieLens-1M 文件。")
        missing = [path for path in required_paths if not os.path.exists(path)]
        if missing:
            print("缺少必要文件:")
            for path in missing:
                print(f"  - {path}")
            return False
        return True

    archive_path = os.path.join(data_dir, MOVIELENS_ARCHIVE)
    if not download_file(MOVIELENS_1M_URL, archive_path, overwrite=overwrite):
        return False
    return extract_movielens_archive(archive_path, data_dir, overwrite=overwrite)


def check_environment(model_type, device):
    """Check GPU, CUDA, and VRAM availability."""
    import torch

    print("\n" + "=" * 80)
    print("环境检查")
    print("=" * 80)

    if device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ 错误：指定了 --device cuda，但系统中没有可用的 GPU")
            return False

        print("✅ GPU 可用")
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   总显存: {total_memory:.2f} GB")

        required_vram = 3 if model_type == 'tinyllama' else 16
        if total_memory < required_vram:
            print(f"⚠️  警告：需要至少 {required_vram}GB 显存，但系统仅有 {total_memory:.2f}GB")
            response = input("   是否继续？(y/n): ").strip().lower()
            if response != 'y':
                return False
        else:
            print(f"✅ 显存充足（需要 {required_vram}GB，实际 {total_memory:.2f}GB）")
    else:
        print("✅ 使用 CPU 进行计算（速度会较慢）")

    print("✅ 环境检查通过\n")
    return True


def get_model_name_and_dim(model_type):
    """Return the HuggingFace repo id and hidden dimension for a model type."""
    if model_type == 'tinyllama':
        return 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 2048
    if model_type == 'baichuan2':
        return 'baichuan-inc/Baichuan2-7B-Chat', 4096
    raise ValueError(f"Unsupported model_type: {model_type}")


def check_model_exists(model_type):
    """Check if the model exists in the HuggingFace cache."""
    model_name, _ = get_model_name_and_dim(model_type)

    if os.path.isdir(model_name):
        print(f"\n检查本地模型目录: {model_name}")
        print(f"✅ 使用本地模型: {model_name}")
        return True
    if os.path.isabs(os.path.expanduser(model_name)):
        print(f"\n❌ 本地模型目录不存在: {model_name}")
        return False

    print(f"\n检查模型缓存: {model_name}")

    # Get HuggingFace cache directory
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"   HuggingFace 缓存目录: {hf_cache_dir}")

    try:
        from huggingface_hub import snapshot_download

        cache_path = snapshot_download(model_name, revision='main', local_files_only=True)
        print(f"✅ 模型已在缓存中: {cache_path}")
        return True
    except Exception as e:
        print(f"   缓存检查异常: {e}")

    # Model not in cache, ask user
    print(f"\n⚠️  模型 {model_name} 未在本地缓存中")
    print("   首次使用需要从 HuggingFace 下载（约 2-7GB）")
    response = input("   是否从 HuggingFace 下载？(y/n): ").strip().lower()

    if response != 'y':
        print("❌ 用户选择不下载模型，退出")
        return False

    return True


def extract_movie_text(data_dir, output_dir):
    """Extract movie text information from MovieLens-1M."""
    print("\n" + "=" * 80)
    print("步骤 1: 提取电影文本信息")
    print("=" * 80)

    movies_file = os.path.join(data_dir, 'movies.dat')
    if not os.path.exists(movies_file):
        print(f"❌ 错误：电影数据文件不存在: {movies_file}")
        return None

    print(f"加载电影元数据: {movies_file}")
    mnames = ['movie_id', 'title', 'genres']
    try:
        movies = pd.read_csv(movies_file, sep='::', header=None, names=mnames, engine='python', encoding='ISO-8859-1')
    except Exception as e:
        print(f"❌ 错误：无法读取电影数据文件: {e}")
        return None

    print(f"✅ 加载了 {len(movies)} 部电影")

    movie_text_map = {}
    for idx, row in movies.iterrows():
        movie_id = int(row['movie_id'])
        title = str(row['title']).strip()
        genres = str(row['genres']).strip()
        # Official ByteDance HLLM format:
        # "{item_prompt}title: {title}genres: {genres}"
        # Note: Using 'genres' instead of 'tag' for MovieLens dataset
        text = f"{ITEM_PROMPT}title: {title}genres: {genres}"
        movie_text_map[movie_id] = text

        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1} / {len(movies)} 部电影")

    print(f"✅ 生成了 {len(movie_text_map)} 条文本描述")

    # Save text mapping
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'movie_text_map.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(movie_text_map, f)
    print(f"✅ 文本映射已保存到: {output_file}")

    return movie_text_map


def generate_item_embeddings(model_type, movie_text_map, output_dir, device, vocab_path=None):
    """Generate item embeddings using a pre-trained LLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 80)
    print(f"步骤 2: 生成 Item Embeddings ({model_type})")
    print("=" * 80)

    if vocab_path is None:
        vocab_path = os.path.join(output_dir, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.pkl not found at {vocab_path}. "
                                "Run preprocess_ml_hstu.py first.")

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    if 'item_to_idx' in vocab:
        item_to_idx = vocab['item_to_idx']
    else:
        item_to_idx = {movie_id: token_id for movie_id, token_id in vocab.items() if token_id != 0}
    vocab_size = max(item_to_idx.values()) + 1

    model_name, d_model = get_model_name_and_dim(model_type)

    print(f"加载模型: {model_name}")
    print(f"设备: {device}")
    print(f"vocab_size: {vocab_size}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == 'cuda' else torch.float32, device_map=device, trust_remote_code=True)
    model.eval()

    # Official ByteDance HLLM approach: No special [ITEM] token needed
    # Uses last token's hidden state as item embedding
    print("✅ 使用官方 ByteDance HLLM 格式（最后一个 token 的隐藏状态）")

    # Generate embeddings
    total_items = sum(1 for token_id in item_to_idx.values() if token_id != 0)
    print(f"\n生成 {total_items} 个 item embeddings...")
    os.makedirs(output_dir, exist_ok=True)

    embeddings_array = np.zeros((vocab_size, d_model), dtype=np.float32)
    missing = []
    generated = 0

    with torch.no_grad():
        for movie_id, token_id in tqdm.tqdm(sorted(item_to_idx.items(), key=lambda item: item[1]), desc="Generating embeddings"):
            if token_id == 0:
                continue

            movie_id_int = int(movie_id)
            text = movie_text_map.get(movie_id_int)
            if text is None:
                missing.append((movie_id_int, int(token_id)))
                continue

            # Text already contains ITEM_PROMPT prefix from extract_movie_text()

            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            # Use last token's hidden state as item embedding
            # This matches official implementation where item_emb_token_n=1
            item_emb = hidden_states[0, -1, :].cpu().numpy()

            embeddings_array[token_id] = item_emb
            generated += 1

    # Save embeddings
    embeddings_tensor = torch.from_numpy(embeddings_array).float()
    output_file = os.path.join(output_dir, f'item_embeddings_{model_type}.pt')
    torch.save(embeddings_tensor, output_file)
    if missing:
        print(f"⚠️  {len(missing)} 个 token_id 缺失文本，留零向量: 前 10 个 {missing[:10]}")
    print(f"\n✅ Item embeddings已保存到: {output_file}")
    print(f"   形状: {tuple(embeddings_tensor.shape)} (vocab_size, d_model)")
    print(f"   覆盖: {generated}/{total_items} items")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified HLLM data preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_hllm_data.py --model_type tinyllama --device cuda
  python preprocess_hllm_data.py --model_type baichuan2 --device cuda
        """
    )
    parser.add_argument('--model_type', default='tinyllama', choices=['tinyllama', 'baichuan2'], help='Type of LLM to use')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--data_dir', default=None, help='Directory containing movies.dat')
    parser.add_argument('--output_dir', default=None, help='Directory to save processed files')
    parser.add_argument('--no_download', action='store_true', help='Skip download and use existing local files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing downloaded/extracted files')

    args = parser.parse_args()

    # Use default paths if not provided
    data_dir = args.data_dir if args.data_dir else _DEFAULT_DATA_DIR
    output_dir = args.output_dir if args.output_dir else _DEFAULT_OUTPUT_DIR

    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)

    print(f"\n📂 数据目录: {data_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"下载: {not args.no_download}")
    print(f"覆盖已有文件: {args.overwrite}\n")

    # Download or validate MovieLens-1M raw files before extracting text.
    if not ensure_movielens_data(data_dir, download=not args.no_download, overwrite=args.overwrite):
        sys.exit(1)

    # Check environment
    if not check_environment(args.model_type, args.device):
        sys.exit(1)

    # Check model exists
    if not check_model_exists(args.model_type):
        sys.exit(1)

    # Extract text
    movie_text_map = extract_movie_text(data_dir, output_dir)
    if movie_text_map is None:
        sys.exit(1)

    # Generate embeddings
    generate_item_embeddings(args.model_type, movie_text_map, output_dir, args.device)

    print("\n" + "=" * 80)
    print("✅ HLLM 数据预处理完成！")
    print("=" * 80)
    print("\n输出文件:")
    print(f"  - {os.path.join(output_dir, 'movie_text_map.pkl')}")
    print(f"  - {os.path.join(output_dir, f'item_embeddings_{args.model_type}.pt')}")
    print("\nPipeline order:")
    print("  1. python preprocess_ml_hstu.py")
    print("  2. python preprocess_hllm_data.py --model_type tinyllama --device cuda")


if __name__ == '__main__':
    main()
