import os
from pathlib import Path

# 常见的编码格式列表，按尝试顺序排列
ENCODING_CANDIDATES = ['utf-8', 'gbk', 'latin-1']


def fix_file_encoding(file_path):
    """
    尝试用多种编码格式读取文件，并将其统一重写为 UTF-8。
    这能修复因编码问题（如GBK）导致的工具崩溃，同时保留中文等合法字符。
    """
    for encoding in ENCODING_CANDIDATES:
        try:
            # 尝试用当前编码格式读取
            content = file_path.read_text(encoding=encoding)

            # 如果不是UTF-8，则转换为UTF-8并重写
            if encoding != 'utf-8':
                print(f"Fixing encoding for: {file_path} (detected {encoding})")
                file_path.write_text(content, encoding='utf-8')
                return True  # 返回已修复

            # 如果本身就是UTF-8且读取成功，则无需操作
            return False  # 返回未修复

        except (UnicodeDecodeError, FileNotFoundError):
            # 如果当前编码格式失败，则继续尝试下一个
            continue

    # 如果所有编码都尝试失败，打印错误信息
    print(f"Error: Could not decode file {file_path} with any of the candidate encodings.")
    return False


def main():
    """
    扫描项目中的.py文件，并将其编码统一修复为UTF-8。
    """
    project_root = Path(__file__).parent.parent
    target_dirs = ["torch_rechub", "examples", "tests"]
    fixed_count = 0

    print("🚀 Starting encoding fix scan...")
    for target_dir in target_dirs:
        for root, _, files in os.walk(project_root / target_dir):
            for file in files:
                if file.endswith(".py"):
                    if fix_file_encoding(Path(root) / file):
                        fixed_count += 1
    
    if fixed_count > 0:
        print(f"\n✅ Successfully fixed encoding for {fixed_count} file(s).")
    else:
        print("\n✅ All files already have correct UTF-8 encoding.")


if __name__ == "__main__":
    main()