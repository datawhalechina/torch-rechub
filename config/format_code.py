import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """运行一个格式化命令，并在失败时退出。"""
    print(f"Running: {description}")
    process = subprocess.Popen(command, text=True, cwd=Path(__file__).parent.parent)
    process.communicate()
    if process.returncode != 0:
        print(f"--- ❌ {description} failed ---", file=sys.stderr)
        sys.exit(1)
    print(f"--- ✅ {description} finished successfully ---")


def main():
    """
    运行一个两段式代码格式化流程:
    1. isort: 整理和排序import语句。
    2. yapf:  应用我们定制的Google风格进行最终排版。
    """
    source_dirs = ["torch_rechub", "examples", "tests"]

    print("========================================")
    print("🚀 启动 isort + yapf (定制版Google风格) 格式化流程...")
    print("========================================")

    # 阶段一: isort
    print("\n--- 阶段一: 使用 isort 排序导入 ---")
    isort_command = [sys.executable, '-m', 'isort', '--profile', 'black'] + source_dirs
    run_command(isort_command, "isort")

    # 阶段二: yapf
    print("\n--- 阶段二: 使用 yapf 应用定制的 Google 风格 ---")
    yapf_style = (
        "{based_on_style: google, "
        "column_limit: 248, "
        "join_multiple_lines: false, "
        "split_all_comma_separated_values: true, "
        "split_before_logical_operator: true, "
        "dedent_closing_brackets: true, "
        "align_closing_bracket_with_visual_indent: true, "
        "indent_width: 4}"
    )
    yapf_command = [
        "yapf",
        "--in-place",
        "--recursive",
        f"--style={yapf_style}",
        *source_dirs
    ]
    run_command(yapf_command, "yapf")

    print("\n\n🎉🎉🎉 所有代码已成功格式化! 🎉🎉🎉")
    sys.exit(0)


if __name__ == "__main__":
    main()