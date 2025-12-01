"""代码格式化和质量检查工具

运行流程:
1. isort: 整理和排序 import 语句
2. yapf: 应用定制的 Google 风格进行代码格式化
3. flake8: 代码质量检查（包括 F541 等错误）

使用方法:
    python config/format_code.py

注意：F541 错误（f-string 无占位符）需要手动修复，将 f"text" 改为 "text"
"""

import io
import subprocess
import sys
from pathlib import Path

# Windows UTF-8 编码支持
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT_DIR = Path(__file__).parent.parent
SOURCE_DIRS = ["torch_rechub", "examples", "tests"]

YAPF_STYLE = (
    "{based_on_style: google, column_limit: 248, join_multiple_lines: false, "
    "split_all_comma_separated_values: true, split_before_logical_operator: true, "
    "dedent_closing_brackets: true, align_closing_bracket_with_visual_indent: true, "
    "indent_width: 4}"
)

FLAKE8_IGNORE = (
    "E203,W503,E501,E722,E402,F821,F523,E711,E741,F401,"
    "E265,C901,E301,E305,W293,E261,W291,W292,E111,E117,F841,E302"
)


def run_command(command, description, exit_on_error=True):
    """运行命令并返回是否成功"""
    result = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True)
    success = result.returncode == 0
    status = "OK" if success else "FAILED"
    print(f"  [{status}] {description}")
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    if not success and exit_on_error:
        sys.exit(1)
    return success


def main():
    print("=" * 50)
    print("代码格式化和质量检查")
    print("=" * 50)

    # 阶段 1: isort
    print("\n[阶段 1] isort 排序导入")
    run_command([sys.executable, "-m", "isort", "--profile", "black"] + SOURCE_DIRS, "isort")

    # 阶段 2: yapf
    print("\n[阶段 2] yapf 代码格式化")
    run_command(["yapf", "--in-place", "--recursive", f"--style={YAPF_STYLE}"] + SOURCE_DIRS, "yapf")

    # 阶段 3: flake8
    print("\n[阶段 3] flake8 代码质量检查")
    flake8_ok = run_command(
        ["flake8", "--max-line-length=248", f"--extend-ignore={FLAKE8_IGNORE}", "--max-complexity=30"] + SOURCE_DIRS,
        "flake8",
        exit_on_error=False
    )

    # 结果
    print("\n" + "=" * 50)
    if flake8_ok:
        print("所有检查通过!")
        sys.exit(0)
    else:
        print("flake8 检查发现问题，请修复后再提交")
        sys.exit(1)


if __name__ == "__main__":
    main()