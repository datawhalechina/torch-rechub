import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = PROJECT_ROOT / "torch_rechub"
OUTPUT_FILES = {
    "en": PROJECT_ROOT / "docs" / "en" / "api" / "api.md",
    "zh": PROJECT_ROOT / "docs" / "zh" / "api" / "api.md",
}

LOCALES = {
    "en": {
        "title": "torch-rechub API Reference",
        "generated_note": "> Generated from AST by scanning `torch_rechub/`. Files named `__init__.py` are excluded.",
        "module_label": "Module",
        "no_docstring": "No docstring provided.",
        "sections": {
            "Parameters": "Parameters",
            "Returns": "Returns",
            "Raises": "Raises",
            "Shape": "Shape",
            "Notes": "Notes",
            "Examples": "Examples",
            "Methods": "Methods",
            "Attributes": "Attributes",
        },
    },
    "zh": {
        "title": "torch-rechub API \u6587\u6863",
        "generated_note": "> \u57fa\u4e8e AST \u626b\u63cf `torch_rechub/` \u81ea\u52a8\u751f\u6210\u3002\u5df2\u6392\u9664 `__init__.py` \u6587\u4ef6\u3002",
        "module_label": "\u6a21\u5757",
        "no_docstring": "\u672a\u63d0\u4f9b\u6587\u6863\u8bf4\u660e\u3002",
        "sections": {
            "Parameters": "\u53c2\u6570",
            "Returns": "\u8fd4\u56de",
            "Raises": "\u5f02\u5e38",
            "Shape": "\u5f20\u91cf\u5f62\u72b6",
            "Notes": "\u8bf4\u660e",
            "Examples": "\u793a\u4f8b",
            "Methods": "\u65b9\u6cd5",
            "Attributes": "\u5c5e\u6027",
        },
    },
}

SECTION_ALIASES = {
    "arg": "Parameters",
    "args": "Parameters",
    "argument": "Parameters",
    "arguments": "Parameters",
    "param": "Parameters",
    "params": "Parameters",
    "parameter": "Parameters",
    "parameters": "Parameters",
    "return": "Returns",
    "returns": "Returns",
    "yield": "Returns",
    "yields": "Returns",
    "raise": "Raises",
    "raises": "Raises",
    "shape": "Shape",
    "shapes": "Shape",
    "note": "Notes",
    "notes": "Notes",
    "example": "Examples",
    "examples": "Examples",
    "method": "Methods",
    "methods": "Methods",
    "attribute": "Attributes",
    "attributes": "Attributes",
}

PARAM_RE = re.compile(r"^(\*{0,2}[\w.\-]+)\s*\(([^)]*)\)\s*:\s*(.*)$")
NUMPY_RE = re.compile(r"^(\*{0,2}[\w.\-]+)\s*:\s*(.+)$")


@dataclass
class FunctionDoc:
    name: str
    signature: str
    doc: str


@dataclass
class MethodDoc:
    name: str
    signature: str
    doc: str


@dataclass
class ClassDoc:
    name: str
    doc: str
    methods: list[MethodDoc] = field(default_factory=list)


@dataclass
class FileDoc:
    path: Path
    module: str
    functions: list[FunctionDoc] = field(default_factory=list)
    classes: list[ClassDoc] = field(default_factory=list)


def get_full_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = get_full_name(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    if isinstance(node, ast.Subscript):
        value = get_full_name(node.value)
        slice_value = get_full_name(node.slice)
        return f"{value}[{slice_value}]" if slice_value else value
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Tuple):
        return ", ".join(filter(None, (get_full_name(elt) for elt in node.elts)))
    if isinstance(node, ast.List):
        return "[" + ", ".join(filter(None, (get_full_name(elt) for elt in node.elts))) + "]"
    if isinstance(node, ast.Dict):
        pairs = []
        for key, value in zip(node.keys, node.values):
            key_text = get_full_name(key)
            value_text = get_full_name(value)
            if key_text and value_text:
                pairs.append(f"{key_text}: {value_text}")
        return "{" + ", ".join(pairs) + "}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = get_full_name(node.left)
        right = get_full_name(node.right)
        if left and right:
            return f"{left} | {right}"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = get_full_name(node.operand)
        return f"-{operand}" if operand else None
    if hasattr(ast, "Index") and isinstance(node, ast.Index):
        return get_full_name(node.value)
    return None


def clean_docstring(doc: str | None) -> str:
    return (doc or "__NO_DOCSTRING__").strip()


def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def normalize_section_name(name: str) -> str | None:
    key = name.strip().rstrip(":").lower()
    return SECTION_ALIASES.get(key)


def is_section_heading(lines: list[str], index: int) -> tuple[str | None, int]:
    stripped = lines[index].strip()
    canonical = normalize_section_name(stripped)
    if canonical and stripped.endswith(":"):
        return canonical, 1
    if index + 1 >= len(lines):
        return None, 0
    underline = lines[index + 1].strip()
    canonical = normalize_section_name(stripped)
    if canonical and underline and set(underline) <= {"-", "="} and len(underline) >= len(stripped):
        return canonical, 2
    return None, 0


def split_sections(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    i = 0

    while i < len(lines):
        title, consumed = is_section_heading(lines, i)
        if title:
            if current_lines or current_title is not None:
                sections.append((current_title, current_lines))
            current_title = title
            current_lines = []
            i += consumed
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        current_lines.append(lines[i])
        i += 1

    if current_lines or current_title is not None:
        sections.append((current_title, current_lines))

    return sections


def sanitize_examples_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return f"**{stripped.lstrip('#').strip()}**"
    return line


def format_freeform_lines(lines: list[str], *, in_examples: bool = False) -> list[str]:
    formatted: list[str] = []
    in_doctest = False
    i = 0

    def close_doctest() -> None:
        nonlocal in_doctest
        if in_doctest:
            formatted.append("```")
            in_doctest = False

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        is_doctest_line = stripped.startswith(">>>") or stripped.startswith("...")
        is_doctest_blank = in_doctest and stripped == ""

        if is_doctest_line:
            if not in_doctest:
                formatted.append("```python")
                in_doctest = True
            formatted.append(stripped)
            i += 1
            continue

        if is_doctest_blank:
            formatted.append("")
            i += 1
            continue

        close_doctest()
        formatted.append(sanitize_examples_line(line) if in_examples else line)
        i += 1

    close_doctest()
    return formatted


def dedent_block(lines: list[str]) -> list[str]:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return []
    min_indent = min(leading_spaces(line) for line in non_empty)
    return [line[min_indent:] if line.strip() else "" for line in lines]


def collect_indented_block(lines: list[str], start: int, base_indent: int) -> tuple[list[str], int]:
    collected: list[str] = []
    i = start

    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            collected.append("")
            i += 1
            continue
        indent = leading_spaces(line)
        if indent <= base_indent:
            break
        collected.append(line)
        i += 1

    return dedent_block(collected), i


def parse_param_like_section(lines: list[str], allow_type_only: bool = False) -> tuple[list[dict[str, object]], list[str]]:
    items: list[dict[str, object]] = []
    remainder: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        indent = leading_spaces(line)
        label = None
        type_text = None
        desc_lines: list[str] = []

        match = PARAM_RE.match(stripped)
        if match:
            label, type_text, inline_desc = match.groups()
            if inline_desc:
                desc_lines.append(inline_desc)
        else:
            match = NUMPY_RE.match(stripped)
            if match:
                label, type_text = match.groups()
            elif allow_type_only:
                label = stripped
            else:
                remainder.append(line)
                i += 1
                continue

        block_lines, next_index = collect_indented_block(lines, i + 1, indent)
        if block_lines:
            desc_lines.extend(block_lines)

        items.append({"label": label, "type": type_text, "desc": desc_lines})
        i = next_index

    return items, remainder


def parse_shape_section(lines: list[str]) -> tuple[list[dict[str, object]], list[str]]:
    items: list[dict[str, object]] = []
    remainder: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        indent = leading_spaces(line)
        if indent > 0:
            remainder.append(line)
            i += 1
            continue

        body_lines, next_index = collect_indented_block(lines, i + 1, indent)
        if body_lines:
            body_items, body_remainder = parse_param_like_section(body_lines, allow_type_only=False)
            items.append({"label": stripped, "items": body_items, "desc": body_remainder})
        else:
            remainder.append(line)
        i = next_index if next_index > i else i + 1

    return items, remainder


def render_item(label: str | None, type_text: str | None, desc_lines: list[str], indent: str = "") -> list[str]:
    lines: list[str] = []
    head = f"{indent}- "

    if label and type_text:
        head += f"`{label}` (`{type_text}`)"
    elif label:
        head += f"`{label}`"
    elif type_text:
        head += f"`{type_text}`"

    if desc_lines:
        first_line = desc_lines[0].strip()
        if label or type_text:
            if first_line:
                head += f": {first_line}"
        else:
            head += first_line
        lines.append(head.rstrip())
        for extra_line in desc_lines[1:]:
            if extra_line.strip():
                lines.append(f"{indent}  {extra_line.strip()}")
            else:
                lines.append("")
    else:
        lines.append(head.rstrip())

    return lines


def render_shape_item(item: dict[str, object]) -> list[str]:
    lines = [f"- **{item['label']}**"]
    for sub_item in item["items"]:
        lines.extend(render_item(sub_item["label"], sub_item["type"], sub_item["desc"], indent="  "))
    for raw_line in format_freeform_lines(item["desc"]):
        if raw_line:
            lines.append(f"  {raw_line}")
        else:
            lines.append("")
    return lines


def render_section(title: str, content_lines: list[str], locale: str) -> list[str]:
    title_text = LOCALES[locale]["sections"][title]
    rendered = [f"**{title_text}**", ""]

    if title == "Parameters":
        items, remainder = parse_param_like_section(content_lines, allow_type_only=False)
        if items:
            for item in items:
                rendered.extend(render_item(item["label"], item["type"], item["desc"]))
            if remainder:
                rendered.append("")
                rendered.extend(format_freeform_lines(remainder))
            return rendered

    if title in {"Returns", "Raises", "Attributes", "Methods"}:
        items, remainder = parse_param_like_section(content_lines, allow_type_only=True)
        if items:
            for item in items:
                rendered.extend(render_item(item["label"], item["type"], item["desc"]))
            if remainder:
                rendered.append("")
                rendered.extend(format_freeform_lines(remainder))
            return rendered

    if title == "Shape":
        items, remainder = parse_shape_section(content_lines)
        if items:
            for item in items:
                rendered.extend(render_shape_item(item))
            if remainder:
                rendered.append("")
                rendered.extend(format_freeform_lines(remainder))
            return rendered

    rendered.extend(format_freeform_lines(content_lines, in_examples=(title == "Examples")))
    return rendered


def format_docstring(doc: str, locale: str) -> str:
    if doc == "__NO_DOCSTRING__":
        return LOCALES[locale]["no_docstring"]

    sections = split_sections(doc.splitlines())
    rendered: list[str] = []

    for title, content_lines in sections:
        if rendered and rendered[-1] != "":
            rendered.append("")
        if title is None:
            rendered.extend(format_freeform_lines(content_lines))
        else:
            rendered.extend(render_section(title, content_lines, locale))

    return "\n".join(rendered).strip()


def format_arg(arg: ast.arg, default: ast.AST | None = None) -> str:
    text = arg.arg
    annotation = get_full_name(arg.annotation)
    if annotation:
        text += f": {annotation}"
    if default is not None:
        default_text = get_full_name(default) or "..."
        text += f" = {default_text}"
    return text


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    positional_only = getattr(args, "posonlyargs", [])
    positional = list(positional_only) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    if positional_only:
        for arg, default in zip(positional_only, defaults[: len(positional_only)]):
            parts.append(format_arg(arg, default))
        parts.append("/")

    for arg, default in zip(args.args, defaults[len(positional_only) :]):
        parts.append(format_arg(arg, default))

    if args.vararg:
        parts.append(f"*{format_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(format_arg(arg, default))

    if args.kwarg:
        parts.append(f"**{format_arg(args.kwarg)}")

    signature = f"({', '.join(parts)})"
    returns = get_full_name(node.returns)
    if returns:
        signature += f" -> {returns}"
    return signature


def is_public_name(name: str) -> bool:
    return not name.startswith("_")


def module_name_from_path(py_file: Path) -> str:
    rel = py_file.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(rel.parts)


def iter_python_files(package_dir: Path) -> Iterable[Path]:
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name == "__init__.py":
            continue
        yield path


def extract_file_doc(py_file: Path) -> FileDoc | None:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    file_doc = FileDoc(path=py_file, module=module_name_from_path(py_file))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public_name(node.name):
            file_doc.functions.append(
                FunctionDoc(
                    name=node.name,
                    signature=format_signature(node),
                    doc=clean_docstring(ast.get_docstring(node)),
                )
            )
        elif isinstance(node, ast.ClassDef) and is_public_name(node.name):
            methods: list[MethodDoc] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_public_name(item.name):
                    methods.append(
                        MethodDoc(
                            name=item.name,
                            signature=format_signature(item),
                            doc=clean_docstring(ast.get_docstring(item)),
                        )
                    )
            file_doc.classes.append(ClassDoc(name=node.name, doc=clean_docstring(ast.get_docstring(node)), methods=methods))

    if not file_doc.functions and not file_doc.classes:
        return None
    return file_doc


def heading(level: int, title: str) -> str:
    return f"{'#' * max(1, min(level, 6))} {title}"


def format_path_label(parts: tuple[str, ...], is_file: bool) -> str:
    label = parts[-1]
    return f"`{label}`" if is_file else f"`{label}/`"


def render_function(lines: list[str], level: int, func: FunctionDoc, locale: str) -> None:
    lines.append(heading(level, f"`{func.name}`"))
    lines.append("")
    lines.append(f"```python\n{func.name}{func.signature}\n```")
    lines.append("")
    lines.append(format_docstring(func.doc, locale))
    lines.append("")


def render_class(lines: list[str], level: int, cls: ClassDoc, locale: str) -> None:
    lines.append(heading(level, f"`{cls.name}`"))
    lines.append("")
    lines.append(format_docstring(cls.doc, locale))
    lines.append("")
    for method in cls.methods:
        lines.append(heading(level + 1, f"`{cls.name}.{method.name}`"))
        lines.append("")
        lines.append(f"```python\n{method.name}{method.signature}\n```")
        lines.append("")
        lines.append(format_docstring(method.doc, locale))
        lines.append("")


def generate_markdown(package_dir: Path, locale: str) -> str:
    locale_text = LOCALES[locale]
    file_docs = [doc for path in iter_python_files(package_dir) if (doc := extract_file_doc(path))]

    lines: list[str] = [
        f"# {locale_text['title']}",
        "",
        locale_text["generated_note"],
        "",
    ]

    rendered_sections: set[tuple[str, ...]] = set()

    for file_doc in file_docs:
        rel_parts = file_doc.path.relative_to(package_dir).with_suffix("").parts
        dir_parts = rel_parts[:-1]

        for depth in range(1, len(dir_parts) + 1):
            section = dir_parts[:depth]
            if section in rendered_sections:
                continue
            lines.append(heading(depth + 1, format_path_label(section, is_file=False)))
            lines.append("")
            rendered_sections.add(section)

        file_level = len(dir_parts) + 2
        lines.append(heading(file_level, format_path_label(rel_parts, is_file=True)))
        lines.append("")
        lines.append(f"{locale_text['module_label']}: `{file_doc.module}`")
        lines.append("")

        entity_level = file_level + 1
        for func in file_doc.functions:
            render_function(lines, entity_level, func, locale)
        for cls in file_doc.classes:
            render_class(lines, entity_level, cls, locale)

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if not PACKAGE_DIR.exists():
        raise FileNotFoundError(f"Package directory not found: {PACKAGE_DIR}")

    for locale, output_file in OUTPUT_FILES.items():
        markdown = generate_markdown(PACKAGE_DIR, locale)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8")
        print(f"API markdown generated [{locale}]: {output_file}")


if __name__ == "__main__":
    main()
