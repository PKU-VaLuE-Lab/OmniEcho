import ast
import io
from pathlib import Path
import re
import subprocess
import sys
import tokenize


def check(root):
    errors = []
    count = 0
    chinese = re.compile(r"[\u3400-\u9fff]")
    private = re.compile(r"/(?:cpfs\w*|home/xj_share_data_new)/|" +
                         r"https?://[^\s\"']*(?:ali" + r"baba|ali" + r"yuncs|model" + r"router)", re.I)
    secret = re.compile(r"\bsk" + r"-[A-Za-z0-9_-]{16,}")
    allowed = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".toml", ".example"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            errors.append(f"{relative}: symlinks are not allowed in the source distribution")
            continue
        if path.is_dir():
            if path.name in {".git", "__pycache__", ".pytest_cache"}:
                errors.append(f"{relative}: history/cache directory")
            continue
        if path.suffix not in allowed and path.name not in {".gitignore", "LICENSE", "NOTICE"}:
            errors.append(f"{relative}: non-code asset or unsupported file type")
            continue
        count += 1
        text = path.read_text(encoding="utf-8")
        if private.search(text) or secret.search(text):
            errors.append(f"{relative}: private endpoint/path or embedded credential")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(relative))
                for token in tokenize.generate_tokens(io.StringIO(text).readline):
                    if token.type == tokenize.COMMENT and chinese.search(token.string):
                        errors.append(f"{relative}:{token.start[0]}: Chinese comment")
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        doc = ast.get_docstring(node)
                        if doc and chinese.search(doc):
                            errors.append(f"{relative}: Chinese docstring")
            except (SyntaxError, tokenize.TokenError) as error:
                errors.append(f"{relative}: {error}")
        elif path.suffix == ".sh":
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f"{relative}: {result.stderr.strip()}")
            for number, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("#") and chinese.search(line):
                    errors.append(f"{relative}:{number}: Chinese comment")
    return count, errors


def main():
    root = Path(__file__).resolve().parents[1]
    count, errors = check(root)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {count} source/config/documentation files; {len(errors)} errors.")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
