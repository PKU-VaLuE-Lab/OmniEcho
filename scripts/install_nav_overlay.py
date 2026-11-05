import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description="Install OmniEcho code into an external InternNav checkout.")
    parser.add_argument("internnav", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace differing overlay files; no backups are made.")
    parser.add_argument("--check", action="store_true", help="Validate layout and conflicts without writing.")
    args = parser.parse_args()
    destination = args.internnav.resolve()
    required = ["internnav/agent/base.py", "internnav/evaluator/__init__.py",
                "internnav/habitat_extensions/vln/measures.py"]
    for relative in required:
        if not (destination / relative).is_file():
            parser.error(f"Missing required InternNav module: {relative}")
    source = Path(__file__).resolve().parents[1] / "nav"
    files = [path for folder in ("scripts", "internnav")
             for path in (source / folder).rglob("*")
             if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}]
    conflicts = [str(path.relative_to(source)) for path in files
                 if (destination / path.relative_to(source)).exists()
                 and (destination / path.relative_to(source)).read_bytes() != path.read_bytes()]
    if conflicts and not args.force:
        parser.error("Differing files exist; review before using --force: " + ", ".join(conflicts))
    if args.check:
        print(f"Layout checked: {len(files)} source/config files; {len(conflicts)} replacements.")
        return
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print(f"Installed {len(files)} source/config files. No data, weights, or histories were copied.")


if __name__ == "__main__":
    main()
