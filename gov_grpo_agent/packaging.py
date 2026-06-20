import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


INCLUDE_PATHS = [
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "docs",
    "gov_grpo_agent",
    "tests",
]

EXCLUDED_PARTS = {
    ".git",
    ".agents",
    ".codex",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "dist",
}

EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def build_server_package(root_dir, output_path):
    root = Path(root_dir).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for relative in _iter_package_files(root):
            archive.write(root / relative, relative.as_posix())
    return output


def _iter_package_files(root):
    for include in INCLUDE_PATHS:
        path = root / include
        if not path.exists():
            continue
        if path.is_file():
            relative = path.relative_to(root)
            if not _is_excluded(relative):
                yield relative
            continue
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                relative = file_path.relative_to(root)
                if not _is_excluded(relative):
                    yield relative


def _is_excluded(relative_path):
    if any(part in EXCLUDED_PARTS for part in relative_path.parts):
        return True
    return relative_path.suffix in EXCLUDED_SUFFIXES


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a server-uploadable source bundle.")
    parser.add_argument("--root-dir", default=".", help="Project root to package.")
    parser.add_argument(
        "--output",
        default="dist/gov_grpo_agent_server_bundle.zip",
        help="Output zip path.",
    )
    args = parser.parse_args(argv)
    package_path = build_server_package(args.root_dir, args.output)
    print(package_path)


if __name__ == "__main__":
    main()
