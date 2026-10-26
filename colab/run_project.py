"""Run one portfolio project in Google Colab, with Google Drive as the only storage.

Called by ``colab/run_project.ipynb`` after Drive is mounted and the repository is cloned:

    python colab/run_project.py --project p06 --profile free \
        --drive /content/drive/MyDrive/portfolio

Steps, all on the local disk of the Colab machine except the copies to and from Drive:

1. Install the project and the local packages its Makefile installs, with the pinned
   dependencies, except the packages that Colab provides in its own build (torch, torchvision).
2. Copy the raw files from ``<drive>/raw/<project folder>/`` to ``data/raw/`` of the project,
   and check them against ``colab/raw_files.yaml``: files with other names or inside a zip are
   moved or extracted to the expected place. When a file is still missing, the run stops before
   the install, and the list of missing files is shown and written to
   ``<drive>/out/<project folder>/missing_files.txt``.
3. With ``--resume``, copy the saved outputs from ``<drive>/out/<project folder>/`` back.
4. Run the Makefile target of the profile, with the project ``colab.yaml`` (if any) in
   ``PORTFOLIO_CONFIG``, while a background thread copies outputs to Drive every few minutes.
5. Copy the outputs to Drive a last time, also when the run fails.

The script uses the standard library and PyYAML only, so that it runs before the install.
"""

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROFILES_DIR = REPO / "colab" / "profiles"
PROVIDED = {"torch", "torchvision"}  # Colab's own CUDA builds stay in place
OUTPUTS = ("artifacts", "reports")
PROFILE_KEYS = {"target", "sample", "mirror_minutes", "mirror_exclude", "projects"}
PROJECT_KEYS = {"target", "sample"}
RAW_FILES = REPO / "colab" / "raw_files.yaml"
ENTRY_KEYS = {"file", "folder", "kind", "min", "also", "inside", "extract", "source"}
MISSING_NAME = "missing_files.txt"
PARTIAL = ".part"  # suffix of a download in progress; never used
NO_DATA_TARGETS = {"install", "lint", "test", "smoke", "clean"}  # targets that read no raw files


def project_dir(name: str) -> Path:
    """The folder of a project, from its number (``p06``) or its full folder name."""
    matches = sorted((REPO / "projects").glob(f"{name}*"))
    if len(matches) != 1:
        raise SystemExit(f"no single project matches {name!r}: {[m.name for m in matches]}")
    return matches[0]


def read_profile(name: str) -> dict:
    import yaml

    path = PROFILES_DIR / f"{name}.yaml"
    if not path.is_file():
        names = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))
        raise SystemExit(f"no profile {name!r}; choose from {names}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    unknown = sorted(set(data) - PROFILE_KEYS)
    for key, block in (data.get("projects") or {}).items():
        unknown += [f"projects.{key}.{k}" for k in sorted(set(block or {}) - PROJECT_KEYS)]
    if unknown:
        raise SystemExit(f"{path}: unknown keys {unknown}")
    return data


def for_project(profile: dict, project: Path) -> dict:
    """The profile values, with the entry of the project (by its number, p06) on top."""
    own = (profile.get("projects") or {}).get(project.name[:3]) or {}
    return {**{k: v for k, v in profile.items() if k != "projects"}, **own}


def local_packages(project: Path) -> list[Path]:
    """The folders that the ``install`` target of the project Makefile installs with ``-e``."""
    text = (project / "Makefile").read_text(encoding="utf-8")
    block = re.search(r"^install:.*?\n((?:\t.*\n?)+)", text, flags=re.M)
    if block is None:
        return [REPO / "common", project]
    found = re.findall(r"-e\s+\"?([^\s\"\[]+)", block.group(1).replace("\\\n", " "))
    return [(project / p).resolve() for p in found]


def requirements(packages: list[Path]) -> list[str]:
    """Pinned dependencies of the packages, without the local ones and the ones Colab provides."""
    local = set()
    pins: dict[str, str] = {}
    for folder in packages:
        with (folder / "pyproject.toml").open("rb") as handle:
            meta = tomllib.load(handle)["project"]
        local.add(meta["name"].lower().replace("_", "-"))
        for spec in meta.get("dependencies", []):
            name = re.split(r"[\s<>=!~\[;]", spec, maxsplit=1)[0].lower().replace("_", "-")
            pins.setdefault(name, spec)
    return [spec for name, spec in pins.items() if name not in local and name not in PROVIDED]


def install(project: Path) -> None:
    packages = local_packages(project)
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run([*pip, *requirements(packages)], check=True)
    editable = [arg for folder in packages for arg in ("-e", str(folder))]
    subprocess.run([*pip, "--no-deps", *editable], check=True)


def copy_tree(source: Path, target: Path, exclude: tuple[str, ...] = ()) -> int:
    """Copy files whose size or time differ; skip any path with a part named in ``exclude``."""
    count = 0
    if not source.is_dir():
        return count
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or any(part in exclude for part in relative.parts):
            continue
        out = target / relative
        stat = path.stat()
        if (
            out.is_file()
            and out.stat().st_size == stat.st_size
            and out.stat().st_mtime >= stat.st_mtime
        ):
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        count += 1
    return count


def read_raw_files(project: Path) -> list:
    """The entries of ``raw_files.yaml`` for a project (by its number, p06)."""
    import yaml

    with RAW_FILES.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get(project.name[:3]) or []


def _matches(raw: Path, pattern: str) -> list[Path]:
    return sorted(p for p in raw.glob(pattern) if p.is_file())


def _move(path: Path, out: Path, raw: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    path.rename(out)
    print(f"[raw] moved {path.relative_to(raw)} to {out.relative_to(raw)}", flush=True)


def _extract(raw: Path, archives: list[Path], entry: dict, kind: str, out_name: str | None) -> int:
    """Extract the members whose name matches ``kind`` and whose path matches ``inside``.

    With ``out_name`` the first match is written under that name; otherwise every match keeps
    its own name. Returns the number of files written.
    """
    target_dir = (raw / entry["file"]).parent
    inside = entry.get("inside", "*")
    count = 0
    for archive in archives:
        try:
            handle = zipfile.ZipFile(archive)
        except zipfile.BadZipFile:
            continue
        with handle:
            for member in handle.infolist():
                base = Path(member.filename).name
                if member.is_dir() or not fnmatch.fnmatch(base, kind):
                    continue
                if not fnmatch.fnmatch(member.filename, inside):
                    continue
                out = target_dir / (out_name or base)
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, out.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                print(f"[raw] extracted {member.filename} from {archive.name}", flush=True)
                count += 1
                if out_name:
                    return count
    return count


def _members(archives: list[Path], kind: str, inside: str) -> list[str]:
    found = []
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as handle:
                found += [
                    m.filename
                    for m in handle.infolist()
                    if not m.is_dir()
                    and fnmatch.fnmatch(Path(m.filename).name, kind)
                    and fnmatch.fnmatch(m.filename, inside)
                ]
        except zipfile.BadZipFile:
            continue
    return found


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix != PARTIAL)


def _from_folder(raw: Path, entry: dict) -> str | None:
    """Take the data from the subfolder of its kind (``folder``), whatever the file names.

    A folder for one file must hold exactly one file of the kind (or one zip that holds it);
    otherwise the reason is returned. A folder for many files gives all files of the kind.
    """
    folder = raw / entry.get("folder", "")
    if "folder" not in entry or not folder.is_dir():
        return None
    pattern = entry["file"]
    name = Path(pattern).name
    wildcard = any(c in name for c in "*?[")
    target_dir = (raw / pattern).parent
    if wildcard:
        if folder != target_dir:
            for path in _files(folder):
                if fnmatch.fnmatch(path.name, name) and not (target_dir / path.name).exists():
                    _move(path, target_dir / path.name, raw)
        if len(_matches(raw, pattern)) < entry.get("min", 1) and entry.get("extract", True):
            _extract(raw, [p for p in _files(folder) if p.suffix == ".zip"], entry, name, None)
        return None
    if (raw / pattern).is_file():
        return None
    kind = entry.get("kind", "*" + Path(name).suffix)
    found = [p for p in _files(folder) if fnmatch.fnmatch(p.name, kind)]
    if len(found) == 1:
        _move(found[0], raw / pattern, raw)
        return None
    shown = f"{entry['folder']}/"
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        return f"{shown} must hold one {kind} file, found {len(found)}: {names}"
    zips = [p for p in _files(folder) if p.suffix == ".zip"]
    # In a zip: the expected name first; else any member of the kind, when there is only one.
    if (
        entry.get("extract", True)
        and not _extract(raw, zips, entry, name, name)
        and len(_members(zips, kind, entry.get("inside", "*"))) == 1
    ):
        _extract(raw, zips, entry, kind, name)
    return None  # when nothing was found here, the other rules and the missing list apply


def _place(raw: Path, entry: dict) -> None:
    """Move files with other names or places, then extract zip members, to the expected place."""
    pattern = entry["file"]
    target_dir = (raw / pattern).parent
    name = Path(pattern).name
    wildcard = any(c in name for c in "*?[")
    expected = set(_matches(raw, pattern))
    names = list(entry.get("also", [])) + ([] if wildcard else [name])
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path in expected or path.suffix == PARTIAL:
            continue
        if any(fnmatch.fnmatch(path.name, n) for n in names):
            out = target_dir / (path.name if wildcard else name)
            if out.exists():
                continue
            _move(path, out, raw)
            if not wildcard:
                return
    if len(_matches(raw, pattern)) >= entry.get("min", 1) or not entry.get("extract", True):
        return
    _extract(raw, sorted(raw.rglob("*.zip")), entry, name, None if wildcard else name)


def _complete(raw: Path, entries: list) -> bool:
    return all(
        _complete_one_of(raw, e["one_of"])
        if "one_of" in e
        else len(_matches(raw, e["file"])) >= e.get("min", 1)
        for e in entries
    )


def _complete_one_of(raw: Path, options: list) -> bool:
    return any(_complete(raw, option) for option in options)


def _describe(entry: dict, raw: Path) -> str:
    found = len(_matches(raw, entry["file"]))
    count = f" (at least {entry['min']}, found {found})" if entry.get("min", 1) > 1 else ""
    where = f"{entry['folder']}/ (any file name)" if "folder" in entry else entry["file"]
    return f"{where}{count}\n      from: {entry.get('source', 'see colab/README.md')}"


def check_raw(raw: Path, entries: list) -> list[str]:
    """Place the raw files of a project; return the lines that describe the missing ones."""
    missing = []
    for entry in entries:
        if "one_of" in entry:
            for option in entry["one_of"]:
                for item in option:
                    _from_folder(raw, item)
                    _place(raw, item)
                if _complete(raw, option):
                    break
            else:
                missing.append("one of these sets:")
                for i, option in enumerate(entry["one_of"], start=1):
                    missing += [f"  set {i}: {_describe(item, raw)}" for item in option]
            continue
        unknown = sorted(set(entry) - ENTRY_KEYS)
        if unknown:
            raise SystemExit(f"{RAW_FILES}: unknown keys {unknown} in {entry}")
        problem = _from_folder(raw, entry)
        if problem:
            missing.append(problem)
            continue
        _place(raw, entry)
        if len(_matches(raw, entry["file"])) < entry.get("min", 1):
            missing.append(_describe(entry, raw))
    return missing


def report_missing(project: Path, missing: list[str], drive_raw: Path, drive_out: Path) -> None:
    """Show the missing files and keep the list on Drive; remove an old list when none is."""
    note = drive_out / MISSING_NAME
    if not missing:
        if note.is_file():
            note.unlink()
        return
    text = "\n".join(
        [
            f"{project.name}: raw files missing. Upload them to {drive_raw}/",
            "(names may differ, and zips that hold them are opened), then run again:",
            "",
            *[f"  - {line}" if not line.startswith("  ") else line for line in missing],
            "",
        ]
    )
    drive_out.mkdir(parents=True, exist_ok=True)
    note.write_text(text, encoding="utf-8")
    print(text + f"\nThis list is also in {note}", flush=True)
    raise SystemExit(2)


class Mirror:
    """Copies the outputs of a project to Drive every ``minutes`` and once more at the end."""

    def __init__(self, project: Path, drive_out: Path, minutes: float, exclude: tuple[str, ...]):
        self.project, self.drive_out, self.exclude = project, drive_out, exclude
        self.seconds = max(60.0, minutes * 60)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def copy(self) -> int:
        return sum(
            copy_tree(self.project / name, self.drive_out / name, self.exclude) for name in OUTPUTS
        )

    def _loop(self) -> None:
        while not self.stop.wait(self.seconds):
            try:
                print(f"[mirror] {self.copy()} files copied to Drive", flush=True)
            except OSError as error:  # Drive may be slow or briefly unavailable; try again later
                print(f"[mirror] copy failed, retrying later: {error}", flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        print(f"[mirror] final copy: {self.copy()} files copied to {self.drive_out}", flush=True)
        return False


def make_command(project: Path, target: str, sample: float | None) -> list[str]:
    """``target`` may name several Makefile targets, separated by spaces."""
    command = ["make", "-C", str(project), *target.split(), f"PYTHON={sys.executable}"]
    if sample is not None:
        command.append(f"SAMPLE={sample}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True, help="number (p06) or folder name")
    parser.add_argument("--profile", default="free", help="a file name in colab/profiles")
    parser.add_argument("--drive", type=Path, default=Path("/content/drive/MyDrive/portfolio"))
    parser.add_argument("--target", help="Makefile target; overrides the profile")
    parser.add_argument("--sample", type=float, help="share of the data; overrides the profile")
    parser.add_argument("--resume", action="store_true", help="copy saved outputs back first")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    project = project_dir(args.project)
    profile = for_project(read_profile(args.profile), project)
    target = args.target or profile.get("target", "sample")
    sample = args.sample if args.sample is not None else profile.get("sample")
    drive_raw = args.drive / "raw" / project.name
    drive_out = args.drive / "out" / project.name

    raw = project / "data" / "raw"
    print(f"Raw files from {drive_raw}:", copy_tree(drive_raw, raw), flush=True)
    if not set(target.split()) <= NO_DATA_TARGETS:
        report_missing(project, check_raw(raw, read_raw_files(project)), drive_raw, drive_out)
    if not args.skip_install:
        print("Installing", project.name, flush=True)
        install(project)
    if args.resume:
        restored = sum(copy_tree(drive_out / n, project / n) for n in OUTPUTS)
        print(f"Restored {restored} files from {drive_out}", flush=True)

    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
    overrides = project / "colab.yaml"
    if overrides.is_file():
        env["PORTFOLIO_CONFIG"] = os.pathsep.join(
            p for p in (env.get("PORTFOLIO_CONFIG", ""), str(overrides)) if p
        )
    command = make_command(project, target, sample)
    print("Running", " ".join(command), flush=True)
    exclude = tuple(profile.get("mirror_exclude", ()))
    with Mirror(project, drive_out, float(profile.get("mirror_minutes", 10)), exclude):
        result = subprocess.run(command, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
