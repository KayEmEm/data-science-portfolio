"""YAML control interface: the settings of every command and settings class of a project.

Each project keeps one ``config.yaml`` in its folder. A top level key names a command module
(``train`` for ``python -m <package>.train``) and holds the values of its options, by option
name with underscores (``--data-dir`` is ``data_dir``). The key ``settings`` holds the field
values of the settings classes in ``config.py``, by class name.

Order of precedence, last wins: values in the code, the project file, the files named by the
environment variable ``PORTFOLIO_CONFIG`` (separated by the path separator of the system), the
files given with ``--config``, and the options on the command line. A relative path in a file is
relative to the folder of that file. An unknown key stops the run with its name.
"""

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import yaml

FILE_NAME = "config.yaml"
ENV_VAR = "PORTFOLIO_CONFIG"
SETTINGS_KEY = "settings"


def read(path: Path) -> dict[str, Any]:
    """The mapping in one YAML file; an empty file gives an empty mapping."""
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping, got {type(data).__name__}")
    return data


def extra_files(argv: list[str] | None = None) -> list[Path]:
    """Files named by ``PORTFOLIO_CONFIG``, then by each ``--config`` option, in that order."""
    files = [Path(p) for p in os.environ.get(ENV_VAR, "").split(os.pathsep) if p]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", action="append", type=Path, default=[])
    known, _ = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    return files + known.config


def layers(project_dir: Path, argv: list[str] | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """Each config file that applies, with its content, in order of precedence."""
    own = Path(project_dir) / FILE_NAME
    out = [(own, read(own))] if own.is_file() else []
    for path in extra_files(argv):
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        out.append((path, read(path)))
    return out


def _convert(action: argparse.Action, value: Any, base: Path) -> Any:
    """A YAML value as the option would parse it; relative paths resolve from ``base``."""
    if isinstance(value, list):
        return [_convert(action, item, base) for item in value]
    kind = action.type
    if kind is None and isinstance(action.default, Path | list) and action.default:
        sample = action.default[0] if isinstance(action.default, list) else action.default
        kind = Path if isinstance(sample, Path) else None
    if kind is not None and value is not None and not isinstance(value, bool):
        value = kind(value if isinstance(value, str) else str(value))
    if isinstance(value, Path) and not value.is_absolute():
        value = base / value
    if action.choices is not None and value is not None and value not in action.choices:
        raise ValueError(f"{action.dest}: {value!r} is not one of {list(action.choices)}")
    return value


def parse_args(
    parser: argparse.ArgumentParser,
    source: str | Path,
    argv: list[str] | None = None,
    section: str | None = None,
) -> argparse.Namespace:
    """Parse the command line with defaults from the config files.

    ``source`` is the ``__file__`` of the command module, at ``<project>/src/<package>/<name>.py``:
    it gives the project folder and, unless ``section`` is given, the key ``<name>``.
    """
    project_dir = Path(source).resolve().parents[2]
    section = section or Path(source).stem
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=[],
        help=f"YAML file with values for this command (key '{section}'); may be repeated",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved values of this command as YAML and stop",
    )
    actions = {
        a.dest: a
        for a in parser._actions
        if a.option_strings and a.dest not in ("help", "config", "print_config")
    }
    values: dict[str, Any] = {}
    for path, data in layers(project_dir, argv):
        block = data.get(section) or {}
        if not isinstance(block, dict):
            parser.error(f"{path}: '{section}' must be a mapping")
        unknown = sorted(set(block) - set(actions))
        if unknown:
            parser.error(f"{path}: unknown keys under '{section}': {', '.join(unknown)}")
        for key, value in block.items():
            try:
                values[key] = _convert(actions[key], value, path.parent)
            except (TypeError, ValueError) as error:
                parser.error(f"{path}: {section}.{key}: {error}")
    parser.set_defaults(**values)
    args = parser.parse_args(argv)
    if args.print_config:
        resolved = {k: v for k, v in vars(args).items() if k not in ("config", "print_config")}
        yaml.safe_dump({section: plain(resolved)}, sys.stdout, sort_keys=False)
        raise SystemExit(0)
    return args


def plain(value: Any) -> Any:
    """A value with paths as text and tuples as lists, so that YAML can write it."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def settings(project_dir: Path):
    """Class decorator, placed below ``@dataclass``: field defaults from ``settings.<class name>``.

    The decorator runs when ``config.py`` is imported, before any option is parsed, so it reads
    the project file, ``PORTFOLIO_CONFIG`` and the ``--config`` options of the command line.
    """

    def apply(cls):
        if dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__}: place @settings below @dataclass")
        fields = cls.__dict__.get("__annotations__", {})
        for path, data in layers(project_dir):
            block = (data.get(SETTINGS_KEY) or {}).get(cls.__name__) or {}
            unknown = sorted(set(block) - set(fields))
            if unknown:
                raise ValueError(
                    f"{path}: unknown keys under settings.{cls.__name__}: {', '.join(unknown)}"
                )
            for key, value in block.items():
                current = cls.__dict__.get(key)
                if isinstance(current, dataclasses.Field):
                    setattr(cls, key, dataclasses.field(default_factory=_factory(value)))
                elif isinstance(current, tuple) and isinstance(value, list):
                    setattr(cls, key, tuple(tuple(v) if isinstance(v, list) else v for v in value))
                else:
                    setattr(cls, key, value)
        return cls

    return apply


def _factory(value: Any):
    return lambda: yaml.safe_load(yaml.safe_dump(value))


def defaults(cls) -> dict[str, Any]:
    """The field defaults of a settings class, for writing a config file."""
    out = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            out[field.name] = plain(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            out[field.name] = plain(field.default_factory())
    return out
