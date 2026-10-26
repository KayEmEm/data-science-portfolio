import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from portfolio_common import config

REPO = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/default/data"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mode", choices=["a", "b"], default="a")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--out", default=Path("/default/out"))  # a path option without a type
    return parser


def _source(project_dir: Path) -> str:
    return str(project_dir / "src" / "pkg" / "train.py")


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_parse_without_file_keeps_code_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    args = config.parse_args(_parser(), _source(tmp_path), [])
    assert args.n == 10 and args.data_dir == Path("/default/data") and not args.quick


def test_parse_layers_in_order_and_command_line_wins(tmp_path, monkeypatch):
    _write(
        tmp_path / "config.yaml", {"train": {"n": 20, "rate": 0.5, "data_dir": "data", "out": "o"}}
    )
    user = _write(tmp_path / "user.yaml", {"train": {"n": 30, "quick": True, "sizes": [4, 5]}})
    env = _write(tmp_path / "env.yaml", {"train": {"n": 25, "mode": "b"}})
    monkeypatch.setenv(config.ENV_VAR, str(env))
    args = config.parse_args(_parser(), _source(tmp_path), ["--config", str(user), "--rate", "0.7"])
    assert args.n == 30  # --config file after the environment file
    assert args.mode == "b" and args.quick and args.sizes == [4, 5]
    assert args.rate == 0.7  # the command line wins
    assert args.data_dir == tmp_path / "data"  # relative to the folder of the file
    assert args.out == tmp_path / "o"  # a path, as the code default is one


def test_parse_other_sections_are_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    _write(tmp_path / "config.yaml", {"evaluate": {"other": 1}, "train": {"n": 3}})
    assert config.parse_args(_parser(), _source(tmp_path), []).n == 3


@pytest.mark.parametrize("block", [{"nn": 3}, {"mode": "c"}])
def test_parse_stops_on_unknown_key_or_bad_value(tmp_path, monkeypatch, block):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    _write(tmp_path / "config.yaml", {"train": block})
    with pytest.raises(SystemExit):
        config.parse_args(_parser(), _source(tmp_path), [])


def test_parse_missing_config_file_stops(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError):
        config.parse_args(_parser(), _source(tmp_path), ["--config", str(tmp_path / "none.yaml")])


def test_print_config_writes_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    _write(tmp_path / "config.yaml", {"train": {"n": 7}})
    with pytest.raises(SystemExit) as stop:
        config.parse_args(_parser(), _source(tmp_path), ["--print-config"])
    assert stop.value.code == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["train"]["n"] == 7 and printed["train"]["data_dir"] == "/default/data"


def test_settings_decorator_reads_field_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.setattr("sys.argv", ["prog"])
    _write(tmp_path / "config.yaml", {"settings": {"Run": {"steps": 5, "seeds": [1, 2]}}})

    @dataclass(frozen=True)
    @config.settings(tmp_path)
    class Run:
        steps: int = 100
        seeds: tuple[int, ...] = (42,)
        rate: float = 0.1

    run = Run()
    assert run.steps == 5 and run.seeds == (1, 2) and run.rate == 0.1
    assert Run(steps=9).steps == 9
    assert config.defaults(Run) == {"steps": 5, "seeds": [1, 2], "rate": 0.1}


def test_settings_decorator_stops_on_unknown_field(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.setattr("sys.argv", ["prog"])
    _write(tmp_path / "config.yaml", {"settings": {"Run": {"stepz": 5}}})
    with pytest.raises(ValueError, match="stepz"):

        @dataclass
        @config.settings(tmp_path)
        class Run:
            steps: int = 100


@pytest.mark.parametrize(
    "path", sorted(REPO.glob("projects/*/config.yaml")), ids=lambda p: p.parent.name
)
def test_project_settings_blocks_name_settings_classes(path):
    # The decorator reads only blocks named after a class it decorates; others are never read.
    blocks = set(config.read(path).get(config.SETTINGS_KEY) or {})
    source = "".join(f.read_text(encoding="utf-8") for f in path.parent.glob("src/*/config.py"))
    classes = set(re.findall(r"@\w*settings\([^)]*\)\s*\nclass (\w+)", source))
    assert blocks <= classes, f"blocks without a settings class: {sorted(blocks - classes)}"
