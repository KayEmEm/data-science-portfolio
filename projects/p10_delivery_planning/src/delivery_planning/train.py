"""Run every routing search: the benchmark on all instances and the fleet size study on the week.

Each search is stored as one JSON file under ``artifacts/`` and skipped when the file exists, so an
interrupted run resumes where it stopped (``--force`` runs everything again):

* ``benchmark/<config>/<instance>.json``: each routing configuration on each instance, with the
  instance's own fleet and the distance objective;
* ``fleet/<objective>/cap<k>/<instance>.json``: the instances of the week with at most k vehicles,
  unserved customers allowed at a penalty, for the distance and the cost aware objective.

Searches run in WORKERS processes, one OR-Tools search per CPU core. Guided local search stops at
its time limit, so a repeated run can give slightly different routes.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from portfolio_common.config import parse_args

from delivery_planning.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    FLEET_CAPS,
    FLEET_SECONDS,
    OBJECTIVES,
    QUICK,
    ROUTING_CONFIGS,
    WEEK,
    WORKERS,
)
from delivery_planning.data import instance_names, read_instance


def settings(quick: bool) -> dict:
    if quick:
        configs = {name: dict(ROUTING_CONFIGS[name]) for name in QUICK["configs"]}
        for config in configs.values():
            if config["seconds"]:
                config["seconds"] = QUICK["seconds"]
        return {"configs": configs, "caps": QUICK["fleet_caps"], "fleet_seconds": QUICK["fleet_seconds"]}
    return {"configs": ROUTING_CONFIGS, "caps": FLEET_CAPS, "fleet_seconds": FLEET_SECONDS}


def run_job(raw_dir: str, name: str, kwargs: dict, target: str) -> str:
    # Imported here: OR-Tools bundles its own HiGHS, and its library clashes with highspy in one
    # process. evaluate.py imports this module for ``settings`` and must not load OR-Tools.
    from delivery_planning.routing import solve

    instance = read_instance(Path(raw_dir) / f"{name}.vrp")
    result = solve(instance, **kwargs)
    result["settings"] = kwargs
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(result))
    partial.rename(path)
    return f"{path.parent.name}/{path.stem}: {result['distance']} in {result['seconds']} s"


def jobs(raw_dir: Path, artifacts: Path, quick: bool) -> list[tuple[str, dict, Path]]:
    cfg = settings(quick)
    names = instance_names(raw_dir)
    out = []
    for config_name, config in cfg["configs"].items():
        kwargs = {
            "first": config["first"],
            "metaheuristic": config["metaheuristic"],
            "seconds": config["seconds"],
        }
        for name in names:
            out.append((name, kwargs, artifacts / "benchmark" / config_name / f"{name}.json"))
    main = {"first": "PARALLEL_CHEAPEST_INSERTION", "metaheuristic": "GUIDED_LOCAL_SEARCH"}
    for objective in OBJECTIVES:
        for cap in cfg["caps"]:
            for name in WEEK:
                kwargs = {
                    **main,
                    "seconds": cfg["fleet_seconds"],
                    "vehicles": cap,
                    "objective": objective,
                    "allow_drops": True,
                }
                out.append((name, kwargs, artifacts / "fleet" / objective / f"cap{cap}" / f"{name}.json"))
    return out


def run(data_dir: Path, artifacts: Path, quick: bool, force: bool, workers: int = WORKERS) -> None:
    raw_dir = data_dir / "raw"
    todo = [(n, k, p) for n, k, p in jobs(raw_dir, artifacts, quick) if force or not p.exists()]
    print(f"{len(todo)} searches to run")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, str(raw_dir), n, k, str(p)) for n, k, p in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            message = future.result()
            if done % 20 == 0 or done == len(futures):
                print(
                    f"  {done}/{len(futures)} after {time.perf_counter() - started:.0f} s ({message})",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true", help="two configurations and short limits, for tests")
    parser.add_argument("--force", action="store_true", help="run searches that are stored")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parse_args(parser, __file__)
    run(args.data_dir, args.artifacts, args.quick, args.force, args.workers)


if __name__ == "__main__":
    main()
