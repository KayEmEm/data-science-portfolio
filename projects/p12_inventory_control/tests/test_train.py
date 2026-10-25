from inventory_rl.config import FULL, QUICK, SHAPING_PRODUCT, settings
from inventory_rl.train import planned_runs


def test_plan_has_main_and_shaping_runs():
    plan = planned_runs(["fast", "medium", "slow"], FULL)
    main = [r for r in plan if r["shaping"] == "cost" and r["lead"] == "random"]
    assert len(main) == 3 * 2 * len(FULL.seeds)
    fixed = [r for r in plan if r["lead"] == "fixed"]
    assert len(fixed) == len(FULL.shaping_seeds) and {r["product"] for r in fixed} == {"slow"}
    shaping = [r for r in plan if r["shaping"] != "cost"]
    assert len(shaping) == 2 * len(FULL.shaping_seeds)
    assert {r["product"] for r in shaping} == {SHAPING_PRODUCT}
    assert len({r["name"] for r in plan}) == len(plan)


def test_sample_scales_the_run():
    tenth = settings(0.1)
    assert tenth.ppo_steps == FULL.ppo_steps // 10
    assert len(tenth.seeds) == 1 and tenth.test_episodes == FULL.test_episodes // 10
    assert settings(1.0) == FULL
    assert settings(0.5, quick=True) == QUICK


def test_sample_outside_the_unit_interval_is_refused():
    import pytest

    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            FULL.scaled(bad)
