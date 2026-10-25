import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import inventory_rl
from inventory_rl.dynamics import order_to_action, s_s, seeded_scenarios, simulate
from inventory_rl.env import InventoryEnv


def test_registered_environment_passes_the_checkers():
    env = gym.make(inventory_rl.ENV_ID)
    check_env(env.unwrapped)
    sb3_check_env(env.unwrapped)


def test_episode_is_one_year_and_truncated(problem):
    env = InventoryEnv(problem)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        steps += 1
        assert not terminated
        if truncated:
            break
    assert steps == problem.horizon
    assert {"cost", "lost", "order"} <= set(info)


def test_environment_and_batch_simulator_agree(problem):
    """The environment, driven by (s, S) through its action, gives the simulator's cost."""
    scenarios = seeded_scenarios(problem, 11, 3)
    policy = s_s(30, 120)
    expected = simulate(problem, scenarios, policy)["totals"]["cost"]
    env = InventoryEnv(problem, random_start=False)
    for i in range(len(scenarios)):
        env.reset(options={"scenario": scenarios.take(i)})
        total, rewards = 0.0, 0.0
        done = False
        while not done:
            batch = env.batch
            action = order_to_action(policy(batch), batch.position, problem.capacity)
            _, reward, _, done, info = env.step(action)
            total += info["cost"]
            rewards += reward
        assert total == pytest.approx(expected[i])
        assert rewards == pytest.approx(-total / problem.reward_scale)


def run_rewards(env, scenario, action):
    env.reset(options={"scenario": scenario})
    rewards, costs, lost = [], [], []
    done = False
    while not done:
        _, reward, _, done, info = env.step(np.array([action], dtype=np.float32))
        rewards.append(reward)
        costs.append(info["cost"])
        lost.append(info["lost"])
    return np.array(rewards), np.array(costs), np.array(lost)


def test_potential_shaping_telescopes(problem):
    scenario = seeded_scenarios(problem, 5, 1)
    plain = InventoryEnv(problem, random_start=False)
    shaped = InventoryEnv(problem, shaping="potential", base_level=60, random_start=False, gamma=1.0)
    r_plain, _, _ = run_rewards(plain, scenario, 0.1)
    shaped.reset(options={"scenario": scenario})
    start = shaped.potential()
    r_shaped, _, _ = run_rewards(shaped, scenario, 0.1)
    end = shaped.potential()
    # With gamma 1 the added terms sum to the last potential minus the first.
    assert (r_shaped - r_plain).sum() == pytest.approx(end - start)


def test_service_shaping_adds_the_lost_margin(problem):
    scenario = seeded_scenarios(problem, 5, 1, on_hand=0)
    r_plain, costs, lost = run_rewards(InventoryEnv(problem, random_start=False), scenario, -1.0)
    r_service, costs_2, _ = run_rewards(
        InventoryEnv(problem, shaping="service", random_start=False), scenario, -1.0
    )
    np.testing.assert_allclose(costs, costs_2)
    np.testing.assert_allclose(r_plain - r_service, problem.lost_sale * lost / problem.reward_scale)
    assert lost.sum() > 0


def test_random_start_stays_within_half_the_capacity(problem):
    env = InventoryEnv(problem)
    for seed in range(20):
        env.reset(seed=seed)
        assert 0 <= env.batch.on_hand[0] <= problem.capacity // 2


def test_bad_arguments_are_refused(problem):
    with pytest.raises(ValueError):
        InventoryEnv(problem, shaping="bonus")
    with pytest.raises(ValueError):
        InventoryEnv(problem, shaping="potential")
    with pytest.raises(ValueError):
        InventoryEnv(problem).reset(options={"scenario": seeded_scenarios(problem, 1, 2)})
