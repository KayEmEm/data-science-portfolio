"""The Gymnasium environment: one product, one simulated year, daily orders.

Observation (float32, scaled by the capacity): units on hand, units due on each of the next
``max_lead`` mornings, the inventory position, and the weekday as one hot. Action: one number
in [-1, 1], a target inventory position from 0 to the capacity; the order is the gap between
the target and the position, if positive. Every order size is reachable this way, and a base
stock policy is a constant action.

Reward: minus the day's cost over ``Problem.reward_scale``, with an optional shaping term (see
``config.SHAPING``). ``info`` carries the true cost and its parts, so that every policy can be
scored on the same cost whatever it trained on. An episode ends by truncation after
``horizon`` days; it has no terminal state.

``reset(options={"scenario": s})`` replays one fixed scenario (``dynamics.Scenarios`` with one
row); otherwise the scenario is drawn from the environment's own random generator and, when
``random_start`` is set, the starting stock is drawn uniformly up to half the capacity.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from inventory_rl.config import GAMMA, POTENTIAL_WEIGHT
from inventory_rl.dynamics import DEMO, Batch, Problem, Scenarios, action_to_order, draw_scenarios


class InventoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        problem: Problem = DEMO,
        shaping: str = "cost",
        base_level: float | None = None,
        random_start: bool = True,
        gamma: float = GAMMA,
    ):
        if shaping not in ("cost", "potential", "service"):
            raise ValueError(f"unknown shaping {shaping!r}")
        if shaping == "potential" and base_level is None:
            raise ValueError("potential shaping needs the base stock level")
        self.problem = problem
        self.shaping = shaping
        self.base_level = base_level
        self.random_start = random_start
        self.gamma = gamma
        size = 1 + problem.max_lead + 1 + 7
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.batch: Batch | None = None

    def potential(self) -> float:
        gap = abs(float(self.batch.position[0]) - self.base_level) / self.problem.capacity
        return -POTENTIAL_WEIGHT * gap

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        if scenario is None:
            on_hand = None
            if self.random_start:
                on_hand = int(self.np_random.integers(0, self.problem.capacity // 2 + 1))
            scenario = draw_scenarios(self.problem, self.np_random, 1, on_hand=on_hand)
        elif not isinstance(scenario, Scenarios) or len(scenario) != 1:
            raise ValueError("options['scenario'] must be a Scenarios object with one row")
        self.batch = Batch(self.problem, scenario)
        return self.batch.observation()[0], {}

    def step(self, action):
        if self.batch is None:
            raise RuntimeError("call reset first")
        before = self.potential() if self.shaping == "potential" else 0.0
        order = action_to_order(action, self.batch.position, self.problem.capacity)
        out = self.batch.step(order)
        cost = float(out["cost"][0])
        reward = -cost / self.problem.reward_scale
        if self.shaping == "potential":
            reward += self.gamma * self.potential() - before
        elif self.shaping == "service":
            reward -= self.problem.lost_sale * float(out["lost"][0]) / self.problem.reward_scale
        info = {k: float(v[0]) for k, v in out.items()}
        truncated = self.batch.done
        return self.batch.observation()[0], float(reward), False, truncated, info
