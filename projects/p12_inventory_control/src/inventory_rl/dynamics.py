"""Inventory dynamics for a batch of episodes, shared by the Gymnasium environment and the simulator.

One day, in this order:

1. the policy sees the state and places an order ``q``; the order cannot raise the inventory
   position (on hand plus on order) above the capacity;
2. the order gets a lead time ``L`` from 2 to 6 days, drawn at random but confirmed by the
   supplier, so the state shows the day each open order arrives; orders can cross;
3. demand ``D`` arrives; sales are ``min(on hand, D)``, the rest is lost;
4. cost = order cost if q > 0, plus holding cost on the units left on hand, plus the lost margin
   on each lost unit;
5. orders due the next morning arrive.

A scenario fixes the demand and the lead time of every day in advance, so that every policy
can be scored on the same random days (common random numbers).
"""

from dataclasses import dataclass, field

import numpy as np

from inventory_rl import demand as demand_model
from inventory_rl.config import HORIZON, LEAD_PROBS, LEAD_TIMES


@dataclass(frozen=True)
class Problem:
    """One product: its demand model, costs and supply."""

    weekday_mean: tuple[float, ...]
    dispersion: float
    holding: float
    lost_sale: float
    order_cost: float
    capacity: int
    lead_times: tuple[int, ...] = LEAD_TIMES
    lead_probs: tuple[float, ...] = LEAD_PROBS
    horizon: int = HORIZON
    name: str = "product"

    @classmethod
    def from_product(cls, product: dict, **changes) -> "Problem":
        values = {
            "weekday_mean": tuple(product["demand"]["weekday_mean"]),
            "dispersion": product["demand"]["dispersion"],
            "holding": product["holding"],
            "lost_sale": product["lost_sale"],
            "order_cost": product["order_cost"],
            "capacity": int(product["capacity"]),
            "name": product["product"],
        }
        return cls(**(values | changes))

    @property
    def mean_demand(self) -> float:
        return float(np.mean(self.weekday_mean))

    @property
    def max_lead(self) -> int:
        return max(self.lead_times)

    @property
    def mean_lead(self) -> float:
        return float(np.dot(self.lead_times, self.lead_probs))

    @property
    def reward_scale(self) -> float:
        """Cost of one day on which the mean demand is lost; rewards are costs over this scale."""
        return self.lost_sale * max(self.mean_demand, 1e-3)

    @property
    def demand_dict(self) -> dict:
        return {"weekday_mean": list(self.weekday_mean), "dispersion": self.dispersion}

    def initial_stock(self) -> int:
        """Start of every scored episode: cover the mean lead time plus one week."""
        return int(min(self.capacity, round(self.mean_demand * (self.mean_lead + 7))))


DEMO = Problem(
    weekday_mean=(12.0, 11.0, 9.0, 8.0, 8.0, 9.0, 10.0),
    dispersion=8.0,
    holding=0.002,
    lost_sale=1.0,
    order_cost=10.0,
    capacity=420,
    name="demo",
)


@dataclass
class Scenarios:
    demand: np.ndarray  # (episodes x days) units
    lead: np.ndarray  # (episodes x days) lead time of an order placed that day
    first_wday: np.ndarray  # (episodes,) M5 weekday of day 0, 1 Saturday ... 7 Friday
    on_hand: np.ndarray  # (episodes,) units on hand at the start

    def __len__(self) -> int:
        return len(self.first_wday)

    def take(self, index) -> "Scenarios":
        index = np.atleast_1d(index)
        return Scenarios(self.demand[index], self.lead[index], self.first_wday[index], self.on_hand[index])

    def repeat(self, times: int) -> "Scenarios":
        """Each scenario ``times`` times in a row, for scoring a grid of policies at once."""
        return Scenarios(
            np.repeat(self.demand, times, axis=0),
            np.repeat(self.lead, times, axis=0),
            np.repeat(self.first_wday, times),
            np.repeat(self.on_hand, times),
        )


def draw_scenarios(
    problem: Problem,
    rng: np.random.Generator,
    n: int,
    first_wday=None,
    demand: np.ndarray | None = None,
    on_hand=None,
) -> Scenarios:
    """``n`` scenarios; ``demand`` replays one given path (with fresh lead times) instead of drawing."""
    first = rng.integers(1, 8, size=n) if first_wday is None else np.broadcast_to(first_wday, (n,)).copy()
    if demand is None:
        units = demand_model.draw(problem.demand_dict, rng, n, problem.horizon, first)
    else:
        units = np.tile(np.asarray(demand, dtype="int64")[: problem.horizon], (n, 1))
    lead = rng.choice(np.asarray(problem.lead_times), p=np.asarray(problem.lead_probs), size=(n, problem.horizon))
    start = np.full(n, problem.initial_stock()) if on_hand is None else np.broadcast_to(on_hand, (n,)).copy()
    return Scenarios(units, lead.astype("int64"), first.astype("int64"), start.astype("int64"))


def seeded_scenarios(problem: Problem, seed: int, n: int, **kwargs) -> Scenarios:
    return draw_scenarios(problem, np.random.default_rng(seed), n, **kwargs)


COST_PARTS = ("order_cost", "holding_cost", "lost_cost")


@dataclass
class Batch:
    """The state of many episodes, advanced one day at a time."""

    problem: Problem
    scenarios: Scenarios
    on_hand: np.ndarray = field(init=False)
    pipeline: np.ndarray = field(init=False)  # pipeline[:, k] arrives on the morning of day t + 1 + k
    t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.on_hand = self.scenarios.on_hand.astype("int64").copy()
        self.pipeline = np.zeros((len(self.scenarios), self.problem.max_lead), dtype="int64")

    @property
    def size(self) -> int:
        return len(self.on_hand)

    @property
    def position(self) -> np.ndarray:
        return self.on_hand + self.pipeline.sum(axis=1)

    @property
    def wday(self) -> np.ndarray:
        return (self.scenarios.first_wday - 1 + self.t) % 7 + 1

    @property
    def done(self) -> bool:
        return self.t >= self.problem.horizon

    def observation(self) -> np.ndarray:
        """Scaled on hand, the units due on each of the next days, the position, the weekday."""
        cap = float(self.problem.capacity)
        weekday = np.zeros((self.size, 7), dtype="float32")
        weekday[np.arange(self.size), self.wday - 1] = 1.0
        return np.hstack(
            [
                (self.on_hand / cap)[:, None],
                self.pipeline / cap,
                (self.position / cap)[:, None],
                weekday,
            ]
        ).astype("float32")

    def step(self, order: np.ndarray) -> dict[str, np.ndarray]:
        if self.done:
            raise RuntimeError("the episode is over")
        p = self.problem
        rows = np.arange(self.size)
        order = np.clip(np.rint(np.asarray(order, dtype="float64")), 0, None).astype("int64")
        order = np.minimum(order, np.maximum(p.capacity - self.position, 0))
        lead = self.scenarios.lead[:, self.t]
        self.pipeline[rows, lead - 1] += order
        demand = self.scenarios.demand[:, self.t]
        sales = np.minimum(self.on_hand, demand)
        lost = demand - sales
        self.on_hand = self.on_hand - sales
        out = {
            "order": order,
            "demand": demand,
            "sales": sales,
            "lost": lost,
            "on_hand_end": self.on_hand.copy(),
            "order_cost": p.order_cost * (order > 0),
            "holding_cost": p.holding * self.on_hand,
            "lost_cost": p.lost_sale * lost,
        }
        out["cost"] = out["order_cost"] + out["holding_cost"] + out["lost_cost"]
        arrived = self.pipeline[:, 0].copy()
        self.on_hand = self.on_hand + arrived
        self.pipeline[:, :-1] = self.pipeline[:, 1:]
        self.pipeline[:, -1] = 0
        out["arrived"] = arrived
        self.t += 1
        return out


# Policies map a batch to the order of each episode.


def base_stock(level):
    level = np.asarray(level)
    return lambda batch: np.maximum(level - batch.position, 0)


def s_s(reorder, up_to):
    """(s, S): when the position is at or below s, order up to S."""
    reorder, up_to = np.asarray(reorder), np.asarray(up_to)
    return lambda batch: np.where(batch.position <= reorder, up_to - batch.position, 0)


def action_to_order(action: np.ndarray, position: np.ndarray, capacity: int) -> np.ndarray:
    """An action in [-1, 1] is a target position from 0 to the capacity; order the gap, if any."""
    target = (np.clip(np.asarray(action, dtype="float64").reshape(-1), -1, 1) + 1) / 2 * capacity
    return np.maximum(np.rint(target - position), 0).astype("int64")


def order_to_action(order: np.ndarray, position: np.ndarray, capacity: int) -> np.ndarray:
    """The inverse for positive orders; a zero order maps to a target equal to the position."""
    target = np.asarray(position) + np.asarray(order)
    return (2 * target / capacity - 1).astype("float32")


def model_policy(model):
    """A Stable Baselines3 model, acting deterministically on the scaled observation."""

    def act(batch: Batch) -> np.ndarray:
        action, _ = model.predict(batch.observation(), deterministic=True)
        return action_to_order(action, batch.position, batch.problem.capacity)

    return act


def simulate(problem: Problem, scenarios: Scenarios, policy, trace: int = 0) -> dict:
    """Run every scenario to the end; totals per episode, and day by day records of the first ``trace``."""
    batch = Batch(problem, scenarios)
    totals = {
        k: np.zeros(batch.size)
        for k in ("cost", *COST_PARTS, "demand", "sales", "lost", "orders", "units_ordered", "on_hand")
    }
    days = []
    while not batch.done:
        state = {"on_hand_start": batch.on_hand[:trace].copy(), "position": batch.position[:trace].copy()}
        out = batch.step(policy(batch))
        for k in ("cost", *COST_PARTS, "demand", "sales", "lost"):
            totals[k] += out[k]
        totals["orders"] += out["order"] > 0
        totals["units_ordered"] += out["order"]
        totals["on_hand"] += out["on_hand_end"]
        if trace:
            days.append(state | {k: v[:trace].copy() for k, v in out.items()})
    totals["on_hand"] /= problem.horizon
    totals["fill_rate"] = np.divide(
        totals["sales"], totals["demand"], out=np.ones(batch.size), where=totals["demand"] > 0
    )
    result = {"totals": totals}
    if trace:
        result["trace"] = {k: np.stack([d[k] for d in days], axis=1) for k in days[0]}
    return result
