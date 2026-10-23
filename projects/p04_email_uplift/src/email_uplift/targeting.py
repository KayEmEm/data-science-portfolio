"""Qini curves and the value of a targeting policy, estimated from randomized data.

Both measures use only the observed outcomes and the known randomization,
so they need no model of the outcome and stay unbiased whatever the scores.
"""

import numpy as np

from email_uplift.config import Economics

N_POINTS = 101  # curve resolution: share targeted from 0 to 1 in steps of 0.01


def qini_curve(score: np.ndarray, y: np.ndarray, treated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve of one arm against control.

    Customers are ranked by ``score`` (highest first). At each share phi of
    the ranking, Q(phi) = Y_t(phi) - Y_c(phi) * N_t(phi) / N_c(phi): the
    incremental outcome among the treated customers in the top phi, with the
    control customers in the same top phi as their counterfactual. The value
    is divided by the number of treated customers, so the curve reads as
    incremental outcome per treated customer.
    """
    score, y, treated = np.asarray(score, float), np.asarray(y, float), np.asarray(treated, bool)
    order = np.argsort(-score, kind="stable")
    y, treated = y[order], treated[order]
    cum_t = np.cumsum(treated)
    cum_c = np.cumsum(~treated)
    sum_t = np.cumsum(y * treated)
    sum_c = np.cumsum(y * ~treated)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cum_c > 0, sum_t - sum_c * cum_t / cum_c, sum_t)
    q = np.concatenate([[0.0], q]) / max(treated.sum(), 1)
    shares = np.linspace(0, 1, N_POINTS)
    index = np.round(shares * len(y)).astype(int)
    return shares, q[index]


def qini_coefficient(score: np.ndarray, y: np.ndarray, treated: np.ndarray) -> float:
    """Area between the Qini curve and the straight line of random targeting.

    Zero means no better than random; the units are those of the outcome per
    treated customer (here dollars).
    """
    shares, q = qini_curve(score, y, treated)
    random_line = shares * q[-1]
    return float(np.trapezoid(q - random_line, shares))


def choose_arms(effects: np.ndarray, economics: Economics) -> np.ndarray:
    """Arm per customer that maximizes expected profit: 0 none, 1 mens, 2 womens.

    An email is sent only when margin times its predicted incremental spend
    exceeds its cost.
    """
    gains = economics.margin * np.asarray(effects, float) - economics.cost_per_email
    gains = np.column_stack([np.zeros(len(gains)), gains])
    return gains.argmax(axis=1)


def rewards(spend: np.ndarray, treatment: np.ndarray, economics: Economics) -> np.ndarray:
    """Profit per customer as observed: margin on spend, less the email cost if one was sent."""
    sent = np.asarray(treatment) != 0
    return economics.margin * np.asarray(spend, float) - economics.cost_per_email * sent


def policy_value(policy: np.ndarray, treatment: np.ndarray, reward: np.ndarray, n_arms: int = 3) -> float:
    """Expected profit per customer if every customer got the arm ``policy`` assigns.

    Horvitz Thompson estimate with the observed arm shares: for each arm,
    the mean reward of the customers who were randomized to it and whom the
    policy would also give it, scaled to the whole arm.
    """
    policy, treatment, reward = np.asarray(policy), np.asarray(treatment), np.asarray(reward, float)
    value = 0.0
    for arm in range(n_arms):
        in_arm = treatment == arm
        if in_arm.any():
            value += float(np.mean(reward[in_arm] * (policy[in_arm] == arm)))
    return value


def bootstrap_policy_values(
    policies: dict[str, np.ndarray],
    treatment: np.ndarray,
    reward: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Policy values over bootstrap samples drawn within each arm (one shared draw per sample).

    Sharing the draws across policies gives paired samples, so differences
    between two policies can be read from the same replicates.
    """
    rng = np.random.default_rng(seed)
    treatment = np.asarray(treatment)
    arm_rows = [np.flatnonzero(treatment == arm) for arm in np.unique(treatment)]
    out = {name: np.empty(n_bootstrap) for name in policies}
    for b in range(n_bootstrap):
        rows = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in arm_rows])
        for name, policy in policies.items():
            out[name][b] = policy_value(policy[rows], treatment[rows], reward[rows])
    return out
