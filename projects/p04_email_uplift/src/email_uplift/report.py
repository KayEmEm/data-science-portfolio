"""Tables, figures and the experiment readout memo, built from the run outputs.

Every number in ``reports/`` comes from the JSON files that ``experiment``,
``train``, ``evaluate`` and ``observational`` write under ``artifacts/``.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.plotting import PALETTE, apply_style

from email_uplift.config import ARTIFACTS_DIR, REPORTS_DIR

MODEL_LABELS = {
    "random": "Random order",
    "past_spend_rule": "Past spend rule",
    "category_rule": "Category rule",
    "t_learner": "T learner",
    "x_learner": "X learner",
    "causal_forest": "Causal forest",
    "dr_learner_causalml": "DR learner (CausalML)",
}
POLICY_LABELS = {
    "no_email": "No email",
    "all_mens": "Mens email to all",
    "all_womens": "Womens email to all",
    "category_rule": "Category rule",
    **{f"model_{k}": f"{v} policy" for k, v in MODEL_LABELS.items()},
}
OUTCOME_UNITS = {"visit": "percentage points", "conversion": "percentage points", "spend": "dollars"}
NEUTRAL = "#8d8d8d"


def _scale(outcome: str) -> float:
    return 100.0 if outcome in ("visit", "conversion") else 1.0


def _fmt_effect(row: dict) -> str:
    k = _scale(row["outcome"])
    unit = " pp" if k == 100 else ""
    prefix = "" if k == 100 else "$"

    def money(v: float) -> str:
        return f"{'-' if v < 0 else ''}{prefix}{abs(v) * k:.2f}{unit}"

    return f"{money(row['effect'])} ({money(row['ci_low'])} to {money(row['ci_high'])})"


def load_outputs(artifacts_dir: Path) -> dict:
    names = ["experiment", "training", "evaluation", "observational", "qini_curves"]
    return {name: json.loads((artifacts_dir / f"{name}.json").read_text()) for name in names}


# Figures


def figure_effects(experiment: dict, path: Path) -> None:
    effects = pd.DataFrame(experiment["effects"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, outcome in zip(axes, ["visit", "conversion", "spend"], strict=True):
        sub = effects[effects["outcome"] == outcome].reset_index(drop=True)
        k = _scale(outcome)
        for i, row in sub.iterrows():
            color = PALETTE[0] if row["arm"] == "mens" else PALETTE[1]
            ax.errorbar(
                row["effect"] * k,
                i,
                xerr=[[(row["effect"] - row["ci_low"]) * k], [(row["ci_high"] - row["effect"]) * k]],
                fmt="o",
                color=color,
                markersize=8,
                capsize=4,
                linewidth=2,
            )
        ax.axvline(0, color=NEUTRAL, linewidth=1)
        ax.set_yticks(range(len(sub)), [f"{a.capitalize()} email" for a in sub["arm"]])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(f"{outcome.capitalize()} ({OUTCOME_UNITS[outcome]})")
        ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    fig.suptitle("Effect of each email against no email, with 95 percent intervals", y=1.02)
    fig.savefig(path)
    plt.close(fig)


def figure_qini(curves: dict, evaluation: dict, path: Path) -> None:
    shares = np.asarray(curves["shares"]) * 100
    models = ["category_rule", *evaluation_models(evaluation)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for r, outcome in enumerate(["visit", "spend"]):
        k = _scale(outcome)
        for c, arm in enumerate(["mens", "womens"]):
            ax = axes[r, c]
            ends = []
            for i, name in enumerate(models):
                q = np.asarray(curves[f"{outcome}|{name}|{arm}"]) * k
                ax.plot(shares, q, color=PALETTE[i], linewidth=2, label=MODEL_LABELS[name])
                ends.append(q[-1])
            ax.plot(
                [0, 100], [0, np.mean(ends)], color=NEUTRAL, linestyle="--", linewidth=1.5, label="Random"
            )
            unit = "pp" if k == 100 else "dollars"
            ax.set_title(f"{arm.capitalize()} email, {outcome}")
            if c == 0:
                ax.set_ylabel(f"Uplift per treated customer ({unit})")
            if r == 1:
                ax.set_xlabel("Customers targeted, ranked by score (percent)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.03)
    )
    fig.suptitle("Qini curves on the test split")
    fig.savefig(path)
    plt.close(fig)


def figure_policies(evaluation: dict, path: Path) -> None:
    rows = pd.DataFrame(evaluation["policies"]).sort_values("value")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(rows))
    best = rows["value"].idxmax()
    colors = [PALETTE[0] if i == best else NEUTRAL for i in rows.index]
    ax.barh(y, rows["value"], color=colors, height=0.6)
    ax.errorbar(
        rows["value"],
        y,
        xerr=[rows["value"] - rows["ci_low"], rows["ci_high"] - rows["value"]],
        fmt="none",
        ecolor="#3d405b",
        capsize=3,
        linewidth=1,
    )
    ax.set_yticks(y, [POLICY_LABELS[p] for p in rows["policy"]])
    ax.axvline(0, color=NEUTRAL, linewidth=1)
    econ = evaluation["economics"]
    ax.set_xlabel(
        f"Expected profit per customer, dollars (margin {econ['margin']:.0%}, "
        f"${econ['cost_per_email']:.2f} per email)"
    )
    ax.set_title("Targeting policies on the test split, with 95 percent bootstrap intervals")
    fig.savefig(path)
    plt.close(fig)


def figure_cost(evaluation: dict, path: Path) -> None:
    rows = pd.DataFrame(evaluation["cost_sensitivity"])
    model = f"model_{evaluation['selected_model']}"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, name in enumerate(["no_email", "all_mens", "category_rule", model]):
        ax.plot(rows["cost_per_email"], rows[name], marker="o", markersize=6, linewidth=2, color=PALETTE[i])
        ax.annotate(
            POLICY_LABELS[name],
            (rows["cost_per_email"].iloc[-1], rows[name].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Cost per email, dollars")
    ax.set_ylabel("Expected profit per customer, dollars")
    ax.set_xlim(right=rows["cost_per_email"].max() * 1.32)
    ax.set_title("Policy value on the test split as the email cost changes")
    fig.savefig(path)
    plt.close(fig)


def figure_observational(observational: dict, path: Path, outcome: str = "visit") -> None:
    rows = pd.DataFrame([r for r in observational["estimates"] if r["outcome"] == outcome]).iloc[::-1]
    k = _scale(outcome)
    truth = rows.loc[rows["estimate"] == "experiment (benchmark)", "effect"].iloc[0] * k
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(rows))
    colors = [
        PALETTE[3] if "full" in e else PALETTE[1] if ("incomplete" in e or "naive" in e) else PALETTE[0]
        for e in rows["estimate"]
    ]
    ax.errorbar(rows["effect"] * k, y, xerr=1.96 * rows["sd"] * k, fmt="none", ecolor=NEUTRAL, capsize=3)
    ax.scatter(rows["effect"] * k, y, color=colors, s=60, zorder=3)
    ax.axvline(truth, color=PALETTE[0], linestyle="--", linewidth=1.5)
    ax.set_yticks(y, rows["estimate"].str.replace("_", " ").str.capitalize())
    ax.set_xlabel(f"Estimated effect of the mens email on visits ({OUTCOME_UNITS[outcome]})")
    ax.set_title("Observational estimates against the experimental benchmark (dashed)")
    fig.savefig(path)
    plt.close(fig)


# Tables


def evaluation_models(evaluation: dict) -> list[str]:
    return [
        m for m in MODEL_LABELS if m in {r["model"] for r in evaluation["qini"]} and m not in BASELINE_SET
    ]


BASELINE_SET = {"random", "past_spend_rule", "category_rule"}


def ab_tables(experiment: dict) -> str:
    effects = pd.DataFrame(experiment["effects"])
    effects["effect (95% CI)"] = [_fmt_effect(r) for r in experiment["effects"]]
    effects["relative lift"] = effects["relative_lift"].map(lambda v: f"{v:.1%}")
    effects["p (Holm)"] = effects["p_holm"].map(lambda v: f"{v:.1e}")
    k = effects["outcome"].map(_scale)
    effects["control mean"] = (effects["control_mean"] * k).round(3)
    effects["arm mean"] = (effects["treated_mean"] * k).round(3)
    t_effects = effects[
        ["arm", "outcome", "control mean", "arm mean", "effect (95% CI)", "relative lift", "p (Holm)"]
    ]

    power = pd.DataFrame(experiment["power"])
    kp = power["outcome"].map(_scale)
    power["minimum detectable effect"] = (power["mde"] * kp).round(3)
    power["observed effect"] = (power["observed_effect"] * kp).round(3)
    power = power.rename(
        columns={"n_per_arm": "customers in arm", "n_needed_for_observed": "n per arm needed"}
    )
    t_power = power[
        [
            "arm",
            "outcome",
            "customers in arm",
            "minimum detectable effect",
            "observed effect",
            "n per arm needed",
        ]
    ]

    vr = pd.DataFrame(experiment["variance_reduction"])
    t_vr = pd.DataFrame(
        {
            "arm": vr["arm"],
            "outcome": vr["outcome"],
            "corr. with past spend": vr["corr_with_pre_period"].round(3),
            "SE raw": vr["se_raw"].map(lambda v: f"{v:.4g}"),
            "SE CUPED": vr["se_cuped"].map(lambda v: f"{v:.4g}"),
            "SE regression": vr["se_regression"].map(lambda v: f"{v:.4g}"),
            "variance cut CUPED": vr["variance_reduction_cuped"].map(lambda v: f"{v:.2%}"),
            "variance cut regression": vr["variance_reduction_regression"].map(lambda v: f"{v:.2%}"),
        }
    )
    balance = pd.DataFrame(experiment["balance"]).round(4)
    sr = experiment["sample_ratio"]
    return "\n".join(
        [
            "# A/B test readout tables",
            "",
            f"Data: `{experiment['data']}`, {experiment['n_customers']:,} customers.",
            "",
            "## Randomization checks",
            "",
            f"Sample ratio: arm counts {sr['counts']}; chi square {sr['chi2']:.3f}, p = {sr['p_value']:.3f} "
            "against the planned equal split.",
            "",
            f"Largest absolute standardized mean difference of a pre treatment feature: "
            f"{experiment['max_abs_smd']:.4f} (adequate balance is below 0.1).",
            "",
            balance.to_markdown(index=False),
            "",
            "## Effects against control",
            "",
            "Visit and conversion in percentage points (pp), spend in dollars per customer, over the two "
            "weeks after the send. Intervals use Welch standard errors; p values are Holm adjusted over "
            "the six tests.",
            "",
            t_effects.to_markdown(index=False),
            "",
            "## Power",
            "",
            f"Minimum detectable effect at {experiment['power_target']:.0%} power and a two sided level of "
            f"{experiment['alpha']}, with the arm sizes of this test. The last column is the number of "
            "customers per arm a new test would need to detect the observed effect.",
            "",
            t_power.to_markdown(index=False),
            "",
            "## Variance reduction",
            "",
            "CUPED uses past year spend (`history`) as the pre period covariate. Regression adjustment "
            "uses all pre treatment features, interacted with the arm (Lin, 2013), with HC2 standard "
            "errors.",
            "",
            t_vr.to_markdown(index=False),
            "",
        ]
    )


def uplift_table(evaluation: dict, training: dict) -> str:
    qini = pd.DataFrame(evaluation["qini"])
    out = []
    for outcome in ("visit", "spend"):
        k = _scale(outcome)
        sub = qini[qini["outcome"] == outcome]
        wide = []
        for model in MODEL_LABELS:
            rows = sub[sub["model"] == model]
            if rows.empty:
                continue
            entry = {"model": MODEL_LABELS[model]}
            for _, r in rows.iterrows():
                entry[f"{r['arm']} email"] = (
                    f"{r['qini'] * k:.3f} ({r['ci_low'] * k:.3f} to {r['ci_high'] * k:.3f})"
                )
            wide.append(entry)
        unit = "percentage points" if k == 100 else "dollars"
        out += [
            f"## Qini coefficient, {outcome} ({unit} per treated customer)",
            "",
            pd.DataFrame(wide).to_markdown(index=False),
            "",
        ]
    val = training["validation"]["spend"]
    sel = pd.DataFrame(
        [
            {
                "model": MODEL_LABELS[m],
                "validation policy value (dollars per customer)": round(r["policy_value"], 4),
            }
            for m, r in val.items()
            if "policy_value" in r
        ]
    )
    return "\n".join(
        [
            "# Uplift model comparison",
            "",
            f"Test split: {evaluation['n_test']:,} customers, each arm compared with control. Intervals are "
            "95 percent bootstrap intervals over test customers; for the random order they are the spread of "
            "the coefficient over random rankings, which is the noise floor a model must clear.",
            "",
            *out,
            "## Model selection on the validation split",
            "",
            sel.to_markdown(index=False),
            "",
            f"Selected model: {MODEL_LABELS[training['selected_model']]}. "
            f"Selected policy: {POLICY_LABELS[training['selected_policy']]}.",
            "",
        ]
    )


def policy_table(evaluation: dict) -> str:
    ref = evaluation["policy_reference"]
    rows = pd.DataFrame(evaluation["policies"])
    table = pd.DataFrame(
        {
            "policy": rows["policy"].map(POLICY_LABELS),
            "profit per customer (95% CI)": [
                f"{r.value:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f})" for r in rows.itertuples()
            ],
            f"difference to {POLICY_LABELS[ref]} (95% CI)": [
                f"{getattr(r, f'diff_vs_{ref}'):.3f} ({r.diff_ci_low:.3f} to {r.diff_ci_high:.3f})"
                for r in rows.itertuples()
            ],
            "share mens": rows["share_mens"].map(lambda v: f"{v:.0%}"),
            "share womens": rows["share_womens"].map(lambda v: f"{v:.0%}"),
            "share none": rows["share_none"].map(lambda v: f"{v:.0%}"),
        }
    )
    cost = pd.DataFrame(evaluation["cost_sensitivity"]).round(4)
    names = {"cost_per_email": "cost per email", "model_share_sent": "share mailed by the model policy"}
    cost = cost.rename(columns={c: names.get(c, POLICY_LABELS.get(c, c)) for c in cost.columns})
    econ = evaluation["economics"]
    return "\n".join(
        [
            "# Targeting policies",
            "",
            f"Profit per customer = {econ['margin']:.0%} margin on two week spend, less "
            f"${econ['cost_per_email']:.2f} per email sent. Both are assumptions. Values are estimated on "
            "the test split from the randomized arms (Horvitz Thompson), with paired bootstrap intervals.",
            "",
            table.to_markdown(index=False),
            "",
            "## Sensitivity to the email cost",
            "",
            cost.to_markdown(index=False),
            "",
        ]
    )


def observational_table(observational: dict) -> str:
    rows = pd.DataFrame(observational["estimates"])
    out = []
    for outcome in ("visit", "spend"):
        k = _scale(outcome)
        sub = rows[rows["outcome"] == outcome]
        out += [
            f"## {outcome.capitalize()} ({OUTCOME_UNITS[outcome]})",
            "",
            pd.DataFrame(
                {
                    "estimate": sub["estimate"].str.replace("_", " "),
                    "effect": (sub["effect"] * k).round(3),
                    "spread (sd)": (sub["sd"] * k).round(3),
                    "bias against experiment": (sub["bias"] * k).round(3),
                }
            ).to_markdown(index=False),
            "",
        ]
    refs = pd.DataFrame(observational["refutations_visit"]).round(4)
    low, high = observational["selection_probability_range"]
    return "\n".join(
        [
            "# Observational analysis with DoWhy",
            "",
            f"Each of {observational['n_replicates']} draws keeps about "
            f"{observational['n_sample_mean']:,.0f} customers of the mens email and control arms, as if a "
            "marketer had mailed recent, high value web customers more often (selection probability "
            f"{low:.2f} to {high:.2f}; "
            f"{observational['share_mailed']:.0%} of kept customers mailed). Effects are means over draws; "
            "the spread is the standard deviation over draws (for the experiment, its standard error).",
            "",
            f"Full adjustment set: {', '.join(observational['confounders'])}. "
            f"Incomplete set: {', '.join(observational['incomplete_confounders'])} (past spend left out).",
            "",
            *out,
            "## Refutation tests (visit, linear regression, first draw)",
            "",
            refs.to_markdown(index=False),
            "",
        ]
    )


# Memo


def _effect(experiment: dict, arm: str, outcome: str) -> dict:
    return next(r for r in experiment["effects"] if r["arm"] == arm and r["outcome"] == outcome)


def _qini(evaluation: dict, outcome: str, model: str, arm: str) -> dict:
    return next(
        r for r in evaluation["qini"] if r["outcome"] == outcome and r["model"] == model and r["arm"] == arm
    )


def targeting_findings(evaluation: dict) -> dict:
    """Facts the memo states about targeting, derived rather than written by hand."""
    models = evaluation_models(evaluation)
    spend_winners = [
        m for m in models for a in ("mens", "womens") if _qini(evaluation, "spend", m, a)["ci_low"] > 0
    ]
    cost = pd.DataFrame(evaluation["cost_sensitivity"])
    model_col = f"model_{evaluation['selected_model']}"
    uniform_best = cost[["no_email", "all_mens", "category_rule"]].max(axis=1)
    zero_cost = cost[cost["cost_per_email"] == 0].iloc[0]
    return {
        "spend_winners": sorted(set(spend_winners)),
        "targeting_wins_at": cost.loc[cost[model_col] > uniform_best, "cost_per_email"].tolist(),
        # all_mens beats no_email while the email costs less than margin times its spend effect
        "break_even_cost": float(zero_cost["all_mens"] - zero_cost["no_email"]),
        "max_cost": float(cost["cost_per_email"].max()),
    }


def memo(outputs: dict) -> str:
    exp, ev, obs = outputs["experiment"], outputs["evaluation"], outputs["observational"]
    tr = outputs["training"]
    m_visit, m_spend = _effect(exp, "mens", "visit"), _effect(exp, "mens", "spend")
    w_visit, w_spend = _effect(exp, "womens", "visit"), _effect(exp, "womens", "spend")
    m_conv, w_conv = _effect(exp, "mens", "conversion"), _effect(exp, "womens", "conversion")
    all_significant = all(r["p_holm"] < exp["alpha"] for r in exp["effects"])
    policies = {p["policy"]: p for p in ev["policies"]}
    best = max(policies.values(), key=lambda p: p["value"])
    chosen = policies[tr["selected_policy"]]  # chosen on the validation split
    none, selected = policies["no_email"], policies[f"model_{tr['selected_model']}"]
    gain = chosen["value"] - none["value"]
    spend_raised = m_spend["ci_low"] > 0 and w_spend["ci_low"] > 0
    facts = targeting_findings(ev)
    econ = ev["economics"]
    forest_w = _qini(ev, "visit", "causal_forest", "womens")
    rule_w = _qini(ev, "visit", "category_rule", "womens")
    rand_w = _qini(ev, "visit", "random", "womens")
    mde_spend = next(r for r in exp["power"] if r["arm"] == "mens" and r["outcome"] == "spend")["mde"]
    vr = pd.DataFrame(exp["variance_reduction"]).set_index(["arm", "outcome"])
    ov = pd.DataFrame(obs["estimates"]).query("outcome == 'visit'").set_index("estimate")["effect"] * 100
    refs = pd.DataFrame(obs["refutations_visit"])
    incomplete_pass = bool((refs.query("adjustment == 'incomplete'")["p_value"] > 0.05).all())
    counts = ", ".join(f"{k} {v:,}" for k, v in exp["sample_ratio"]["counts"].items())

    if facts["spend_winners"]:
        names = ", ".join(MODEL_LABELS[m] for m in facts["spend_winners"])
        spend_text = f"Only {names} ranks customers by incremental spend better than chance."
    else:
        spend_text = (
            "No uplift model ranks customers by incremental spend better than chance: every spend Qini "
            "interval on the test split includes zero."
        )
    if facts["targeting_wins_at"]:
        costs = ", ".join(f"${c:.2f}" for c in facts["targeting_wins_at"])
        cost_text = f"The model policy beats every simple policy only at an email cost of {costs}."
    else:
        cost_text = (
            "At every email cost from $0.00 to "
            f"${facts['max_cost']:.2f}, a simple policy earns at least as much as the model policy."
        )
    recommendation = {
        "all_mens": "send the mens email to every customer",
        "all_womens": "send the womens email to every customer",
        "no_email": "send no email",
    }.get(chosen["policy"], f"use the {POLICY_LABELS[chosen['policy']]}")
    chosen_note = (
        "it is also the best policy on the test split"
        if chosen is best
        else (f"on the test split {POLICY_LABELS[best['policy']]} earns more (${best['value']:.3f})")
    )
    heading3 = "pays" if chosen["policy"].startswith("model_") else "does not pay yet"
    rule_matches = forest_w["ci_low"] <= rule_w["qini"] <= forest_w["ci_high"]
    visit_detected = forest_w["ci_low"] > rand_w["ci_high"]
    if visit_detected:
        heading4 = "Response differs by customer in visits" + (
            "." if facts["spend_winners"] else ", not measurably in spend."
        )
    else:
        heading4 = "No model finds customers whose visits respond more than others."
    band = f"random rankings fall between {rand_w['ci_low'] * 100:.2f} and {rand_w['ci_high'] * 100:.2f}"
    if visit_detected:
        visit_text = (
            "For the womens email, ranking by the causal forest finds the customers whose visits rise most "
            f"(Qini {forest_w['qini'] * 100:.2f} points; {band})."
        )
        if rule_matches:
            visit_text += (
                f" Ranking simply by past womens purchases does as well ({rule_w['qini'] * 100:.2f})."
            )
    else:
        visit_text = (
            f"For the womens email, the causal forest reaches a Qini of {forest_w['qini'] * 100:.2f} points, "
            f"inside the band where {band}."
        )
    lines = [
        "# Experiment readout: email test of March 2008",
        "",
        "| | |",
        "| --- | --- |",
        "| **To** | Marketing leadership |",
        "| **From** | Analytics |",
        "| **Data** | Hillstrom MineThatData email test: 64,000 customers, two weeks of outcomes |",
        "| **Question** | Did the campaign raise spending, and who should get the next email? |",
        "",
        "## Summary",
        "",
        f"1. **{'Both emails raised spending.' if spend_raised else 'The spend effect is not clear.'}** "
        f"Over two weeks the mens email added {_fmt_effect(m_spend)} "
        f"per customer against no email (+{m_spend['relative_lift']:.0%}), the womens email "
        f"{_fmt_effect(w_spend)} (+{w_spend['relative_lift']:.0%}). Visits rose by "
        f"{m_visit['effect'] * 100:.1f} and {w_visit['effect'] * 100:.1f} percentage points, conversions "
        f"by {m_conv['effect'] * 100:.2f} and {w_conv['effect'] * 100:.2f} points."
        + (" All six effects remain significant after Holm correction." if all_significant else ""),
        f"2. **Recommendation: {recommendation} in this population.** Under the stated economics "
        f"({econ['margin']:.0%} margin, ${econ['cost_per_email']:.2f} per email) this earns "
        f"${chosen['value']:.3f} per customer on the test split against ${none['value']:.3f} with no "
        f"email: ${gain:.3f} more per customer, about ${gain * 100_000:,.0f} per 100,000 customers. "
        f"The policy was chosen on the validation split; {chosen_note}.",
        f"3. **Model based targeting {heading3}.** {spend_text} The policy of the model chosen on "
        f"the validation split ({MODEL_LABELS[tr['selected_model']]}) earns ${selected['value']:.3f} per "
        f"customer on the test split. {cost_text}",
        f"4. **{heading4}** {visit_text}",
        "",
        "## Was the test sound?",
        "",
        f"* Arm sizes ({counts}) match the planned equal split (p = {exp['sample_ratio']['p_value']:.2f}).",
        "* Every pre treatment feature is balanced (largest standardized difference "
        f"{exp['max_abs_smd']:.3f}; threshold 0.1).",
        f"* Power: the test could detect a spend effect of ${mde_spend:.2f} per customer with "
        f"{exp['power_target']:.0%} power; "
        + (
            "both observed effects are larger."
            if min(m_spend["effect"], w_spend["effect"]) > mde_spend
            else "an observed effect is smaller than that."
        ),
        "* Variance reduction: past year spend barely predicts two week spend (correlation "
        f"{vr.loc[('mens', 'spend'), 'corr_with_pre_period']:.3f}), so CUPED cuts the spend variance by "
        f"{vr.loc[('mens', 'spend'), 'variance_reduction_cuped']:.2%}. Regression on all pre treatment "
        "features cuts the visit variance by "
        f"{vr.loc[('mens', 'visit'), 'variance_reduction_regression']:.1%}. Neither changes a conclusion.",
        "",
        "## What if there had been no experiment?",
        "",
        "A simulated targeted send (recent, high value web customers mailed more often) shows what an "
        "observational study would have reported for the mens email. The naive comparison gives "
        f"{ov['naive difference']:.1f} points of extra visits against the true "
        f"{ov['experiment (benchmark)']:.1f}. Adjusting for every confounder except past spend still gives "
        f"{ov['linear_regression, incomplete adjustment']:.1f}. Only the full set recovers the truth "
        f"({ov['propensity_weighting, full adjustment']:.1f} with propensity weighting, "
        f"{ov['linear_regression, full adjustment']:.1f} with regression)."
        + (
            " DoWhy's refutation tests pass for the incomplete set as well, so they cannot reveal a "
            "missing confounder."
            if incomplete_pass
            else ""
        ),
        "",
        "## Next steps",
        "",
        "* Mail the full file, and keep a random holdout without email to keep measuring the effect.",
        "* Before investing in targeting models, run a larger test or track a longer outcome window, so "
        "that differences in spend response become measurable.",
        "* Replace the assumed margin and email cost with finance figures. At a "
        f"{econ['margin']:.0%} margin, mailing everyone beats mailing no one while an email costs less "
        f"than ${facts['break_even_cost']:.2f}.",
        "",
        "Details: [A/B tables](ab_results.md), [uplift models](uplift_comparison.md), "
        "[policies](policy.md), [observational analysis](observational.md).",
        "",
    ]
    return "\n".join(lines)


def key_numbers(outputs: dict) -> dict:
    exp, ev, tr = outputs["experiment"], outputs["evaluation"], outputs["training"]
    policies = {p["policy"]: p for p in ev["policies"]}
    return {
        "data": exp["data"],
        "n_customers": exp["n_customers"],
        "sample_ratio_p": exp["sample_ratio"]["p_value"],
        "effects": {f"{r['arm']}_{r['outcome']}": r for r in exp["effects"]},
        "selected_model": tr["selected_model"],
        "selected_policy": tr["selected_policy"],
        "test_policy_values": {k: v["value"] for k, v in policies.items()},
        "economics": ev["economics"],
    }


def run(artifacts_dir: Path, reports_dir: Path) -> None:
    outputs = load_outputs(artifacts_dir)
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    apply_style()
    figure_effects(outputs["experiment"], figures / "effects.png")
    figure_qini(outputs["qini_curves"], outputs["evaluation"], figures / "qini_curves.png")
    figure_policies(outputs["evaluation"], figures / "policy_values.png")
    figure_cost(outputs["evaluation"], figures / "cost_sensitivity.png")
    figure_observational(outputs["observational"], figures / "observational.png")
    (reports_dir / "ab_results.md").write_text(ab_tables(outputs["experiment"]))
    (reports_dir / "uplift_comparison.md").write_text(
        uplift_table(outputs["evaluation"], outputs["training"])
    )
    (reports_dir / "policy.md").write_text(policy_table(outputs["evaluation"]))
    (reports_dir / "observational.md").write_text(observational_table(outputs["observational"]))
    (reports_dir / "experiment_readout.md").write_text(memo(outputs))
    pd.DataFrame(outputs["evaluation"]["qini"]).to_csv(reports_dir / "uplift_comparison.csv", index=False)
    pd.DataFrame(outputs["evaluation"]["policies"]).to_csv(reports_dir / "policy_values.csv", index=False)
    pd.DataFrame(outputs["experiment"]["effects"]).to_csv(reports_dir / "ab_effects.csv", index=False)
    (reports_dir / "results.json").write_text(json.dumps(key_numbers(outputs), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run(args.artifacts, args.reports)
    print(f"reports written to {args.reports}")


if __name__ == "__main__":
    main()
