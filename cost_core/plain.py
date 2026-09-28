# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
plain.py - What a result means, in plain words.

Every command prints its tables, and the tables are what an analyst works
from. But the first thing anyone asks of a result is what it says, and a
table of percentiles does not say it. Each function here turns one result
into a few sentences a program manager could read aloud: the headline
number, how confident it is, and the one thing to look at next. They state
only what the numbers show; the method and its caveats stay in the tables,
the assumptions file and the documentation.
"""

from __future__ import annotations

import textwrap
from typing import List

import numpy as np


#: Unit words that need no quoting at any command prompt, for the labels
#: they stand for: ``$`` is special to Unix shells, and single quotes are
#: kept as part of the text by Windows' Command Prompt.
UNIT_WORDS = {"dollars": "$", "thousands": "$K", "millions": "$M", "billions": "$B"}


def units_label(units: str) -> str:
    """``--units thousands`` as the label ``$K``; anything else as given."""
    return UNIT_WORDS.get(str(units).strip().lower(), units)


def _money(v: float, units: str = "") -> str:
    """``$17,016K`` for units "$K", ``$229M`` for "$M", ``17,016 euro`` otherwise."""
    units = (units or "").strip()
    if units in ("", "as entered", "file currency"):
        return f"{v:,.0f}"
    if units.startswith("$"):
        return f"${v:,.0f}{units[1:]}"
    return f"{v:,.0f} {units}"


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 102nd."""
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pct(p: float) -> str:
    return "under 1%" if 0 < p < 0.01 else f"{p:.0%}"


def _chance(p: float) -> str:
    """A probability as a phrase that reads in a sentence."""
    if p <= 0:
        return "almost no chance (none of the simulations)"
    if p < 0.01:
        return "less than a 1% chance"
    if p >= 1:
        return "near certainty (every simulation)"
    pct = f"{p:.0%}"
    article = "an" if pct[0] == "8" or pct.startswith(("11%", "18%")) else "a"
    return f"{article} {pct} chance"


def show(lines: List[str], width: int = 88) -> str:
    """The sentences as a wrapped "What this means" block."""
    body = "\n".join(textwrap.fill(line, width, initial_indent="  - ",
                                   subsequent_indent="    ", break_on_hyphens=False)
                     for line in lines)
    return f"\nWhat this means:\n{body}\n"


def evm(data, fc, units: str = "", where: str = "listed above") -> List[str]:
    """An EVM status and its forecast. ``where`` says where the warning
    signs are: above in the terminal, elsewhere in a workbook or a deck."""
    m = data.metrics().iloc[-1]
    out = [_cpi_sentence(m, units)]
    months = -m.sv_t
    if abs(months) >= 0.05:
        out.append(f"It is {abs(months):.1f} reporting periods "
                   f"{'behind' if months > 0 else 'ahead of'} schedule, and at this pace "
                   f"finishes around period {np.quantile(fc.finish, 0.5):.0f} of a "
                   f"{data.planned_duration}-period plan.")
    p50, p80 = np.quantile(fc.eac, 0.5), np.quantile(fc.eac, 0.8)
    out.append(f"The most likely final cost is about {_money(p50, units)}; plan on "
               f"{_money(p80, units)} to be 80% sure. The budget of "
               f"{_money(data.bac, units)} has {_chance(fc.confidence_of_cost(data.bac))} "
               "of being enough.")
    if fc.contractor_eac is not None and np.isfinite(fc.contractor_eac):
        conf = fc.confidence_of_cost(fc.contractor_eac)
        verdict = ("optimistic: it needs the rest of the program to go better than the "
                   "record so far" if conf < 0.2 else
                   "in line with the program's record so far" if conf < 0.8 else
                   "cautious: the record so far points lower")
        out.append(f"The contractor's estimate of {_money(fc.contractor_eac, units)} has "
                   f"{_chance(conf)} of holding, so it is {verdict}.")
    raised = data.flags()
    raised = raised[raised["raised"]]
    if len(raised) == 1:
        out.append(f"There is 1 warning sign, {where}; start with it.")
    elif len(raised) > 1:
        out.append(f"There are {len(raised)} warning signs, {where}; start with those.")
    out.extend(_evm_monthly(data))
    return out


def _evm_monthly(data) -> List[str]:
    """The data checks and the accounts owing a variance report, in a line each."""
    from cost_core.evm.checks import data_checks, thresholds, variance_breaches

    out = []
    if not hasattr(data, "accounts"):
        return out  # a summary built from something other than EVM data
    checks = data_checks(data)
    if len(checks):
        kinds = checks["check"].value_counts()
        out.append(f"{len(checks)} data check{'s' if len(checks) > 1 else ''} need a question "
                   f"asked before the numbers are trusted, most often "
                   f"'{kinds.index[0].lower()}'.")
    breaches = variance_breaches(data, **thresholds(data))
    if len(breaches):
        names = ", ".join(map(repr, breaches["account"].head(3)))
        more = f" and {len(breaches) - 3} more" if len(breaches) > 3 else ""
        out.append(f"{len(breaches)} account{'s' if len(breaches) > 1 else ''} break"
                   f"{'' if len(breaches) > 1 else 's'} the variance thresholds and owe a "
                   f"variance analysis report: {names}{more}.")
    return out


def _cpi_sentence(m, units: str = "") -> str:
    """What the CPI says, in words; or why there's nothing to say yet."""
    if not (np.isfinite(m.cpi) and m.cpi > 0):
        return (f"No work has been earned yet against {_money(m.acwp, units)} spent, so "
                "there's no cost efficiency to measure." if m.acwp > 0 else
                "Nothing has been earned or spent yet, so there's no cost efficiency to "
                "measure.")
    # Spending 1/CPI dollars for each dollar of planned work: at CPI 0.5 the
    # work costs twice its budget, 100% over, not 50%.
    overrun = 1.0 / m.cpi - 1.0
    cost_word = "over" if overrun > 0 else "under"
    return (f"Each dollar spent so far has bought {m.cpi * 100:.0f} cents' worth of the "
            f"planned work: the work done has cost {abs(overrun):.0%} {cost_word} its "
            "budget.")


def evm_status(data, why: str, units: str = "") -> List[str]:
    """An EVM status without a forecast: early in a program, say."""
    m = data.metrics().iloc[-1]
    out = [_cpi_sentence(m, units)]
    if np.isfinite(m.sv_t) and abs(m.sv_t) >= 0.05:
        out.append(f"It is {abs(m.sv_t):.1f} reporting periods "
                   f"{'behind' if m.sv_t < 0 else 'ahead of'} schedule.")
    out.append(f"There's no forecast of the final cost yet: {why}")
    raised = data.flags()
    n = int(raised["raised"].sum())
    if n:
        out.append(f"There {'is 1 warning sign' if n == 1 else f'are {n} warning signs'}, "
                   "listed above.")
    out.extend(_evm_monthly(data))
    return out


def jcl(result, confidence: float = 0.7, units: str = "") -> List[str]:
    """A joint cost and schedule simulation."""
    pct = f"{confidence:.0%}"
    cost_p = float(np.quantile(result.cost, confidence))
    fin_p = float(np.quantile(result.finish, confidence))
    out = [f"The plan ({result.point_finish:.1f} months, {_money(result.point_cost, units)}) "
           f"has {_chance(result.point_jcl)} of being met on both cost and schedule."]
    out.append(f"To be {pct} sure of cost alone takes {_money(cost_p, units)}, and of "
               f"schedule alone {fin_p:.1f} months, but the two together have only "
               f"{_chance(result.joint(cost_p, fin_p))}: both have to hold at once.")
    budget = result.cost_at(confidence, fin_p + 3)
    if np.isfinite(budget):
        out.append(f"One pair that does reach {pct} jointly: {_money(budget, units)} "
                   f"with {fin_p + 3:.1f} months (the frontier table has the others).")
    # Not the most often critical: an activity everything waits on is critical
    # in every draw whether or not it is uncertain. The one whose own spread
    # moves the finish most is where attention changes the answer.
    crit = result.criticality()
    top = crit.sort_values("rank_corr_finish", ascending=False).iloc[0]
    if top.rank_corr_finish > 0:
        out.append(f"The activity whose uncertainty moves the finish most is "
                   f"{top.activity!r} (on the critical path in {top.criticality:.0%} of "
                   "simulated projects): reducing its risk buys the most schedule.")
    return out


def dcma(result, schedule) -> List[str]:
    """A DCMA 14-point assessment."""
    t = result.table
    failed = t[t["passed"] == False]  # noqa: E712 - None is "could not assess"
    unknown = int(t["passed"].isna().sum())
    out = [f"{result.passed} of the 14 checks passed and {result.failed} failed"
           + (f"; {unknown} could not be assessed from this file." if unknown else ".")]
    first = {1: "tasks missing logic", 5: "hard date constraints", 7: "negative float",
             12: "a broken critical path", 2: "leads"}
    serious = [first[c] for c in (12, 1, 5, 7, 2) if c in set(failed["check"])]
    if serious:
        out.append("Fix these first, since they make the dates the schedule produces "
                   "unreliable: " + ", ".join(serious) + ". dcma_tasks.csv lists every task.")
    elif result.failed == 0:
        out.append("The logic is sound enough to trust the dates it produces, and to run a "
                   "schedule risk analysis (JCL) on.")
    return out


def aoa(result, units: str = "") -> List[str]:
    """An analysis of alternatives."""
    s = result.summary.sort_values("p50")
    best = s.iloc[0]
    out = [f"{best.alternative!r} is the cheapest in {best.p_cheapest:.0%} of simulations, "
           f"with a most likely life-cycle cost of {_money(best.p50, units)} "
           f"({_money(best.p80, units)} to be 80% sure)."]
    if len(s) > 1:
        second = s.iloc[1]
        out.append(f"Next is {second.alternative!r} at {_money(second.p50, units)}, "
                   f"{second.p50 / best.p50 - 1:.0%} more.")
    dom = s[s["dominated_by"].notna()] if "dominated_by" in s else s.iloc[0:0]
    for r in dom.itertuples():
        out.append(f"{r.alternative!r} costs more than {r.dominated_by!r} and does no more, "
                   "so it can be set aside.")
    return out


def cost_risk(result, units: str = "") -> List[str]:
    """A cost risk analysis of an estimate."""
    sim = result.sim
    point = result.point_estimate
    below = sim.point_estimate_percentile / 100.0
    if below < 0.5:
        out = [f"The point estimate of {_money(point, units)} has {_chance(1.0 - below)} of "
               f"being exceeded: only {_pct(below)} of the simulated costs come in at or "
               "under it."]
    elif below < 1.0:
        out = [f"The point estimate of {_money(point, units)} is already {_pct(below)} "
               f"sure: there's {_chance(1.0 - below)} of it being exceeded."]
    else:
        out = [f"The point estimate of {_money(point, units)} is above every simulated "
               "cost: it already carries more than the ranges and risks call for."]
    reserve = sim.p80 - point
    if reserve >= 0:
        out.append(f"To be 50% sure takes {_money(sim.p50, units)}, and 80% sure "
                   f"{_money(sim.p80, units)}: {_money(reserve, units)} "
                   f"({sim.p80 / point - 1:.0%}) over the point estimate. 90% takes "
                   f"{_money(sim.p90, units)}.")
    else:
        out.append(f"To be 50% sure takes {_money(sim.p50, units)}, and 80% sure "
                   f"{_money(sim.p80, units)}, {_money(-reserve, units)} "
                   f"({1 - sim.p80 / point:.0%}) below the point estimate. 90% takes "
                   f"{_money(sim.p90, units)}.")
    top = result.drivers()
    top = top[top["variance_share"] > 0].head(3)
    if len(top):
        first = top.iloc[0]
        kind = "risk" if first.kind == "discrete risk" else "element"
        rest = " and ".join(repr(c) for c in top["component"].iloc[1:])
        out.append(f"The {kind} {first.component!r} drives the most uncertainty "
                   f"({first.variance_share:.0%} of the spread)"
                   + (f", then {rest}." if rest else ".")
                   + " Narrowing those ranges moves the P80 most.")
    alloc = result.allocation(0.8)
    reserve = float(alloc["reserve"].sum())
    if reserve > 0:
        top = alloc.iloc[0]
        own = float(alloc["own_p80"].sum())
        out.append(f"Shared out, the {_money(reserve, units)} of reserve to reach P80 goes "
                   f"mostly to {top.component!r} ({_money(top.reserve, units)}, "
                   f"{top.share_of_reserve:.0%}). Funding every element at its own P80 "
                   f"instead would total {_money(own, units)}, {_money(own - sim.p80, units)} "
                   "more than the P80 of the whole: percentiles don't add.")
    impact = result.impact
    # Only while there is a reserve to lose: with the estimate above the P80
    # the share is undefined.
    if impact.correlated.p80 > impact.independent.p80 and \
            np.isfinite(impact.reserve_understatement):
        out.append(f"Treating the elements as independent would put the P80 at "
                   f"{_money(impact.independent.p80, units)} and lose "
                   f"{impact.reserve_understatement:.0%} of the reserve it needs.")
    if sim.cv < 0.10:
        out.append(f"The spread is narrow (the standard deviation is {sim.cv:.0%} of the "
                   "mean). Ranges that tight usually mean some uncertainty is missing "
                   "rather than that the estimate is safe.")
    out.extend(result.inputs.notes)
    return out


def portfolio(result, n_candidates: int, risk=None, units: str = "") -> List[str]:
    """A portfolio choice."""
    unfunded = [r.candidate for r in result.selected.itertuples() if not r.option
                or (isinstance(r.option, float) and np.isnan(r.option))]
    out = [f"The best use of the budget funds {len(result.funded)} of {n_candidates} "
           f"candidates, for a total value of {result.value:,.1f}."]
    if unfunded:
        out.append("Not funded: " + ", ".join(unfunded) + ".")
    if risk is not None and len(risk):
        col = next((c for c in risk.columns if "prob" in c or "chance" in c or "p_" in c), None)
        if col:
            worst = risk.sort_values(col).iloc[-1]
            year = worst.get("year", "")
            year = int(year) if isinstance(year, float) and year.is_integer() else year
            out.append(f"Once costs grow as history says they do, the riskiest year is {year}, "
                       f"with {_chance(float(worst[col]))} of going over its budget.")
    return out


def menu() -> str:
    """What `ce-core` can do, for someone who has never used it."""
    return textwrap.dedent("""\
        ce-core: cost estimating, EVM and schedule analysis.

        What do you want to do?

          Just have a file? cost-core works out what it is and reads it
              ce-core open my_file.xlsx               (Windows: ce-core open --send-to
                                                       adds it to right-click, Send to)
          Forecast where a program is heading from its EVM data
              ce-core evm --data my_evm.xlsx          (or --ipmdar for an IPMDAR delivery)
          Check a Microsoft Project schedule (DCMA 14-point)
              ce-core schedule-check --mspdi my_schedule.xml
          Joint cost and schedule confidence (JCL)
              ce-core jcl --spec my_jcl.xlsx
          Cost risk on an estimate: S-curve, confidence, drivers
              ce-core cost-risk --data my_estimate.xlsx
          Compare alternatives on life-cycle cost (AoA)
              ce-core aoa --spec my_aoa.xlsx
          Choose which programs to fund within a budget
              ce-core portfolio --spec my_portfolio.xlsx
          Convert between base-year and then-year dollars (your index)
              ce-core inflate --index my_index.csv --data my_phasing.csv --from by2026 --to ty
          Fit a learning curve to production lots
              ce-core fit-lots --csv my_lots.csv --dollar-year 2026
          Unit cost history of real programs from public SARs
              ce-core sar-panel --programs F-35

        New here? See one working first, with example data:
              ce-core demo evm        (or: cost-risk, schedule, jcl, aoa, portfolio)
        Then get a file to fill in with your own numbers:
              ce-core template evm    (or: cost-risk, jcl, aoa, portfolio, lots)

        Set a marking, units or your slide template once:  ce-core settings --write
        Help on any command:  ce-core evm --help
        """)
