# Getting started

This page is for someone who knows cost estimating, EVM or scheduling and
has never used `cost-core`. No Python is needed: everything here is typed
at a command prompt (Command Prompt or PowerShell on Windows, Terminal on a
Mac or Linux).

## 1. Install it

```bash
pip install "cost-core[plots]"
```

`[plots]` adds the charts. Add `optimize` for portfolio choice and `public`
for reading SAR PDFs: `pip install "cost-core[plots,optimize,public]"`. At a
lab behind a proxy or with no internet, see [using-at-a-lab.md](using-at-a-lab.md).

Check it worked:

```bash
ce-core
```

That prints what it can do. If it says `ce-core` is not recognised, pip's
scripts folder isn't on your PATH (common on managed Windows PCs). Use
`python -m cost_core` instead of `ce-core` everywhere below, and it works
the same.

## 2. The pattern for everything

Each task works the same way, in three steps:

1. **See it work:** `ce-core demo <topic>` runs it on example data, with no
   files needed, and writes the results to `ce-core-demo/<topic>`.
2. **Get a file to fill in:** `ce-core template <topic>` writes one in the
   current folder, holding the example so you can see the layout.
3. **Run it on yours:** the command the template prints.

Every run prints its tables, then **What this means**: the result in a few
plain sentences. Everything is also in one Excel workbook, `report.xlsx`, in
the output folder: a Summary sheet with that reading and the key numbers, a
sheet per table, live Excel charts, and an Assumptions sheet recording every
setting used, for the basis of estimate. It prints one page wide. In the EVM
workbook, CV, SV, CPI, SPI, TCPI and the CPI-based EAC are Excel formulas on
the BCWS, BCWP and ACWP columns, so a reviewer can check them in Excel. The
same tables are written as CSV files too, for other tools.

With the `plots` extra installed, each run also writes `brief.pptx`, a short
widescreen PowerPoint briefing. It opens with the bottom line and the key
numbers, then shows the chart, the table behind it and the assumptions.
It's a starting point for a briefing: copy the slides into your own template.

## 3. The tasks

### Forecast a program's final cost and finish from its EVM data

```bash
ce-core demo evm
ce-core template evm                  # writes my_evm.xlsx
ce-core evm --data my_evm.xlsx --units thousands
```

**What you need:** planned value (BCWS), earned value (BCWP) and actual cost
(ACWP) for each reporting period, per control account if you have them, with
the planned value filled in to the end of the plan. The contractor's EAC is
optional. Columns called PV, EV, AC, "Planned Value", "Month", "WBS" and so
on are recognised, so an export from another tool may read as it is.
**If you have an IPMDAR delivery,** try it directly:
`ce-core evm --ipmdar delivery.zip`. That reader is a preview, not yet tested
on a real delivery: check its reconciliation note against the delivery's own
totals, and fall back to the spreadsheet if anything looks off.

**What you get:** CPI, SPI, SPI(t), TCPI and the independent EACs; warning
signs, such as a contractor EAC that needs a better future than the past;
and the final cost and finish as a range (P50, P80), with the chance that
the budget, and the contractor's EAC, will hold.

### Check a schedule's logic (DCMA 14-point)

```bash
ce-core demo schedule
ce-core schedule-check --mspdi my_schedule.xml
```

**What you need:** the schedule saved from Microsoft Project as XML (File >
Save As > XML Format). Primavera P6 exports the same format.
`ce-core template schedule` repeats these steps.

**What you get:** each of the 14 checks passed or failed against its
threshold, and every failing task by name in `dcma_tasks.csv`.

### The chance of meeting a budget and a date together (JCL)

```bash
ce-core demo jcl
ce-core template jcl                  # writes my_jcl.xlsx
ce-core jcl --spec my_jcl.xlsx
```

**What you need:** the activities, their most likely durations and how
uncertain they are, how they link, their costs, and any discrete risks.
Edit the example in `my_jcl.xlsx`. Or point the spec at a Microsoft
Project schedule instead of listing activities: see "From a Microsoft
Project schedule" in the README.

**What you get:** the joint confidence of the plan, the budget and date
pairs that reach 70% together, and which activity's uncertainty moves the
finish most.

### Compare alternatives on life-cycle cost (AoA)

```bash
ce-core demo aoa
ce-core template aoa                  # writes my_aoa.xlsx
ce-core aoa --spec my_aoa.xlsx
```

**What you get:** each alternative's life-cycle cost (base-year, then-year
and present value), P50 and P80, the chance each is cheapest, and which are
dominated: they cost more and do no more.

### Choose which programs to fund within a budget

```bash
pip install "cost-core[optimize]"
ce-core demo portfolio
ce-core template portfolio            # writes my_portfolio.xlsx
ce-core portfolio --spec my_portfolio.xlsx
```

**What you get:** the best set of programs and funding options within each
year's budget, what an extra dollar in each year would buy, and the chance
each year goes over once costs grow.

### Fit a learning curve to production lots

```bash
ce-core template lots                 # writes my_lots.csv
ce-core fit-lots --csv my_lots.csv --dollar-year 2026 --out results/
```

**What you need:** one row per lot: units, and the lot's recurring cost in
one year's dollars.

## 4. When something goes wrong

The message says what's wrong and what to do next. A missing file names
the folder it looked in and the command that makes one. A missing column
lists the columns it found and the ones it needs.

If a result looks wrong, `assumptions.json` beside it records every setting
and every simplification made on the way in. Please report bugs at
<https://github.com/MichaelFowler1/cost-risk-toolkit/issues>, using
invented numbers, never real program data.

## 5. From Python

Everything above is also a Python library, for notebooks and scripts:

```python
from cost_core.evm import EvmData, forecast
data = EvmData.read("my_evm.xlsx")
data.flags()
forecast(data, seed=1).summary()
```

The README describes each part and the methods behind it.
