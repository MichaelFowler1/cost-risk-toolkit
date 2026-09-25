# Roadmap: proposals, not commitments

Ideas for making `cost-core` the main programming package for cost estimators
and EVM and scheduling specialists, written 2026-09-25. Nothing here is
decided. Each item says why it matters and what makes it hard, so it can be
taken, changed or dropped on its merits.

Status markers: **[started]** work exists on a branch; **[idea]** nothing yet.

## 1. Earned value management (the largest gap)

The package had nothing for EVM specialists.

- **[done, unreleased] EVM metrics and forecasting** (`cost_core.evm`): CV, SV, CPI,
  SPI, TCPI, the standard independent EACs, and earned schedule (ES, SPI(t),
  IEAC(t)), since SPI drifts back to 1.0 near the end of a late program.
- **[done, unreleased] Probabilistic EAC and completion date.** Calibration
  measured on synthetic programs: P80 holds 79 to 82% with independent
  periods, 72 to 77% with strongly persistent ones; the persistent case is
  the thing to improve (a short record cannot pin down persistence). Resample the
  program's own periodic efficiencies to give an EAC distribution (P50, P80)
  that narrows as the program matures, plus a joint cost and date confidence
  in the same form as the JCL. Flag the classic warning signs: TCPI more than
  0.10 above the cumulative CPI (Christensen), and an EAC below the IEAC range.
- **[idea] IPMDAR reader.** The government's monthly EVM delivery is JSON
  (the Contract Performance Dataset) under DI-MGMT-81861. Needs the official
  file format specification in hand before writing it: don't guess the
  schema. The spec is published at acq.osd.mil ("IPMDAR Contract Performance
  Dataset File Format Specification"), but that host did not serve the
  cloud container used on 2026-09-25 (certificate chain and a 503), so it
  needs fetching from a machine that can reach it. The reader only has to
  map the CPD tables onto `EvmData.from_frame`'s columns. CPR formats 1 to 5 in Excel as a fallback.
- **[idea] Bayesian EAC.** A prior on the final CPI from historical programs,
  updated monthly. Needs a public or releasable history of CPI trajectories
  to set the prior from; the SAR panel does not have one.
- **[idea] EVM into JCL.** Feed actual progress into the schedule network so
  the joint confidence updates monthly from the status date.

## 2. Methods that need compute

- **[idea] Hierarchical Bayesian CERs.** Pool strength across related
  programs so small-sample CERs (8 to 15 points) get honest uncertainty. Needs
  a sampler (PyMC or numpyro are permissive; check the SBOM licence check).
- **[idea] Conformal prediction intervals** around any CER or model: coverage
  guaranteed without the normality assumptions small samples rarely meet.
- **[idea] Cost growth model from the SAR panel.** About 1,000 reports over 15
  years: predict a program's growth from its attributes and early signals,
  validated out of sample on programs whose outcomes are known. Useful for ICE
  review and for the AoA's growth factors.
- **[idea] Global sensitivity (Sobol)** alongside the tornado, which misses
  interactions.
- **[idea] Vectorised or GPU Monte Carlo** (numpy batching; JAX or CuPy
  optional) so million-draw runs and big sweeps are interactive.
- **[idea] Portfolio optimisation under uncertainty.** Optimise over many
  sampled cost-growth futures: the most value that stays affordable 80% of
  the time, rather than on point estimates.

## 3. Data and formats

- **[idea] Primavera P6 XER reader** beside the Microsoft Project reader.
- **[started] CSDR / FlexFile ingest.** Branch
  `feature/csdr-cer-correlated-risk` is unmerged; finish and merge it.
- **[idea] Official inflation indices** (OSD and service JIC tables) with
  their source and date.
- **[idea] Schedule risk register and probabilistic branching** into the JCL,
  and schedule margin handled explicitly.

## 4. Adoption

- **[idea] An estimate as a file.** One version-controlled spec (WBS,
  methods, data, inflation, uncertainty) that rebuilds the whole estimate.
- **[idea] Generated basis of estimate and GAO compliance report** mapped to
  the GAO Cost Guide's 12 steps and 4 characteristics.
- **[idea] Excel output with live formulas** for reviewers who audit in Excel.
- **[idea] Worked tutorials on public data**: one real program from its SARs
  to a CER, an estimate, schedule risk and a JCL.

## Cautions

- Contractor cost and EVM data is not public. The CSDR and IPMDAR readers can
  only be tested on synthetic files and the published formats until someone
  runs them at a lab.
- An AI assistant for reading contracts and BOEs would have to run locally and
  be optional to be usable with CUI. Leave it until the core is there.
- Every new dependency goes through the SBOM licence check; keep new ones in
  optional extras.
