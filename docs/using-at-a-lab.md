# Using cost-core at a lab or government office

This page is for someone bringing `cost-core` into a research lab, a
university-affiliated research center, or a government cost office. It covers
what a software approval review usually asks, how to install on a managed
machine, and what the library does and does not do with the network and your
data. Each claim here can be checked against the code or the release, and the
page says where.

## Licence

`cost-core` is under the [PolyForm Noncommercial License 1.0.0](../LICENSE)
(versions 1.0.0 and 1.0.1 were Apache-2.0 and stay so). Its *Noncommercial
Organizations* clause reads:

> Use by any charitable organization, educational institution, public
> research organization, public safety or health organization, environmental
> protection organization, or government institution is use for a permitted
> purpose regardless of the source of funding or obligations resulting from
> the funding.

So the question for a lab is whether it is one of those organisations, not
whether the work is sponsored. A university, a university's research
division, a federal agency, or a public research organisation is covered even
when the work is on a government contract. This page doesn't give legal
advice. If your contracts or technology transfer office needs to confirm the
lab is covered, give them the clause above. If the answer is no, for example
because the use is by a commercial contractor, ask for a commercial licence
through [the issue tracker](https://github.com/MichaelFowler1/cost-risk-toolkit/issues).

Anyone who passes a copy on, even inside the organisation, has to pass on the
licence terms and the `Required Notice:` line in [NOTICE](../NOTICE) with it.
The wheel already carries both files.

## What it installs, and under what licences

Every GitHub release has two files attached:

* `cost-core-X.Y.Z.cdx.json`, a CycloneDX software bill of materials (SBOM)
  listing every package a full install pulls in, with versions and licences;
* `cost-core-X.Y.Z-licences.md`, the same list as a table.

Both come from installing that release's wheel with every extra
(`[plots,public,optimize]`) into a clean environment. CI builds the same SBOM
on every push and every Monday, and fails if any dependency is copyleft or
has no licence it can read (`tools/sbom_licences.py`), so a new dependency
with a licence a review would flag is caught before a release, not in your
review.

With every extra installed, all 31 dependencies are under permissive
licences: MIT, BSD, Apache-2.0, PSF, and a few close relatives (MIT-0,
MIT-CMU, 0BSD, Zlib, CC0 inside numpy). Two packages don't declare a licence
in a form the SBOM tool can read, so the check records what their sources
say:

* **orloge** (a log parser PuLP 4 depends on) has no licence in its PyPI
  metadata. Its repository's `LICENSE.txt` is the MIT licence.
* **pypdfium2** (pdfplumber's renderer) declares "BSD-3-Clause, Apache-2.0"
  in free text. The PDFium binary it bundles carries permissive licences,
  listed under `BUILD_LICENSES` in the wheel.

Portfolio optimisation uses the HiGHS solver (MIT). Before the unreleased
version after 2.2.0 it used CBC, which is under the Eclipse Public License, a
weak copyleft. If your review saw CBC on an earlier version, that is why it
is gone.

Only the core is required: numpy, pandas, scipy and openpyxl. Everything else
is an optional extra, so installing only what you need also keeps the list a
reviewer has to read short:

| Extra | Adds | For |
|---|---|---|
| *(none)* | numpy, pandas, scipy, openpyxl | the engine, CERs, risk simulation, AoA, JCL, Excel workbooks |
| `plots` | matplotlib | the charts |
| `public` | pdfplumber | reading SAR PDFs |
| `optimize` | PuLP, highspy | portfolio optimisation |

## Installing on a managed machine

Python 3.9 to 3.14, on Windows, Linux or macOS. Use a virtual environment,
so the install doesn't need administrator rights or touch the system Python:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux or macOS: source .venv/bin/activate
pip install "cost-core[plots]"
```

**Behind a proxy that inspects TLS.** If pip fails with a certificate error,
point it at the organisation's CA bundle rather than turning verification
off:

```bash
pip config set global.cert path/to/org-ca-bundle.pem
```

**Through an internal package mirror** (Artifactory, Nexus and the like), set
it as the index:

```bash
pip config set global.index-url https://mirror.example.org/api/pypi/pypi/simple
```

**On a machine with no internet at all**, build a wheelhouse on a connected
machine with the same operating system and Python version, carry it across,
and install from it:

```bash
# connected machine
pip download "cost-core[plots,public,optimize]" -d wheelhouse
# offline machine
pip install --no-index --find-links wheelhouse "cost-core[plots,public,optimize]"
```

**To pin exactly what was reviewed**, install the versions in the release's
SBOM. `requirements.txt` in the repository is the exact stack the numbers
were recorded against.

**Where the package comes from.** Releases are published to PyPI by the
GitHub workflow `.github/workflows/publish.yml` through PyPI Trusted
Publishing. No API token exists that could publish from anywhere else, and
only a `vX.Y.Z` tag matching the version in `pyproject.toml` can publish.

## Network access and your data

**Only one part of the library ever touches the network:**
`cost_core.public`, when it fetches SARs or lists the Wayback Machine's
index. You can check this with `grep -rl urllib cost_core`: `fetch.py` and
`catalog.py` are the only hits. Everything else, from fitting and simulation
to the workbooks, charts, AoA, JCL and portfolio, runs entirely offline. There
is no telemetry, update check or licence server.

When `cost_core.public` does go out, it asks the official government URL and
then the Internet Archive, identifying itself with the user agent
`cost-core (+https://github.com/MichaelFowler1/cost-risk-toolkit)`. It sends
no data about you or your work, only the URL of a public document. Downloads
are cached under `$COST_CORE_CACHE`, or the user cache directory if that
isn't set.

**To build a SAR panel with no network at all**, put the PDFs in a folder,
from DAMIR or copied off the reading room by hand, and point the panel at it:

```bash
ce-core sar-panel --dir path/to/sars --out panel/
```

Nothing is fetched. Each row records the file's path and SHA-256, as it
would for a download.

**Your own data stays where you put it.** The library reads the files you
give it and writes only where you tell it to (`--out` and the like), plus
the download cache above. It is a library, not a service, so it is as
suitable for sensitive or controlled data as the machine it runs on. The one
way data leaves is if you send it: don't paste controlled or proprietary
numbers into a public GitHub issue. A synthetic reproduction from
`cost_core.synth` shows a bug just as well.

## Reproducibility

The same inputs and seed give the same numbers:

* every simulation takes a seed, and `ce-core full-run` records its seed in
  the `ASSUMPTIONS.md` it writes;
* the test suite pins the engine's numbers against recorded goldens, and
  holds the frozen random draws to a hash;
* `numpy` and `scipy` carry upper bounds (`<2.5`, `<1.18`) because newer
  releases move numbers in the last few digits. The bounds, and what moves
  above them, are in `pyproject.toml` and `CHANGELOG.md`.

To check the numbers on your own machine, run the test suite from the
source release, which carries the tests and the goldens:

```bash
pip download --no-deps --no-binary :all: cost-core && tar xzf cost_core-*.tar.gz
cd cost_core-*/ && pip install ".[plots,public,optimize]" pytest && pytest tests/ -q
```

## Citing it

`CITATION.cff` in the repository gives the citation for each release, and
GitHub's "Cite this repository" button reads it.
