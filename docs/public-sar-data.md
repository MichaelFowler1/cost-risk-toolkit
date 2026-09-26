# Real programs from public SARs

*Part of the [cost-core README](https://github.com/MichaelFowler1/cost-risk-toolkit#readme).*

## Real programs from public SARs

Every major defense program reports to Congress each year in a Selected
Acquisition Report, and DoD releases them. Their unit cost pages are the
longest public record of cost growth there is: what each program expected a
unit to cost when it started, and what it expects now, in constant dollars.

```bash
pip install "cost-core[public]"
ce-core sar-panel --list-cycles
ce-core sar-panel --programs F-35 --out f35/
```

The second command reads the F-35's SARs from December 2010 to the FY 2027
budget and writes three files: `unit_cost.csv` (PAUC and APUC, baseline beside
current estimate, per subprogram and report), `reports.csv` (what was read and
what wasn't, and why) and `checks.csv`. Leave off `--programs` for all of them,
about a thousand reports, which takes an hour or two the first time and
seconds after, since every download is cached.

**On a machine with no internet.** Point it at a folder of SAR PDFs you
already have, from DAMIR, the reading room, or a drive someone handed you,
and nothing touches the network:

```bash
ce-core sar-panel --dir path/to/sars --out panel/
```

Each file's reporting cycle comes from the reading room's folder name if you
kept it (`FY_2014_SARS`, `June_2025_MSARs`), otherwise from the file name
("F-35_SAR_Dec_2017.pdf", "AAG_MSAR_FY2027_PB.pdf"). A file whose name gives no
cycle is still read, and its own date line lands in `report_label`. The file's
path and SHA-256 go beside every row, the same as for a download.

A few things worth knowing before you use the numbers:

- **The files come from the Internet Archive.** The reading room that
  publishes them (esd.whs.mil) refuses scripted downloads, so the library
  asks the Wayback Machine for its copy of each official URL instead, and
  records the capture it used and the file's SHA-256 beside every row.
- **Three templates are read.** The SAR layout changed in 2021 and again for
  the modernised MSAR in December 2023; all three parse, including scans
  whose OCR text layer garbles the headers.
- **Every row is checked against its own arithmetic.** Unit cost times
  quantity has to come back to cost, and the printed percentage change has to
  match the unit costs. Across every cycle, 979 of 1,006 reports read and
  11,856 of 11,919 checks held (99.5%). The failures looked at were errors
  in the SARs themselves (an OCR layer that dropped a decimal point, a
  printed percentage that contradicts its own unit costs), and they stay in
  the table, marked, rather than being quietly fixed. The 27 reports that
  didn't read are archive copies that are truncated in every capture, and
  reports with no unit cost table.
- **Then-year tables are marked.** A report on a program in breach repeats
  its unit cost tables in then-year dollars; those rows have
  `dollars == "TY"` and no base year, and growth analysis skips them.
- **"Original" can be reset.** After a critical Nunn-McCurdy breach the
  original baseline can be revised, which takes the breach out of the growth
  figure. Seventeen programs show it moving between reports, so growth
  against the SAR's original baseline is a floor for the programs that grew
  most.
- **Base years differ between blocks.** A SAR can state its current baseline
  in one base year and its original baseline in another (SDB II, December
  2022: BY2015 and BY2010), so compare growth percentages across programs,
  not dollars, unless you convert them.

In Python:

```python
from cost_core.public import build_sar_panel
panel = build_sar_panel(programs=["DDG 51"], progress=print)
panel.unit_cost[["cycle", "measure", "comparison", "unit_cost_growth_pct"]]

# or, offline, from PDFs already on disk
from cost_core.public import local_catalog
panel = build_sar_panel(catalog=local_catalog("path/to/sars"))
```
