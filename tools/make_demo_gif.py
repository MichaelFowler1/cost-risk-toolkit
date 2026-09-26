#!/usr/bin/env python3
# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Generate docs/demo.gif - the 20-second demo at the top of the README.

Runs the real ``ce-core demo cost-risk`` in a scratch folder, then draws a
terminal typing the command and printing what it actually printed, followed
by the two charts it actually wrote. Nothing is mocked up, so rerunning this
after a release keeps the GIF true to what the tool does.

Needs Pillow (it comes with matplotlib) and a monospaced font; on Windows it
uses Consolas, elsewhere DejaVu Sans Mono.

Run:  python tools/make_demo_gif.py
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cost_core import __version__, cli  # noqa: E402

OUT = ROOT / "docs" / "demo.gif"
W, H = 820, 560
PAD, TOP = 18, 40
BG, BAR, INK, DIM, GREEN, AMBER, BLUE = (
    "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#3fb950", "#d29922", "#58a6ff")
COLOURS = {"$": GREEN, "What this means:": AMBER, "Top drivers": AMBER,
           "Cost risk,": AMBER, "Now with your own data:": AMBER}


def font(size, bold=False):
    names = (["consolab.ttf"] if bold else ["consola.ttf"]) + \
        (["DejaVuSansMono-Bold.ttf"] if bold else ["DejaVuSansMono.ttf"])
    for name in names:
        for folder in (Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
                       Path("/usr/share/fonts/truetype/dejavu")):
            if (folder / name).is_file():
                return ImageFont.truetype(str(folder / name), size)
    return ImageFont.load_default()


MONO, MONO_B, TITLE, SUB = font(14), font(14, True), font(30, True), font(17)
LINE = 18
ROWS = (H - TOP - PAD) // LINE
COLS = int((W - 2 * PAD) // MONO.getlength("M"))


def run_demo() -> tuple[list[str], Path]:
    """Run the demo for real; its printed lines and its output folder."""
    work = Path(tempfile.mkdtemp(prefix="ce-core-gif-"))
    old = Path.cwd()
    os.chdir(work)
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cli.main(["demo", "cost-risk"])
    finally:
        os.chdir(old)
        logging.disable(logging.NOTSET)
    lines = []
    for line in buf.getvalue().splitlines():
        # The example's path on this machine says nothing to a reader.
        line = re.sub(r"\S*[\\/]cost_core[\\/]examples[\\/]", "", line)
        if line.startswith("Example data:"):
            continue
        # Wrap what a real terminal this wide would wrap.
        lines.extend(textwrap.wrap(line, COLS, subsequent_indent="  ",
                                   drop_whitespace=False) or [""])
    return lines, work / "ce-core-demo" / "cost-risk"


def chrome(img: Image.Image, title: str) -> ImageDraw.ImageDraw:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 28], fill=BAR)
    for i, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        d.ellipse([12 + i * 18, 9, 22 + i * 18, 19], fill=c)
    d.text((W // 2, 14), title, fill=DIM, font=MONO, anchor="mm")
    return d


def terminal(lines: list[str], cursor: bool = False) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = chrome(img, "Command Prompt")
    shown = lines[-ROWS:]
    for i, line in enumerate(shown):
        y = TOP + i * LINE
        colour = INK
        for start, c in COLOURS.items():
            if line.startswith(start):
                colour = c
        if line.startswith("$ "):
            d.text((PAD, y), "$", fill=GREEN, font=MONO_B)
            d.text((PAD + 16, y), line[2:], fill=INK, font=MONO_B)
        else:
            d.text((PAD, y), line, fill=colour, font=MONO)
    if cursor:
        last = shown[-1] if shown else ""
        x = PAD + int(MONO.getlength(last.replace("$ ", "$  ", 1)))
        y = TOP + (len(shown) - 1) * LINE
        d.rectangle([x + 2, y + 2, x + 10, y + LINE - 2], fill=INK)
    return img


def card(title: str, lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 170), title, fill=INK, font=TITLE, anchor="mm")
    for i, (text, colour) in enumerate(lines):
        d.text((W // 2, 240 + i * 34), text, fill=colour, font=SUB, anchor="mm")
    return img


def picture(path: Path, caption: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = chrome(img, path.name)
    pic = Image.open(path).convert("RGB")
    room_w, room_h = W - 2 * PAD, H - TOP - 44
    pic.thumbnail((room_w, room_h), Image.LANCZOS)
    img.paste(pic, ((W - pic.width) // 2, TOP + 4 + (room_h - pic.height) // 2))
    d.text((W // 2, H - 20), caption, fill=DIM, font=SUB, anchor="mm")
    return img


def build() -> Path:
    printed, out = run_demo()
    frames: list[tuple[Image.Image, int]] = []

    def add(img, ms):
        frames.append((img, ms))

    add(card("cost-core", [("Cost risk from an estimate kept in Excel", DIM),
                           ("S-curve, confidence levels, what drives the risk", DIM)]), 2200)

    screen: list[str] = []

    def type_command(cmd: str):
        for k in range(0, len(cmd) + 1, 2):
            add(terminal(screen + [f"$ {cmd[:k]}"], cursor=True), 45)
        screen.append(f"$ {cmd}")
        add(terminal(screen), 350)

    type_command("pip install cost-core")
    screen.append(f"Successfully installed cost-core-{__version__}")
    screen.append("")
    add(terminal(screen), 900)

    type_command("ce-core demo cost-risk")
    # Print the real output a few lines at a time, pausing on the parts
    # that matter so they can be read.
    body = [line for line in printed if not line.startswith("Running:")]
    i = 0
    while i < len(body):
        step = 3 if not body[i].strip() else 2
        screen.extend(body[i:i + step])
        i += step
        pause = 60
        if any(s.startswith("  P90") for s in body[i - step:i]):
            pause = 2300
        elif any(s.startswith("What this means") for s in body[i - step:i]):
            pause = 300
        elif any("Narrowing those ranges" in s for s in body[i - step:i]):
            pause = 2600
        add(terminal(screen), pause)
    add(terminal(screen), 1200)

    add(picture(out / "cost_risk_s_curve.png",
                "The S-curve: what each level of confidence costs"), 3200)
    add(picture(out / "cost_risk_drivers.png",
                "What drives the risk. report.xlsx and brief.pptx hold it all."), 3000)
    add(card("Try it", [("pip install cost-core", GREEN),
                        ("ce-core demo cost-risk", GREEN),
                        ("ce-core template cost-risk   (your own estimate)", GREEN),
                        ("Runs on your machine. Nothing is sent anywhere.", DIM)]), 3500)

    # One shared palette keeps the file small and the colours steady.
    # Built from one of each kind of frame (cards, terminal, both charts), so
    # every colour that appears anywhere gets a slot.
    kinds = (0, len(frames) - 4, -3, -2, -1)
    sheet = Image.new("RGB", (W * len(kinds), H))
    for j, idx in enumerate(kinds):
        sheet.paste(frames[idx][0], (W * j, 0))
    pal = sheet.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    images = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f, _ in frames]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(OUT, save_all=True, append_images=images[1:],
                   duration=[ms for _, ms in frames], loop=0, optimize=True, disposal=1)
    total = sum(ms for _, ms in frames) / 1000
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, "
          f"{total:.1f} s)")
    return OUT


if __name__ == "__main__":
    build()
