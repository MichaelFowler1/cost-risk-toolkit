"""
widgets.py - The data-entry grid and input helpers for the desktop tool.

A small scrollable spreadsheet of Entry widgets, because the way an analyst
actually gets lot data into a tool is by pasting a block out of Excel. Ctrl+V
on any cell fills the grid from that row down, splitting on tabs, commas or
runs of spaces, and growing the grid as needed.

Ported unchanged from the original script.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

# Invented sample data for the "Load Example" buttons. Not from any real or
# supplied programme -- it exists so a first-time user has something to press
# Run on, and so the tests have a fixed reference to assert against.
EXAMPLE_ANALOGY = [
    ("2018", "8", "3120.00"),
    ("2019", "16", "2585.50"),
    ("2020", "24", "2402.75"),
    ("2021", "24", "2438.10"),
    ("2022", "18", "2310.40"),
    ("2023", "18", "2266.85"),
]

EXAMPLE_ESTIMATE = [
    ("2030", "6", "1.0"),
    ("2031", "12", "1.0"),
    ("2032", "12", "1.0"),
    ("2033", "12", "1.0"),
    ("2034", "12", "1.0"),
    ("2035", "12", "1.0"),
    ("2036", "12", "1.0"),
    ("2037", "6", "1.0"),
]


def parse_float(text: str) -> float:
    """Parse a number, tolerating $ signs and thousands separators."""
    return float(text.replace("$", "").replace(",", "").strip())


def split_row(line: str) -> list[str]:
    """Split a pasted row on tabs, commas, or runs of spaces."""
    line = line.rstrip("\r")
    if "\t" in line:
        parts = line.split("\t")
    elif "," in line:
        parts = line.split(",")
    else:
        parts = line.split()
    return [p.strip() for p in parts]


def default_output_dir() -> str:
    """Somewhere guaranteed writable: the script's folder, else Documents."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        probe = os.path.join(here, ".__writetest")
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
        return here
    except Exception:
        return os.path.join(os.path.expanduser("~"), "Documents")


class LotGrid(ttk.Frame):
    """A small scrollable spreadsheet of Entry widgets."""

    def __init__(self, parent, headers: list[str], widths: list[int]):
        super().__init__(parent)
        self.headers = headers
        self.widths = widths
        self.rows: list[list[tk.Entry]] = []

        head = ttk.Frame(self)
        head.pack(fill="x", padx=(4, 0))
        ttk.Label(head, text="#", width=4, anchor="center").grid(
            row=0, column=0, padx=1
        )
        for i, (h, w) in enumerate(zip(headers, widths)):
            ttk.Label(
                head, text=h, width=w, anchor="center", style="Head.TLabel"
            ).grid(row=0, column=i + 1, padx=1)

        canvas_wrap = ttk.Frame(self)
        canvas_wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_wrap, highlightthickness=0, height=240)
        scroll = ttk.Scrollbar(
            canvas_wrap, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_wheel(self, event):
        try:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
        except Exception:
            pass

    def add_row(self, values: tuple = None):
        r = len(self.rows)
        ttk.Label(self.body, text=str(r + 1), width=4, anchor="center").grid(
            row=r, column=0, padx=1, pady=1
        )
        entries = []
        for i, w in enumerate(self.widths):
            e = tk.Entry(self.body, width=w, justify="right")
            e.grid(row=r, column=i + 1, padx=1, pady=1)
            if values and i < len(values):
                e.insert(0, values[i])
            e.bind("<Control-v>", self._on_paste)
            e.bind("<Return>", lambda ev: self.add_row())
            entries.append(e)
        self.rows.append(entries)
        self.body.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        return entries

    def _on_paste(self, event):
        """Multi-line clipboard content fills the grid from this row down."""
        try:
            data = self.clipboard_get()
        except tk.TclError:
            return None
        if "\n" not in data.strip():
            return None  # single value: let Tk paste normally

        widget = event.widget
        start = 0
        for idx, row in enumerate(self.rows):
            if widget in row:
                start = idx
                break

        lines = [ln for ln in data.split("\n") if ln.strip()]
        for offset, line in enumerate(lines):
            target = start + offset
            while target >= len(self.rows):
                self.add_row()
            parts = split_row(line)
            for col, entry in enumerate(self.rows[target]):
                entry.delete(0, tk.END)
                if col < len(parts):
                    entry.insert(0, parts[col])
        return "break"

    def delete_last(self):
        if not self.rows:
            return
        for w in self.body.grid_slaves(row=len(self.rows) - 1):
            w.destroy()
        self.rows.pop()
        self.body.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def clear(self):
        while self.rows:
            self.delete_last()

    def load(self, data: list):
        self.clear()
        for values in data:
            self.add_row(values)

    def get_rows(self) -> list[list[str]]:
        """Non-empty rows as raw strings."""
        out = []
        for row in self.rows:
            vals = [e.get().strip() for e in row]
            if any(v for v in vals):
                out.append(vals)
        return out


