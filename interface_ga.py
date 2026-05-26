"""
interface_ga.py
===============
GUI interface for configuring and running the DES blade repair simulation.

Two input sections:
  1. Primary Blades  – 12 tracks split into two side-by-side columns of 6.
  2. Buffer Blades   – up to 3 optional rows, gated by an "Include" checkbox.

Clicking "Run Simulation":
  1. Validates all active fields.
  2. Writes current_status.csv with the entered hours.
  3. Spawns des_simulationBCtester_current.py and streams its output to the
     built-in console panel.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_CSV   = os.path.join(SCRIPT_DIR, "current_status.csv")
INPUT_CSV    = os.path.join(SCRIPT_DIR, "input_jobs.csv")
SIM_SCRIPT   = os.path.join(SCRIPT_DIR, "des_simulation_sa_change.py")

MAX_TRACKS  = 12
MAX_BUFFERS = 3


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class BladeSimInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blade Repair Simulation — Interface")
        self.minsize(940, 580)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        # ── Primary blades – two columns of 6 ────────────────────────
        prim_lf = ttk.LabelFrame(top, text="Primary Blades  (Repair Tracks)",
                                  padding=6)
        prim_lf.pack(fill="x", pady=(0, 5))

        # (hv, av, entry, wv, fv, spinbox, force_cb)  — order matters for _sync/_collect
        self._prim: list[tuple] = []

        def _col_headers(parent: ttk.Frame) -> None:
            for col, lbl, w in [
                (0, "Track",      9), (1, "h rem.",   11),
                (2, "Workers",    9), (3, "Force",     7),
                (4, "Ph.1 done", 10), (5, "Active",    7),
            ]:
                ttk.Label(parent, text=lbl, width=w,
                          anchor="w" if col == 1 else "center").grid(
                    row=0, column=col, padx=2, pady=(0, 3))

        def _add_half(parent: ttk.Frame, offset: int) -> None:
            _col_headers(parent)
            for k in range(6):
                i = offset + k
                hv  = tk.StringVar()
                av  = tk.BooleanVar(value=True)
                wv  = tk.IntVar(value=5)
                fv  = tk.BooleanVar(value=False)
                p1v = tk.BooleanVar(value=True)   # phase 1 already done
                ttk.Label(parent, text=f"Track {i+1:>2}",
                          anchor="e", width=9).grid(
                    row=k+1, column=0, padx=2, pady=1)
                e = ttk.Entry(parent, textvariable=hv, width=11)
                e.grid(row=k+1, column=1, padx=3, pady=1, sticky="w")
                sb = ttk.Spinbox(parent, from_=1, to=9, increment=1,
                                 textvariable=wv, width=5)
                sb.grid(row=k+1, column=2, padx=2, pady=1)
                fcb = ttk.Checkbutton(parent, variable=fv)
                fcb.grid(row=k+1, column=3, pady=1)
                p1cb = ttk.Checkbutton(parent, variable=p1v)
                p1cb.grid(row=k+1, column=4, pady=1)
                ttk.Checkbutton(parent, variable=av,
                                command=self._sync).grid(
                    row=k+1, column=5, pady=1)
                # tuple: hv, av, entry, wv, fv, sb, fcb, p1v, p1cb
                self._prim.append((hv, av, e, wv, fv, sb, fcb, p1v, p1cb))

        lf = ttk.Frame(prim_lf)
        lf.pack(side="left", padx=(0, 4))
        ttk.Separator(prim_lf, orient="vertical").pack(
            side="left", fill="y", padx=6)
        rf = ttk.Frame(prim_lf)
        rf.pack(side="left", padx=(4, 0))

        _add_half(lf, 0)
        _add_half(rf, 6)

        # ── Middle row: Buffer blades + Simulation settings ───────────
        mid = ttk.Frame(top)
        mid.pack(fill="x", pady=(0, 5))

        # Buffer blades
        buf_lf = ttk.LabelFrame(mid, text="Buffer Blades  (up to 3)", padding=6)
        buf_lf.pack(side="left", anchor="n", padx=(0, 5))

        for col, lbl, w in [
            (0, "Slot",    7), (1, "h rem.", 9), (2, "Incl.", 5),
        ]:
            ttk.Label(buf_lf, text=lbl, width=w,
                      anchor="w" if col == 1 else "center").grid(
                row=0, column=col, padx=2, pady=(0, 3))

        self._buf: list[tuple[tk.StringVar, tk.BooleanVar, ttk.Entry]] = []
        for i in range(MAX_BUFFERS):
            hv = tk.StringVar()
            av = tk.BooleanVar(value=False)
            ttk.Label(buf_lf, text=f"Buffer {i+1}", anchor="e",
                      width=7).grid(row=i+1, column=0, padx=2, pady=2)
            e = ttk.Entry(buf_lf, textvariable=hv, width=9)
            e.grid(row=i+1, column=1, padx=3, pady=2, sticky="w")
            ttk.Checkbutton(buf_lf, variable=av,
                            command=self._sync).grid(row=i+1, column=2, pady=2)
            self._buf.append((hv, av, e))

        # Simulation settings
        set_lf = ttk.LabelFrame(mid, text="Simulation Settings", padding=6)
        set_lf.pack(side="left", fill="both", expand=True, anchor="n")

        ttk.Label(set_lf, text="Max generated jobs:", anchor="w").grid(
            row=0, column=0, sticky="w", padx=4, pady=3)
        self._max_gen = tk.IntVar(value=35)
        ttk.Spinbox(set_lf, from_=0, to=500, increment=1,
                    textvariable=self._max_gen, width=7).grid(
            row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(set_lf,
                  text="PERT-generated blades per buffer dispatch  (0 = disabled)",
                  foreground="#555").grid(row=0, column=2, sticky="w", padx=6)

        ttk.Label(set_lf, text="Oven ready in (h):", anchor="w").grid(
            row=1, column=0, sticky="w", padx=4, pady=3)
        self._oven_offset = tk.DoubleVar(value=0.0)
        ttk.Spinbox(set_lf, from_=0, to=200, increment=1,
                    textvariable=self._oven_offset, width=7).grid(
            row=1, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(set_lf,
                  text="Hours until oven accepts first blade  (0 = immediately)",
                  foreground="#555").grid(row=1, column=2, sticky="w", padx=6)

        ttk.Label(set_lf, text="B-workers available:", anchor="w").grid(
            row=2, column=0, sticky="w", padx=4, pady=3)
        self._workers_b = tk.IntVar(value=35)
        ttk.Spinbox(set_lf, from_=1, to=200, increment=1,
                    textvariable=self._workers_b, width=7).grid(
            row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(set_lf, text="Total B-worker pool (used in phase 2)",
                  foreground="#555").grid(row=2, column=2, sticky="w", padx=6)

        ttk.Label(set_lf, text="C-workers available:", anchor="w").grid(
            row=3, column=0, sticky="w", padx=4, pady=3)
        self._workers_c = tk.IntVar(value=25)
        ttk.Spinbox(set_lf, from_=1, to=200, increment=1,
                    textvariable=self._workers_c, width=7).grid(
            row=3, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(set_lf, text="Total C-worker pool (used in phases 1 and 3)",
                  foreground="#555").grid(row=3, column=2, sticky="w", padx=6)

        # ── Controls ──────────────────────────────────────────────────
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=8, pady=(2, 4))

        self._run_btn = ttk.Button(ctrl, text="▶  Run Simulation",
                                    command=self._on_run)
        self._run_btn.pack(side="left", padx=(0, 12))

        self._status = tk.StringVar(value="Ready.")
        ttk.Label(ctrl, textvariable=self._status,
                  foreground="#444").pack(side="left")

        # ── Output console ────────────────────────────────────────────
        out_lf = ttk.LabelFrame(self, text="Simulation Output", padding=6)
        out_lf.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._console = scrolledtext.ScrolledText(
            out_lf,
            state="disabled",
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            wrap="none",
        )
        self._console.pack(fill="both", expand=True)

        self._sync()

    # ------------------------------------------------------------------
    # Entry enable/disable based on checkboxes
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        for row in self._prim:
            s = "normal" if row[1].get() else "disabled"
            row[2].config(state=s)   # hours entry
            row[5].config(state=s)   # workers spinbox
            row[6].config(state=s)   # force checkbox
            row[8].config(state=s)   # ph.1 done checkbox
        for _, av, e in self._buf:
            e.config(state="normal" if av.get() else "disabled")

    # ------------------------------------------------------------------
    # Collect & validate input
    # ------------------------------------------------------------------

    def _collect(self) -> list[tuple[float, int, int]] | None:
        # Each entry: (hours, buffer_flag, skip_phase1)
        rows: list[tuple[float, int, int]] = []

        for i, row in enumerate(self._prim):
            hv, av, p1v = row[0], row[1], row[7]
            if not av.get():
                continue
            try:
                h = float(hv.get().strip().replace(",", "."))
                if h <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid input",
                    f"Track {i+1}: enter a positive number of hours (e.g. 51.0).",
                )
                return None
            rows.append((h, 0, int(p1v.get())))

        if not rows:
            messagebox.showerror(
                "No primary blades",
                "Enable at least one primary blade track before running.",
            )
            return None

        for i, (hv, av, _) in enumerate(self._buf):
            if not av.get():
                continue
            try:
                h = float(hv.get().strip().replace(",", "."))
                if h <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid input",
                    f"Buffer {i+1}: enter a positive number of hours (e.g. 110.0).",
                )
                return None
            rows.append((h, 1, 0))   # buffer blades always have phase 1 remaining

        return rows

    # ------------------------------------------------------------------
    # Write current_status.csv
    # ------------------------------------------------------------------

    def _write_csv(self, rows: list[tuple[float, int, int]]) -> None:
        with open(STATUS_CSV, "w", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["total_remaining", "buffer", "skip_phase1"])
            for h, buf, skip in rows:
                w.writerow([h, buf, skip])

    def _write_input_csv(self) -> None:
        """Write a detailed CSV of all real input jobs (tracks + buffers, no generated)."""
        with open(INPUT_CSV, "w", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["job_type", "id", "hours_remaining",
                        "workers", "force_workers", "phase1_done"])
            for i, row in enumerate(self._prim):
                hv, av, _, wv, fv, _, _, p1v, _ = row
                if not av.get():
                    continue
                w.writerow([
                    "Track",
                    f"Track {i + 1}",
                    hv.get().strip().replace(",", "."),
                    wv.get(),
                    int(fv.get()),
                    int(p1v.get()),
                ])
            for i, (hv, av, _) in enumerate(self._buf):
                if not av.get():
                    continue
                w.writerow([
                    "Buffer",
                    f"Buffer {i + 1}",
                    hv.get().strip().replace(",", "."),
                    "",   # workers not applicable
                    "",   # force not applicable
                    0,    # phase 1 always remaining for buffers
                ])

    # ------------------------------------------------------------------
    # Console helpers
    # ------------------------------------------------------------------

    def _log(self, text: str) -> None:
        self._console.config(state="normal")
        self._console.insert("end", text)
        self._console.see("end")
        self._console.config(state="disabled")

    def _clear_log(self) -> None:
        self._console.config(state="normal")
        self._console.delete("1.0", "end")
        self._console.config(state="disabled")

    # ------------------------------------------------------------------
    # Run button handler
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        rows = self._collect()
        if rows is None:
            return

        self._write_csv(rows)
        self._write_input_csv()

        n_p = sum(1 for _, b, _ in rows if b == 0)
        n_b = sum(1 for _, b, _ in rows if b == 1)

        self._clear_log()
        self._log(
            f"Written to current_status.csv: "
            f"{n_p} primary blade(s), {n_b} buffer blade(s)\n"
        )
        self._log(f"Written to input_jobs.csv: {n_p + n_b} real job(s) (no generated)\n")
        self._log("=" * 60 + "\n")
        self._log("Launching des_simulation_sa_change.py …\n\n")

        # Build forced-worker string: "job_idx:workers,..." for checked rows
        force_parts = []
        job_idx = 0
        for row in self._prim:
            if not row[1].get():
                continue
            if row[4].get():
                force_parts.append(f"{job_idx}:{row[3].get()}")
            job_idx += 1
        force_str = ",".join(force_parts)

        self._run_btn.config(state="disabled")
        self._status.set("Running simulation…")
        threading.Thread(
            target=self._worker,
            args=(self._max_gen.get(), force_str, self._oven_offset.get(),
                  self._workers_b.get(), self._workers_c.get()),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Background simulation thread
    # ------------------------------------------------------------------

    def _worker(self, max_gen_jobs: int, force_str: str, oven_offset: float,
                workers_b: int, workers_c: int) -> None:
        cmd = [sys.executable, SIM_SCRIPT,
               "--max_gen_jobs", str(max_gen_jobs),
               "--oven_offset",  str(oven_offset),
               "--workers_b",    str(workers_b),
               "--workers_c",    str(workers_c)]
        if force_str:
            cmd += ["--force_workers", force_str]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                self.after(0, self._log, line)
            proc.wait()
            self.after(0, self._finish, proc.returncode)
        except Exception as ex:
            self.after(0, self._log, f"\n[ERROR] {ex}\n")
            self.after(0, self._finish, -1)

    def _finish(self, rc: int) -> None:
        self._run_btn.config(state="normal")
        if rc == 0:
            self._status.set("Simulation complete.")
            self._log("\n[Done — simulation finished successfully]\n")
        else:
            self._status.set(f"Simulation exited with code {rc}.")
            self._log(f"\n[Done — exit code {rc}]\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = BladeSimInterface()
    app.mainloop()
