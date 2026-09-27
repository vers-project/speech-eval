"""Figures for displacement results — generic, static, publication-shaped.

Three forms, each answering a different question about the same table:

    displacement_scatter  did it move by the right amount?   (the argument)
    cell_heatmap          where does it fail?                (the diagnosis)
    group_slopes          how much does it depend on who?    (the honesty)

These are matplotlib figures destined for a paper, so the interaction and theme-switching
parts of a web charting method do not apply — there is no hover layer and no dark mode to
select.  What does apply, and is followed here:

* **Diverging data gets a diverging map with a neutral midpoint.**  Signed error is
  diverging by nature: zero is "nothing happened" and must read as nothing.  A sequential
  ramp would make zero a colour, and a rainbow would make a bright band out of an
  arbitrary value.  Blue↔gray↔red, symmetric about zero, always.
* **Categorical hues in fixed order, capped at three.**  A scatter puts every pair of
  series on screen at once, and past three slots the default palette cannot hold both the
  colour-vision and the normal-vision separation floors.  Asking for a fourth raises
  rather than cycling a hue nobody can distinguish; facet instead.
* **Recessive chrome.**  Hairline grid, muted axes, ink-coloured text.  The marks carry
  the data; the frame should not compete.
* **A legend whenever there is more than one series**, so identity is never colour alone.

matplotlib is imported inside the functions: it is an optional extra, and importing
``speech_eval`` must not require it.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

#: Categorical slots, in fixed order.  Three only — see the module docstring.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MAX_SERIES = len(SERIES)

#: Diverging poles and their neutral midpoint.  Two hues that read as opposite, and a gray
#: centre so that "no difference" is not a colour.
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Figures need matplotlib, which is an optional extra: install with "
            "`--extra cpu` (or your project's equivalent)."
        ) from error
    return plt


def style_axes(ax, *, grid: str = "both") -> None:
    """Recessive chrome: hairline grid behind the marks, muted spines, ink labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_PRIMARY)
    if grid:
        ax.grid(True, axis=grid, color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


def diverging_cmap(name: str = "displacement"):
    """Blue → neutral gray → red, for a quantity symmetric about zero."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(DIVERGING), N=256)


def displacement_scatter(
    table: pd.DataFrame,
    *,
    metric: str | None = None,
    hue: str | None = None,
    ax=None,
    unit: str = "",
    title: str | None = None,
    point_size: float = 12.0,
    fit: bool = True,
):
    """Achieved against requested displacement, with the identity diagonal drawn.

    A perfect system puts every point on the diagonal.  Downward transformations land on
    the left, upward on the right, identity cases at the origin — nothing is averaged away,
    which is the whole reason this is a scatter and not a bar.

    ``hue`` colours by a categorical column, capped at three levels: a scatter shows every
    pair of series simultaneously, and past three the palette cannot hold its separation
    floors.  Beyond that, facet.
    """
    plt = _plt()
    data = table if metric is None else table[table["metric"] == metric]
    data = data.dropna(subset=["requested", "achieved"])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2), dpi=150)

    # The diagonal first, so the marks sit on top of it.
    if len(data):
        span = np.concatenate([data["requested"].to_numpy(), data["achieved"].to_numpy()])
        lo, hi = float(np.nanmin(span)), float(np.nanmax(span))
        pad = 0.05 * (hi - lo or 1.0)
        limits = (lo - pad, hi + pad)
        ax.plot(limits, limits, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                zorder=1, label="_perfect")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)

    if hue is None:
        ax.scatter(data["requested"], data["achieved"], s=point_size, c=SERIES[0],
                   linewidths=0.4, edgecolors=SURFACE, alpha=0.75, zorder=3)
    else:
        levels = list(pd.unique(data[hue].dropna()))
        if len(levels) > MAX_SERIES:
            raise ValueError(
                f"hue={hue!r} has {len(levels)} levels; a scatter validates at most "
                f"{MAX_SERIES} categorical slots because every pair is on screen at once. "
                f"Facet by {hue!r} instead, or map it to the heatmap axes."
            )
        for colour, level in zip(SERIES, levels):
            part = data[data[hue] == level]
            ax.scatter(part["requested"], part["achieved"], s=point_size, c=colour,
                       linewidths=0.4, edgecolors=SURFACE, alpha=0.8, zorder=3,
                       label=str(level))
        legend = ax.legend(frameon=False, fontsize=8, loc="upper left",
                           title=hue, title_fontsize=8)
        for text in legend.get_texts():
            text.set_color(INK_PRIMARY)

    if fit and len(data) > 2:
        slope, intercept = np.polyfit(data["requested"], data["achieved"], 1)
        xs = np.array(ax.get_xlim())
        ax.plot(xs, slope * xs + intercept, color=INK_PRIMARY, linewidth=1.4, zorder=4)
        ax.annotate(f"slope {slope:.2f}", xy=(0.04, 0.94), xycoords="axes fraction",
                    fontsize=9, color=INK_PRIMARY, va="top")

    suffix = f" ({unit})" if unit else ""
    ax.set_xlabel(f"requested displacement{suffix}", fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel(f"achieved displacement{suffix}", fontsize=9, color=INK_PRIMARY)
    ax.set_title(title or (metric or ""), fontsize=10, color=INK_PRIMARY, loc="left")
    style_axes(ax)
    return ax


def proportion_curve(
    table: pd.DataFrame,
    *,
    x: str,
    value: str = "accuracy",
    low: str = "ci_low",
    high: str = "ci_high",
    n: str | None = "n_trials",
    reference: float | None = 0.5,
    reference_label: str = "chance",
    ax=None,
    xlabel: str = "",
    ylabel: str = "proportion correct",
    title: str | None = None,
    color: str | None = None,
):
    """A proportion with its interval against a binned covariate.

    The form for a psychometric curve: magnitude over an ordered covariate, with
    the uncertainty shown rather than implied.  One series, so there is no legend
    -- the title names it -- and no number on every point; the interval is the
    annotation.

    ``reference`` draws the recessive line the whole panel is read against.  For a
    two-alternative task that is 0.5, and without it a reader cannot tell a real
    effect from the floor: an accuracy of 0.54 and one of 0.80 look equally like
    "some blue dots" until chance is on the canvas.

    ``n`` labels each point with its trial count when given.  Bin populations vary
    by design here, and an interval alone does not say whether a wide one comes
    from few trials or from genuine disagreement.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 2.8), dpi=150)
    hue = color or SERIES[0]

    order = table.sort_values(x)
    xs = order[x].to_numpy(dtype=float)
    ys = order[value].to_numpy(dtype=float)

    if reference is not None:
        ax.axhline(reference, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=1)
        # At the LEFT edge: the right-hand end carries the last point and its
        # count label, and the annotation lands on top of them there.
        ax.annotate(reference_label, xy=(xs[0], reference), xytext=(0, 3),
                    textcoords="offset points", fontsize=7, color=INK_MUTED,
                    ha="left", va="bottom")

    if low in order and high in order:
        ax.fill_between(xs, order[low].to_numpy(dtype=float),
                        order[high].to_numpy(dtype=float),
                        color=hue, alpha=0.16, linewidth=0, zorder=2)
    ax.plot(xs, ys, color=hue, linewidth=2.0, zorder=3)
    # A 2px surface ring keeps overlapping markers legible where bins crowd.
    ax.plot(xs, ys, "o", color=hue, markersize=6.5, markeredgecolor=SURFACE,
            markeredgewidth=2.0, zorder=4)

    if n is not None and n in order:
        for xi, yi, ni in zip(xs, ys, order[n]):
            ax.annotate(f"{int(ni)}", xy=(xi, yi), xytext=(0, -11),
                        textcoords="offset points", fontsize=6.5,
                        color=INK_MUTED, ha="center")

    ax.set_xlabel(xlabel or x, fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_PRIMARY)
    if title:
        ax.set_title(title, fontsize=10, color=INK_PRIMARY, loc="left")
    ax.set_ylim(0.0, 1.02)
    style_axes(ax)
    return ax


def cell_heatmap(
    cells: pd.DataFrame,
    *,
    metric: str | None = None,
    row_col: str,
    col_col: str,
    value: str = "mean",
    labels: Sequence[str] | None = None,
    row_label: str = "source",
    col_label: str = "target",
    ax=None,
    unit: str = "",
    title: str | None = None,
    symmetric: bool = True,
    value_label: str | None = None,
):
    """Source × target grid of a signed quantity, on a diverging scale centred at zero.

    ``symmetric`` forces the colour limits to ±max|value| so that zero sits at the neutral
    midpoint.  Without it the midpoint drifts to wherever the data's mean happens to be,
    and a cell with no error stops looking like one.
    """
    plt = _plt()
    data = cells if metric is None else cells[cells["metric"] == metric]
    grid = data.pivot(index=row_col, columns=col_col, values=value)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 3.4), dpi=150)

    finite = grid.to_numpy(dtype=float)
    bound = float(np.nanmax(np.abs(finite))) if np.isfinite(finite).any() else 1.0
    image = ax.imshow(
        finite, cmap=diverging_cmap(), vmin=-bound if symmetric else None,
        vmax=bound if symmetric else None, aspect="auto",
    )

    ticks = labels if labels is not None else [str(v) for v in grid.columns]
    ax.set_xticks(range(len(grid.columns)), ticks, fontsize=8)
    ax.set_yticks(range(len(grid.index)),
                  ticks if labels is not None else [str(v) for v in grid.index],
                  fontsize=8)
    ax.set_xlabel(col_label, fontsize=9, color=INK_PRIMARY)
    ax.set_ylabel(row_label, fontsize=9, color=INK_PRIMARY)
    ax.set_title(title or (metric or ""), fontsize=10, color=INK_PRIMARY, loc="left")

    # The number in every cell: a 4x4 grid is small enough that direct labels beat a
    # colourbar lookup, and they are the relief for readers who cannot separate the hues.
    for i in range(finite.shape[0]):
        for j in range(finite.shape[1]):
            if not np.isfinite(finite[i, j]):
                continue
            shade = abs(finite[i, j]) / (bound or 1.0)
            ax.text(j, i, f"{finite[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if shade > 0.6 else INK_PRIMARY)

    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(f"{value_label or 'mean signed error'}"
                  f"{f' ({unit})' if unit else ''}",
                  fontsize=8, color=INK_PRIMARY)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)
    ax.tick_params(colors=INK_MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def group_slopes(
    slopes: pd.DataFrame,
    *,
    group_col: str = "subject_id",
    ax=None,
    title: str | None = None,
    reference: float = 1.0,
):
    """One dot per group per measure — voice-to-voice spread at a glance.

    More honest than a bar with an error bar: with few groups the bar implies a precision
    the sample does not have, while the dots show exactly how many there are and how far
    apart they sit.  ``reference`` marks a perfect slope; zero marks "ignored the request".
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 0.5 * max(slopes["metric"].nunique(), 1) + 1.4),
                             dpi=150)

    measures = list(pd.unique(slopes["metric"]))
    for y, measure in enumerate(measures):
        part = slopes[slopes["metric"] == measure]
        ax.scatter(part["slope"], np.full(len(part), y), s=34, c=SERIES[0],
                   edgecolors=SURFACE, linewidths=0.6, alpha=0.85, zorder=3)
        ax.scatter([part["slope"].mean()], [y], s=90, marker="|", c=INK_PRIMARY,
                   linewidths=1.6, zorder=4)

    ax.axvline(reference, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(measures)), measures, fontsize=8)
    ax.set_ylim(-0.6, len(measures) - 0.4)
    ax.set_xlabel("achieved / requested slope, one dot per "
                  f"{group_col} (| = mean)", fontsize=9, color=INK_PRIMARY)
    ax.set_title(title or "", fontsize=10, color=INK_PRIMARY, loc="left")
    style_axes(ax, grid="x")
    return ax


def save(ax_or_fig: Any, path, *, tight: bool = True) -> None:
    """Write a figure to ``path`` and close it, so a long loop does not leak figures."""
    plt = _plt()
    figure = getattr(ax_or_fig, "figure", ax_or_fig)
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight" if tight else None, facecolor=SURFACE)
    plt.close(figure)
