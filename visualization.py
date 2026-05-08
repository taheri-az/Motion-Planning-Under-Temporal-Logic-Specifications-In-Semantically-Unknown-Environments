import os
from itertools import product as iproduct

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle


_DEFAULT_PALETTE = ['#FFFF66', '#4080FF', '#66CC66', '#FF4444',
                    '#9D4EDD', '#06AED5', '#A4036F', '#588157']


def _atom_marginals(prob_grid, regions):
    labels = [
        ' && '.join(regions[i] if bits[i] else f'!{regions[i]}' for i in range(len(regions)))
        for bits in iproduct([0, 1], repeat=len(regions))
    ]
    atom_to_idxs = {
        r: [j for j, lbl in enumerate(labels)
            if not lbl.split(' && ')[i].startswith('!')]
        for i, r in enumerate(regions)
    }
    return [
        {r: sum(cell_probs[j] for j in atom_to_idxs[r]) for r in regions}
        for cell_probs in prob_grid
    ]


def _atom_color_map(regions):
    return {r: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
            for i, r in enumerate(regions or [])}


def _draw_step(ax, n, m, traj, prob_grid, regions, atom_color, panel_label):
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if panel_label is not None:
        ax.set_xlabel(panel_label, fontsize=11)

    if regions is not None and prob_grid is not None:
        marginals = _atom_marginals(prob_grid, regions)
        for cell, margs in enumerate(marginals):
            row = n - 1 - cell // n
            col = cell % n
            top_atom, top_p = max(margs.items(), key=lambda kv: kv[1])
            if top_p < 1e-3:
                continue
            ax.add_patch(Rectangle((col, row), 1, 1,
                                    facecolor=atom_color[top_atom],
                                    alpha=0.4 + 0.6 * top_p,
                                    edgecolor='black', linewidth=1, zorder=1))
            if top_p > 1 - 1e-3:
                ax.text(col + 0.5, row + 0.5, top_atom.upper(),
                        ha='center', va='center',
                        fontsize=14, fontweight='bold', zorder=2)
            else:
                ax.text(col + 0.5, row + 0.5,
                        f"{top_atom.upper()}: {top_p:.2f}",
                        ha='center', va='center', fontsize=8, zorder=2)

    for i in range(m + 1):
        ax.axvline(x=i, color='black', linewidth=0.5)
    for i in range(n + 1):
        ax.axhline(y=i, color='black', linewidth=0.5)

    if traj:
        s = traj[0]
        sr = n - 1 - s // n
        sc = s % n
        ax.add_patch(Rectangle((sc, sr), 1, 1,
                                facecolor='lightgray',
                                edgecolor='black', linewidth=1, zorder=1))
        ax.text(sc + 0.05, sr + 0.95, 'Start',
                ha='left', va='top', fontsize=8, zorder=3)

    for i in range(1, len(traj)):
        a, b = traj[i - 1], traj[i]
        if a == b:
            continue
        ar, ac = n - 1 - a // n, a % n
        br, bc = n - 1 - b // n, b % n
        ax_, ay_ = ac + 0.5, ar + 0.5
        bx_, by_ = bc + 0.5, br + 0.5
        color = 'red' if i == len(traj) - 1 else 'black'
        ax.annotate('', xy=(bx_, by_), xytext=(ax_, ay_),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    zorder=4)


def generate_grid_environment(n, m, traj_history, prob_history,
                              regions=None, interval_ms=800, save_path=None):
    # one frame per step. traj_history[i] is the position at step i,
    # prob_history[i] is the belief at the same step.
    # save_path: .gif -> pillow, .mp4 -> ffmpeg.
    # keep the returned anim around, otherwise matplotlib drops it before plt.show().
    if not isinstance(prob_history[0][0], list):
        # single grid was passed, use it for every frame
        prob_history = [prob_history] * len(traj_history)

    num = min(len(traj_history), len(prob_history))
    if num == 0:
        return None

    fig, ax = plt.subplots(figsize=(max(5, m * 0.9), max(5, n * 0.9)))
    atom_color = _atom_color_map(regions)

    def draw_frame(i):
        ax.clear()
        _draw_step(ax, n, m, traj_history[:i + 1], prob_history[i],
                   regions, atom_color, panel_label=f"Step {i + 1} / {num}")

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=num, interval=interval_ms,
        repeat=True, repeat_delay=1500,
    )

    if save_path:
        fps = max(1, int(round(1000 / interval_ms)))
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.mp4':
            anim.save(save_path, writer='ffmpeg', fps=fps)
        else:
            anim.save(save_path, writer='pillow', fps=fps)

    return anim


