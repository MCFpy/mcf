"""Draw fitted policy trees and save PDF and JPEG figures."""
from dataclasses import dataclass, field
from io import BytesIO
from math import sqrt
from pathlib import Path
import re
from textwrap import fill
from typing import Any, TYPE_CHECKING, TypedDict

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from pandas import DataFrame

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy


class TreeFigure(TypedDict):
    """Saved figures and vector artwork used by the PDF report."""

    pdf: Path
    jpeg: Path
    svg: bytes
    width: float
    height: float


@dataclass
class _PlotNode:
    """One displayed split or terminal treatment leaf."""

    label: str
    depth: int
    observations: int
    treatment: int | None = None
    children: list[int] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    height: float = 0.0
    span: float = 0.0


def save_policy_tree_figure(optp_: 'OptimalPolicy',
                            data_df: DataFrame,
                            data_title: str = '',
                            ) -> TreeFigure | None:
    """Export the fitted tree using the observations actually used for training.

    The returned SVG artwork is retained in ``optp_.report`` by the caller.
    This lets the reporting module embed a vector figure without requiring
    additional files. Standalone PDF and JPEG figures use unique filenames.
    """
    if not optp_.gen_cfg.with_output:
        return None

    nodes = _build_plot_tree(optp_.pt_cfg.policy_tree, data_df)
    outpath = Path(optp_.gen_cfg.outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    title = str(data_title or '')
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', title).strip('_')[:60]
    stem = 'policy_tree' + (f'_{name}' if name else '')
    path = outpath / stem
    counter = 1
    while path.with_suffix('.pdf').exists() or path.with_suffix('.jpeg').exists():
        path = outpath / f'{stem}_{counter}'
        counter += 1

    # Use a local style and a non-interactive canvas without changing pyplot.
    style = {'font.family': 'DejaVu Sans', 'font.weight': 'normal',
             'font.style': 'normal', 'text.usetex': False,
             'svg.fonttype': 'path', 'pdf.fonttype': 42, 'savefig.bbox': None,
             }
    with rc_context(style):
        figure = _draw_tree(nodes, title)
        width, height = map(float, figure.get_size_inches())
        try:
            figure.savefig(path.with_suffix('.pdf'), format='pdf', facecolor='white')
            # Bound raster memory for very large trees; the PDF remains vector.
            dpi = min(float(optp_.int_cfg.dpi), 12000 / max(width, height),
                      sqrt(24000000 / (width * height))
                      )
            figure.savefig(path.with_suffix('.jpeg'), format='jpeg', dpi=dpi,
                           facecolor='white', pil_kwargs={'quality': 95})
            svg = BytesIO()
            figure.savefig(svg, format='svg', facecolor='white',
                           metadata={'Date': None, 'Creator': None, 'Format': None, 'Type': None}
                           )
            artwork = svg.getvalue()
            if optp_.int_cfg.show_plots:
                _show_tree_figure(path.with_suffix('.jpeg'), width, height)

        finally:
            figure.clear()

    return {'pdf': path.with_suffix('.pdf'), 'jpeg': path.with_suffix('.jpeg'),
            'svg': artwork, 'width': width, 'height': height}


def _show_tree_figure(path: Path, width: float, height: float) -> None:
    """Display the saved tree in a resizable Matplotlib window."""
    scale = min(1.0, 12.0 / width, 8.0 / height)
    figure = plt.figure(figsize=(width * scale, height * scale))
    try:
        axis = figure.add_axes((0, 0, 1, 1))
        axis.imshow(plt.imread(path))
        axis.set_axis_off()
        plt.show()
    finally:
        plt.close(figure)
        figure.clear()


def _build_plot_tree(policy_tree: list[list[Any]], data_df: DataFrame) -> list[_PlotNode]:
    """Expand stored terminal branches into leaves, using prediction's rules."""
    if not policy_tree:
        raise ValueError('No fitted policy tree is available for plotting.')
    by_id = {node[0]: node for node in policy_tree}
    roots = [node[0] for node in policy_tree if node[1] is None]
    if len(by_id) != len(policy_tree) or len(roots) != 1:
        raise ValueError('Plotting requires unique node IDs and one root.')

    terminal = [node for node in policy_tree if node[2] is None or node[3] is None]
    leaf_names = {}
    for position, node in enumerate(terminal):
        sides = [side for side in (0, 1) if node[side + 2] is None]
        for index, side in enumerate(sides):
            leaf_names[node[0], side] = f'Leaf{position}{index}'

    arrays = {node[5]: data_df[node[5]].to_numpy(copy=False) for node in policy_tree}
    nodes: list[_PlotNode] = []
    visited: set[int] = set()

    def visit(node_id: int, indices: np.ndarray, depth: int) -> int:
        """Add a split and its two branches in preorder."""
        if node_id in visited or node_id not in by_id:
            raise ValueError('The fitted tree contains a repeated or missing child node.')
        visited.add(node_id)
        source = by_id[node_id]
        values = arrays[source[5]][indices]
        if source[6] == 'unord':
            goes_left = np.isin(np.round(values), tuple(source[7]))
            categories = ', '.join(str(value) for value in sorted(source[7]))
            condition = 'in {' + categories + '}'
        else:
            goes_left = values <= source[7]
            condition = '<= ' + str(source[7])
        label = (fill(str(source[5]), width=26) + '\n' + fill(condition, width=26)
                 + f'\nn = {len(indices):,}')
        node_index = len(nodes)
        nodes.append(_PlotNode(label, depth, len(indices)))
        for side, selected in enumerate((indices[goes_left], indices[~goes_left])):
            child_id = source[side + 2]
            if child_id is None:
                treatment = int(source[8][side])
                label = (leaf_names[node_id, side]
                         + f'\nTreatment {treatment} | n = {len(selected):,}'
                         )
                child_index = len(nodes)
                nodes.append(_PlotNode(label, depth + 1, len(selected), treatment))
            else:
                child_index = visit(child_id, selected, depth + 1)
            nodes[node_index].children.append(child_index)
        return node_index

    visit(roots[0], np.arange(len(data_df), dtype=np.intp), 0)
    if len(visited) != len(policy_tree):
        raise ValueError('The fitted tree contains nodes that are unreachable from its root.')

    return nodes


def _draw_tree(nodes: list[_PlotNode], data_title: str) -> Figure:
    """Lay out a complete tree from left to right, without overlapping boxes."""
    font_size = 10.0
    figure = Figure(figsize=(1, 1), dpi=72)
    canvas = FigureCanvasAgg(figure)
    renderer = canvas.get_renderer()
    font = FontProperties(family='DejaVu Sans', size=font_size)
    width = max(renderer.get_text_width_height_descent(line, font, False)[0] / 72
                for node in nodes for line in node.label.splitlines()
                ) + 0.28
    width = max(1.9, width)
    line_height = font_size * 1.25 / 72
    for node in reversed(nodes):
        node.height = len(node.label.splitlines()) * line_height + 0.18
        child_span = sum(nodes[child].span for child in node.children)
        if node.children:
            child_span += 0.16
        node.span = max(node.height, child_span)

    def position(index: int, top: float) -> None:
        """Centre every subtree in its reserved vertical interval."""
        node = nodes[index]
        node.x = 0.32 + node.depth * (width + 0.7)
        node.y = top + node.span / 2
        if node.children:
            left, right = (nodes[child] for child in node.children)
            child_top = top + (node.span - left.span - right.span - 0.16) / 2
            position(node.children[0], child_top)
            position(node.children[1], child_top + left.span + 0.16)

    figure_width = 0.64 + width + max(node.depth for node in nodes) * (width + 0.7)
    title = 'Fitted policy tree' + (f' - {data_title}' if data_title else '')
    title = fill(title, width=max(30, int(figure_width * 8)))
    header_height = 0.35 + 0.23 * len(title.splitlines())
    note = ('Yes: the condition holds; No: it does not. '
            'n counts training observations. Leaf labels match the table.'
            )
    note = fill(note, width=max(30, int(figure_width * 12)))
    footer_height = 0.3 + 0.16 * len(note.splitlines())
    figure_height = header_height + nodes[0].span + footer_height
    position(0, header_height)
    figure.set_size_inches(figure_width, figure_height)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, figure_width)
    axis.set_ylim(figure_height, 0)
    axis.set_axis_off()
    axis.text(0.32, 0.16, title, fontsize=13, fontweight='bold', color='#18324f',
              va='top', parse_math=False,
              )
    palette = ('#d9eaf7', '#f9e3cb', '#dbeddb', '#eadcf1', '#f6dadd', '#d5eeed')
    for node in nodes:
        for side, child_index in enumerate(node.children):
            child = nodes[child_index]
            start = node.x + width
            axis.plot((start, child.x), (node.y, child.y), color='#718096', linewidth=0.9,
                      zorder=1)
            axis.text(start + 0.42, node.y + (child.y - node.y) * 0.6,
                      'Yes' if side == 0 else 'No', fontsize=8, ha='center', va='center',
                      color='#334155', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1},
                      zorder=2, parse_math=False,
                      )
        color = '#f2f5f9' if node.treatment is None else palette[node.treatment % len(palette)]
        box = FancyBboxPatch((node.x, node.y - node.height / 2), width, node.height,
                             boxstyle='round,pad=0.015,rounding_size=0.06',
                             linewidth=0.9, edgecolor='#68788c', facecolor=color, zorder=3,
                             )
        axis.add_patch(box)
        axis.text(node.x + width / 2, node.y, node.label, ha='center', va='center',
                  fontsize=font_size, linespacing=1.25, color='#15283f', parse_math=False,
                  zorder=4,
                  )
    axis.text(0.32, header_height + nodes[0].span + 0.18, note, fontsize=8,
              va='top', color='#475569', parse_math=False)

    return figure
