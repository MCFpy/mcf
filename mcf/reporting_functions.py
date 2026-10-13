"""Created on Fri Dec  8 13:42:19 2023.

Contains classes and methods needed for mcf and optimal policy reporting.

# -*- coding: utf-8 -*-

@author: MLechner
"""
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace
from pandas import DataFrame

from mcf import mcf_general_sys as mcf_sys
from mcf.optpolicy_main import OptimalPolicy, OptimalPolicyVersions
from mcf import reporting_content as content
from mcf import reporting_content_optp as content_optp

if TYPE_CHECKING:
    from mcf.optpolicy_tree_plot import TreeFigure
    from mcf.reporting import McfOptPolReport


class PDF(FPDF):
    """Inherited class from fpdf."""

    def header(self):
        """Define header."""
        # Setting font: helvetica bold 12
        self.set_font("helvetica", "B", 16)
        # Calculating width of title and setting cursor position:
        available_width = self.w - self.l_margin - self.r_margin
        width = self.get_string_width(self.title) + 6
        if width > available_width:
            font_size = 16 * (available_width - 6) / (width - 6)
            self.set_font('helvetica', 'B', font_size)
            width = available_width
        self.set_x(self.l_margin + (available_width - width) / 2)
        # Setting colors for frame, background and text: red, green, blue
        self.set_draw_color(0, 128, 0)
        self.set_fill_color(144+30, 238+17, 144+30)
        self.set_text_color(0)
        # Setting thickness of the frame (1 mm)
        self.set_line_width(1)
        # Printing title:
        self.cell(width,
                  9,
                  self.title,
                  border=1,
                  new_x="LMARGIN",
                  new_y="NEXT",
                  align="C",
                  fill=True,
                  )
        # Performing a line break:
        self.ln(10)

    def footer(self):
        """Define footer."""
        # Setting position at 1.5 cm from bottom:
        self.set_y(-15)
        # Setting font: helvetica italic 8
        self.set_font("helvetica", "I", 8)
        # Setting text color to gray:
        self.set_text_color(128)
        # Printing page number
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self,
                      num_tuple: tuple[int, ...],
                      label: str,
                      color: str | None = None
                      ) -> None:
        """Define chapter title."""
        def _clip(x):  # ensure valid RGB ints
            return int(max(0, min(255, x)))

        # Setting font: helvetica 12
        darker = 20
        match len(num_tuple):
            case 1: size = 16
            case 2: size = 14
            case 3: size = 12
            case _: size = 11
        self.set_font('helvetica', '', size)

        if len(num_tuple) == 1:
            num = str(num_tuple[0])
        else:
            num = '.'.join([str(s) for s in num_tuple])
        # Setting background color: red, green, blue
        match color:
            case 'blue':
                self.set_fill_color(200, 220, 255)
            case 'green' | 'greenT' | 'greenP' | 'greenA' as c:
                step = {'green': 0, 'greenT': 1, 'greenP': 2, 'greenA': 3}[c]
                r, g = 144 - darker*step, 238 - darker*step
                b = 144 - darker*step
                self.set_fill_color(_clip(r), _clip(g), _clip(b))
            case 'red':
                self.set_fill_color(255, 127, 127)
            case 'yellow':
                self.set_fill_color(255, 255, 0)
            case _:
                self.set_fill_color(200)  # grey

        # Printing section name:
        self.cell(0, 6, f"Section {num}: {label}",
                  new_x="LMARGIN", new_y="NEXT", align="L", fill=True,
                  )
        # Performing a line break:
        self.ln(4)

    def section_body(self, txt: str):
        """Define chapter body."""
        # Setting font: Helvetica 12
        self.set_font("helvetica", size=12)
        # Printing justified text:
        self.multi_cell(0, h=5, text=txt, align='L')
        # Performing a line break:
        self.ln()

    def print_section(self,
                      num: tuple[int, ...] | None = None,
                      title: str | None = None,
                      txt: str | None = None,
                      color: str | None = None
                      ) -> None:
        """Print pdf to file."""
        if num is not None and len(num) == 1:
            self.add_page()
        if title is not None:
            self.section_title(num, title, color)
        if txt is not None:
            self.section_body(txt)

    def add_image(self,
                  title: str,
                  image_path: Path,
                  width: int = 90,
                  height: int = 90,
                  ) -> None:
        """Add figure."""
        self.set_font("helvetica", 'I', size=12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        y_before_image = self.get_y()
        self.image(image_path, x=10, y=y_before_image, w=width, h=height)
        self.ln()
        # Set the current position below the figure
        self.set_y(y_before_image + height)

    def add_policy_tree_figure(self, figure: 'TreeFigure', context: str = '') -> None:
        """Insert a vector tree page and resume the previous page format."""
        previous_orientation = self.cur_orientation
        previous_format = (self.w, self.h)
        short, long = sorted(previous_format)
        horizontal_margin = self.l_margin + self.r_margin
        vertical_margin = self.t_margin + self.b_margin + 25
        portrait_scale = min((short - horizontal_margin) / figure['width'],
                             (long - vertical_margin) / figure['height'],
                             )
        landscape_scale = min((long - horizontal_margin) / figure['width'],
                              (short - vertical_margin) / figure['height'],
                              )
        orientation = 'L' if landscape_scale > portrait_scale else 'P'
        self.add_page(orientation=orientation)
        self.set_font('helvetica', size=11)
        self.set_text_color(0)
        if context.strip():
            self.multi_cell(0, h=5, text=context.strip(), align='L',
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                            )
            self.ln(3)
        image_y = self.get_y()
        self.image(BytesIO(figure['svg']), x=self.l_margin, y=image_y,
                   w=self.w - self.l_margin - self.r_margin,
                   h=self.page_break_trigger - image_y - 2,
                   keep_aspect_ratio=True)
        # The caller writes the training text and leaf table on this next page.
        self.add_page(orientation=previous_orientation, format=previous_format)

    def add_figure_row(self,
                       title: str,
                       image_paths: list[Path, ...], *,
                       width: int = 80,
                       height: int = 40,
                       note: str | None = None
                       ) -> None:
        """Print figures in pdf."""
        self.set_font("helvetica", 'I', size=12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

        # Calculate the available width for each figure in a row
        available_width = self.w - 20  # 10 units margin on each side
        figure_width = available_width / 2
        # Filter unavailable figures before grouping them into rows.
        image_paths = [path for path in image_paths if path is not None and path.is_file()]

        for i in range(0, len(image_paths), 2):
            # Start a new row
            self.ln()
            # Keep the figure row above the bottom margin.
            if self.get_y() + height > self.page_break_trigger:
                self.add_page()

            # Add the first figure in the row
            self.image(image_paths[i], x=10, y=self.get_y(), w=width, h=height)

            # Check if there's a second figure in the row
            if i + 1 < len(image_paths):
                # Add the second figure in the row
                second_path = image_paths[i + 1]
                if second_path is not None and second_path.is_file():
                    self.image(second_path,
                               x=figure_width + 20, y=self.get_y(), w=width, h=height,
                               )
            self.set_y(self.get_y() + height)

        if note is not None:
            self.set_font("helvetica", size=8)
            self.multi_cell(0, h=5, text=note, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.ln()

    def add_table(self, title: str, data: DataFrame, *, col_width: int = 30,
                  note: str | None = None, font_size_table: int = 10,
                  index_label: str = 'Comparison', keep_first_columns: int = 0) -> None:
        """Print wrapped tables with repeated headings and panels for excess columns.

        Keep up to keep_first_columns leading data columns together in the first panel.
        """
        if col_width <= 0 or font_size_table <= 0 or keep_first_columns < 0:
            raise ValueError('Table widths and font sizes must be positive; column counts >= 0.')

        font_size_table = max(8, font_size_table)
        padding = 1.2
        available_width = self.w - self.l_margin - self.r_margin
        headers = [str(index_label), *(str(name) for name in data.columns)]
        rows = [[str(value) for value in row] for row in data.itertuples(index=True, name=None)]

        # Measure the displayed strings in their actual fonts, including cell padding.
        self.set_font('helvetica', size=font_size_table)
        body_widths = [max((self.get_string_width(row[col]) for row in rows), default=0.0)
                       for col in range(len(headers))
                       ]
        self.set_font('helvetica', 'I', size=font_size_table)
        header_widths = [self.get_string_width(text) for text in headers]
        label_width = min(max(float(col_width), body_widths[0] + 2 * padding,
                              header_widths[0] + 2 * padding), 0.35 * available_width)
        widths = [label_width]
        for body_width, header_width in zip(body_widths[1:], header_widths[1:], strict=True):
            width = max(12.0, body_width + 2 * padding + 0.2,
                        min(float(col_width), header_width + 2 * padding + 0.2)
                        )
            widths.append(min(width, available_width - label_width))

        # Preserve column order and keep the label column in every panel.
        keep_first_columns = min(keep_first_columns, len(headers) - 1)
        panel = list(range(1, keep_first_columns + 1))
        used_width = label_width + sum(widths[col] for col in panel)
        if used_width > available_width + 1e-8:
            raise ValueError('The columns requested together do not fit on this page.')
        panels: list[list[int]] = []
        for col in range(keep_first_columns + 1, len(headers)):
            if panel and used_width + widths[col] > available_width + 1e-8:
                panels.append(panel)
                panel = []
                used_width = label_width
            panel.append(col)
            used_width += widths[col]
        panels.append(panel)

        for panel_idx, panel in enumerate(panels):
            if panel_idx:
                self.add_page()
            self.set_x(self.l_margin)
            self.set_font('helvetica', size=font_size_table)
            columns = [0, *panel]
            panel_widths = [widths[col] for col in columns]
            if panel:
                extra_width = max(0.0, available_width - sum(panel_widths)) / len(panel)
                panel_widths = [label_width, *(width + extra_width for width in panel_widths[1:])]
            else:
                panel_widths = [available_width]
            panel_title = title
            if len(panels) > 1:
                panel_title += f' (part {panel_idx + 1} of {len(panels)})'

            # Both title and column labels repeat on continuation pages.
            saved_line_width = self.line_width
            self.set_line_width(0.2)
            with self.table(width=available_width, col_widths=panel_widths, align='LEFT',
                            text_align='RIGHT', v_align='MIDDLE', padding=padding,
                            line_height=self.font_size * 1.3, borders_layout='HORIZONTAL_LINES',
                            headings_style=FontFace(emphasis='ITALICS'), num_heading_rows=2,
                            repeat_headings='ON_TOP_OF_EVERY_PAGE'
                            ) as table:
                title_row = table.row()
                title_row.cell(panel_title, colspan=len(columns), align='CENTER',
                               style=FontFace(size_pt=12))
                heading_row = table.row()
                for col in columns:
                    heading_row.cell(headers[col], align='LEFT' if col == 0 else 'CENTER')
                if not rows:
                    table.row().cell('No rows to report.', colspan=len(columns), align='LEFT')
                for values in rows:
                    row = table.row()
                    for col in columns:
                        row.cell(values[col], align='LEFT' if col == 0 else 'RIGHT')
            self.set_line_width(saved_line_width)
            self.ln(3)

        if note is not None:
            self.set_font('helvetica', size=8)
            self.multi_cell(available_width, h=5, text='Note: ' + note, align='L',
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT
                            )
            self.ln(3)


def create_pdf_file(rep_o: 'McfOptPolReport') -> None:
    """Create a pdf report from the text."""
    pdf = PDF()
    pdf.set_title(rep_o.text['header'])
    pdf.set_author('The mcf estimation package')
    # grey: General info; green: mcf; red: sensitivity; blue: optimal policy
    pdf.print_section((1,), 'General information', rep_o.text['general'], color='grey')
    idx1 = 2
    if rep_o.mcf_o is not None:
        report_mcf_core(pdf, rep_o, idx1, iv=rep_o.iv)
        idx1 += 1
    if rep_o.sens_o is not None:
        report_mcf_sense(pdf, rep_o, idx1)
        idx1 += 1
    if rep_o.opt_o is not None:
        report_optpol(pdf, rep_o, idx1)

    pdf.output(rep_o.gen_cfg.outfilename)


def optpol_section_text(opt_o: OptimalPolicy, empty_lines_end: int) -> dict[str, object]:
    """Build section content for one optimal-policy instance."""
    text: dict[str, object] = {}
    if opt_o.report['training']:
        text['opt_training'] = content_optp.opt_training(opt_o, empty_lines_end)
    if opt_o.report['allocation']:
        text['opt_allocation'] = content_optp.opt_allocation(opt_o, empty_lines_end)
    if opt_o.report['evaluation']:
        text['opt_evaluation'] = content_optp.opt_evaluation(opt_o)
    if opt_o.report.get('solvefair', False) or opt_o.report.get('fairscores', False):
        text['opt_fairness'] = content_optp.opt_fairness(opt_o, empty_lines_end)
    if opt_o.report['estriskscores']:
        text['opt_estriskscores'] = content_optp.opt_estimationrisk(opt_o, empty_lines_end)
    return text


def report_optpol(pdf: PDF, rep_o: 'McfOptPolReport', idx1: int) -> None:
    """Write report on optimal policy allocations."""
    report_allocation = False          # Kept for potential later use
    idx2 = 1
    pdf.print_section((idx1,), "Optimal Policy", rep_o.text['opt_general'], 'blue')
    if rep_o.opt_o.report['opt_versions']:
        pdf.print_section((idx1, idx2),
                          "Optimal Policy: Treatment versions",
                          rep_o.text['opt_versions'], 'blue')
        idx2 += 1
        optp_all = rep_o.opt_o.optp
        if not isinstance(optp_all, list):
            optp_all = [optp_all]
    else:
        optp_all = [rep_o.opt_o]
    for idx, optp in enumerate(optp_all):
        if len(optp_all) > 1:
            if idx == 0:
                treats = ', '.join([str(s) for s in range(len(optp_all)-1)])
                add_text = f'\n\nMAIN TREATMENTS ({treats})\n'
            else:
                if optp is None:
                    continue
                add_text = f'\n\nTREATMENT VERSION OF MAIN TREATMENT {idx-1}\n\n'
        else:
            add_text = ''
        opt_text = rep_o.text if idx == 0 else optpol_section_text(optp, empty_lines_end=2)
        if optp.report.get('solvefair', False) or optp.report.get('fairscores', False):
            pdf.print_section((idx1, idx2),
                              "Optimal Policy: Protected variables (fairness)",
                              opt_text['opt_fairness'], 'blue'
                              )
            idx2 += 1

        if optp.report['estriskscores']:
            pdf.print_section((idx1, idx2),
                              'Optimal Policy: Adjustments for estimation risk of policy scores',
                              opt_text['opt_estriskscores'], 'blue'
                              )
            idx2 += 1
        if optp.report['training']:
            tree_figure = optp.report.get('training_tree_figure')
            if tree_figure is not None:
                pdf.add_policy_tree_figure(tree_figure, context=add_text)
            pdf.print_section((idx1, idx2), 'Optimal Policy: Training',
                              add_text + opt_text['opt_training'], 'blue',
                              )
            idx2 += 1

        if optp.report['allocation'] and report_allocation:
            pdf.print_section((idx1, idx2), "Optimal Policy: Allocation",
                              add_text + opt_text['opt_allocation'], 'blue',
                              )
            idx2 += 1

        if optp.report['evaluation']:
            general_text, txt_table_list = opt_text['opt_evaluation']
            pdf.print_section((idx1, idx2),
                              'Optimal Policy: Evaluation of Allocation(s)',
                              add_text + general_text, 'blue'
                              )
            idx2 += 1

            for idx, txt_table in enumerate(txt_table_list):
                pdf.add_table(f'Treatment allocation ({idx})', txt_table[1],
                              col_width=30, font_size_table=10, note=txt_table[0],
                              index_label='Allocation', keep_first_columns=2
                              )


def report_mcf_core(pdf: PDF,
                    rep_o: 'McfOptPolReport',
                    idx1: int,
                    iv: bool = False
                    ) -> None:
    """Write report on mcf training, prediction, and analysis."""
    idx2 = idx3 = 1
    pdf.print_section((idx1,), 'MCF estimation', rep_o.text['mcf_general'], 'green')
    pdf.print_section((idx1, idx2), 'MCF Training',
                      rep_o.text['mcf_training'], 'greenT')
    pdf.print_section((idx1, idx2, idx3),
                      'Preparation of training data (mcf training)',
                      rep_o.text['mcf_descriptives'], 'greenT'
                      )
    idx3 += 1
    if rep_o.mcf_o.fs_cfg.yes:
        pdf.print_section((idx1, idx2, idx3),
                          "Feature selection (mcf training)",
                          rep_o.text['mcf_feature_selection'], 'greenT',
                          )
        idx3 += 1
    if rep_o.mcf_o.cs_cfg.type_ > 0:
        pdf.print_section((idx1, idx2, idx3),
                          "Common support (mcf training)",
                          rep_o.text['mcf_t_common_support'], 'greenT',
                          )
        idx3 += 1
        title = "Common support plots"
        if iv:
            title += (' (instrument propensity score)')
        cs_figures = rep_o.mcf_o.report['cs_t_figs'][0] or []
        if cs_figures and rep_o.mcf_o.cs_cfg.type_ == 1:
            first_stem = cs_figures[0].stem.split('_page', 1)[0]
            cs_figures = [path for path in cs_figures
                          if path.stem.split('_page', 1)[0] != first_stem
                          ]
        if any('_page' in path.stem for path in cs_figures):
            for path in cs_figures:
                if not path.is_file():
                    continue
                pdf.add_page()
                pdf.set_font('helvetica', 'I', size=12)
                pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
                image_y = pdf.get_y()
                image_width = pdf.w - pdf.l_margin - pdf.r_margin
                image_height = pdf.page_break_trigger - image_y
                pdf.image(path, x=pdf.l_margin, y=image_y, w=image_width, h=image_height,
                          keep_aspect_ratio=True)
                pdf.set_y(pdf.page_break_trigger)
        else:
            pdf.add_figure_row(title, cs_figures, width=70, height=60)
    if rep_o.mcf_o.lc_cfg.yes:
        pdf.print_section((idx1, idx2, idx3),
                          "Local centering (mcf training)",
                          rep_o.text['mcf_local_center'], 'greenT',
                          )
        idx3 += 1
    if rep_o.mcf_o.gen_tv_cfg.yes:
        pdf.print_section((idx1, idx2, idx3),
                          "Version estimation (mcf training and prediction)",
                          rep_o.text['mcf_version'], 'greenT',
                          )
        idx3 += 1
    pdf.print_section((idx1, idx2, idx3), "Forest", rep_o.text['mcf_forest'], 'greenT')
    idx2 += 1
    idx3 = 1
    if any(key in rep_o.mcf_o.report for key in ('mcf_pred_results', 'mcf_pred_results_global',
                                                 'mcf_pred_results_local')
           ):
        pdf.print_section((idx1,), "MCF estimation", ' ', 'green')
        pdf.print_section((idx1, idx2), "MCF Prediction of Effects",
                          rep_o.text['mcf_prediction'], 'greenP'
                          )
        if rep_o.mcf_o.cs_cfg.type_ > 0:
            pdf.print_section((idx1, idx2, idx3),
                              "Common support (mcf prediction)",
                              rep_o.text['mcf_p_common_support'], 'greenP'
                              )
            idx3 += 1
        idx4 = 1
        pdf.print_section((idx1, idx2, idx3), "Results", rep_o.text['mcf_results'], 'greenP')
        titel_ = "LATE" if iv else "ATE"
        pdf.print_section((idx1, idx2, idx3, idx4), titel_,
                          rep_o.text['mcf_ate'][0], 'greenP')
        for idx_table in range(1, 3):
            if rep_o.text['mcf_ate'][idx_table] is None:
                break
            for idx, table_df in enumerate(rep_o.text['mcf_ate'][idx_table]):
                y_name = rep_o.mcf_o.var_cfg.y_name[idx]
                if len(y_name) > 3 and y_name[-3:] == '_lc':
                    y_name = y_name[:-3]
                table_df.columns = [name.replace('p_val', 'p-value') for name in table_df.columns]
                table_df.columns = [name.replace('t-val', 't-value') for name in table_df.columns]
                if not iv:
                    effect_label = 'ATE'
                elif idx_table == 1 and 'mcf_pred_results_global' in rep_o.mcf_o.report:
                    effect_label = 'LATE (global compliers)'
                else:
                    effect_label = 'Averaged local complier effects'
                pdf.add_table(f'{effect_label} for {y_name}', table_df, col_width=30,
                              note=rep_o.text['mcf_ate'][3]
                              )
        idx4 += 1
        if ('fig_gate_local' if iv else 'fig_gate') in rep_o.mcf_o.report:
            title = 'LGATE' if iv else 'GATE'
            pdf.print_section((idx1, idx2, idx3, idx4), title, ' ', 'greenP')
            pdf.add_figure_row(title, rep_o.text['mcf_gate'][0], width=70, height=60,
                               note=rep_o.text['mcf_gate'][1]
                               )
            idx4 += 1
        if ('fig_bgate_local' if iv else 'fig_bgate') in rep_o.mcf_o.report:
            title = 'LBGATE' if iv else 'BGATE'
            pdf.print_section((idx1, idx2, idx3, idx4), title,
                              rep_o.text['mcf_bgate'][2], 'greenP')
            pdf.add_figure_row(title, rep_o.text['mcf_bgate'][0], width=70, height=60,
                               note=rep_o.text['mcf_bgate'][1]
                               )
            idx4 += 1
        if ('fig_cbgate_local' if iv else 'fig_cbgate') in rep_o.mcf_o.report:
            title = 'LCBGATE' if iv else 'CBGATE'
            pdf.print_section((idx1, idx2, idx3, idx4), title, 'All covariates are balanced.',
                              'greenP'
                              )
            pdf.add_figure_row(title, rep_o.text['mcf_cbgate'][0], width=70, height=60,
                               note=rep_o.text['mcf_cbgate'][1]
                               )
            idx4 += 1

        qiate_figures = [path for path in (rep_o.mcf_o.report.get('fig_qiate') or [])
                         if path is not None and path.is_file()
                         ]
        if not iv and qiate_figures:
            pdf.print_section((idx1, idx2, idx3, idx4), 'QIATE',
                              'Quantile Individualized Average Treatment Effects.', 'greenP'
                              )
            pdf.add_figure_row('QIATE', qiate_figures, width=70, height=60,
                              note=('The figures compare treatments with the reference treatment. '
                                    'Their titles distinguish QIATE(q) from '
                                    'QIATE(q) - QIATE(0.5). Further results are available in the '
                                    'output files.')
                              )
            idx4 += 1

        if 'iate_text' in rep_o.mcf_o.report:
            title = 'LIATE' if iv else 'IATE'
            pdf.print_section((idx1, idx2, idx3, idx4), title, rep_o.text['iate_part1'],
                              color='greenP',
                              )
            for y_name, effect_type, table_df in rep_o.text['iate_tables']:
                table_title = title
                note = None
                if effect_type == 1:
                    table_title += ' minus average effect'
                    note = 'Weights are not censored for effects minus the average effect.'
                pdf.add_table(f'{table_title} for {y_name}', table_df, note=note)
            idx4 += 1

        if rep_o.mcf_o.p_cfg.bt_yes:
            pdf.print_section((idx1, idx2, idx3, idx4),
                              "Balancing checks (experimental)",
                              rep_o.text['mcf_balance'], 'greenP'
                              )
        idx2 += 1
        idx3 = 1
        if (rep_o.mcf_o.gen_cfg.with_output
            and rep_o.mcf_o.post_cfg.est_stats
            and rep_o.mcf_o.gen_cfg.return_iate_sp
                and rep_o.mcf_o.report.get('fig_iate') is not None):
            pdf.print_section((idx1,), "MCF estimation", ' ', 'green')
            pdf.print_section((idx1, idx2), "Analysis of Estimated IATEs",
                              rep_o.text['mcf_iate_part2'][0], 'greenA'
                              )
            if rep_o.mcf_o.report['fig_iate']:
                titel_ = 'LIATEs and IATEs' if iv else 'IATEs'
                pdf.add_figure_row(titel_, rep_o.text['mcf_iate_part2'][1], width=70, height=60,
                                   note=rep_o.text['mcf_iate_part2'][2]
                                   )
            knn_table = rep_o.text['mcf_iate_part2'][4]
            if knn_table is not None:
                pdf.print_section(txt=rep_o.text['mcf_iate_part2'][3])
                pdf.print_section(txt=knn_table['obs_cluster_str'])
                note = 'Mean of variable in cluster.'
                titel_ = 'LIATE ' if iv else 'IATE '
                pdf.add_table(titel_, knn_table['IATE_df'], col_width=30, font_size_table=10,
                              note=note
                              )
                pdf.add_table('Potential Outcomes ', knn_table['PotOutcomes_df'],
                              col_width=30, font_size_table=10, note=note
                              )
                pdf.add_table('Features ', knn_table['Features_df'],
                              col_width=30, font_size_table=8,
                              note=note +
                              ' Categorical variables are recoded to indicator (dummy) variables.'
                              )

    if 'alloc_mcf_pred_results' in rep_o.mcf_o.report:
        pdf.print_section((idx1,), "MCF estimation", ' ', 'green')
        pdf.print_section((idx1, idx2),
                          'MCF Prediction of average outcomes of different allocations',
                          rep_o.text['alloc_mcf_prediction'],
                          'greenP'
                          )
        if rep_o.mcf_o.cs_cfg.type_ > 0:
            pdf.print_section((idx1, idx2, idx3),
                              "Common support (mcf prediction)",
                              rep_o.text['alloc_mcf_p_common_support'],
                              'greenP'
                              )
            idx3 += 1
        idx4 = 1
        pdf.print_section((idx1, idx2, idx3), "Results", rep_o.text['alloc_mcf_results'], 'greenP')
        titel_ = 'Average outcome for all-in-1-treatment allocations'
        pdf.print_section((idx1, idx2, idx3, idx4), titel_, rep_o.text['alloc_mcf_ate'][0], 'greenP'
                          )
        for idx, table_df in enumerate(rep_o.text['alloc_mcf_ate'][1]):
            y_name = rep_o.mcf_o.var_cfg.y_name[idx]
            if len(y_name) > 3 and y_name[-3:] == '_lc':
                y_name = y_name[:-3]
            table_df.columns = [name.replace('p_val', 'p-value') for name in table_df.columns]
            table_df.columns = [name.replace('t-val', 't-value') for name in table_df.columns]
            pdf.add_table('Average outcome differences for ' + y_name, table_df,
                          col_width=30, note=rep_o.text['alloc_mcf_ate'][3],
                          )
        idx4 += 1

        welfare_tables = rep_o.mcf_o.report.get('alloc_welfare_tables') or []
        if welfare_tables or rep_o.mcf_o.report['alloc_welfare_allocations']:
            pdf.print_section((idx1, idx2, idx3, idx4),
                              'Welfare of different allocations',
                              rep_o.text['alloc_welfare_allocations'],
                              color='greenP',
                              )
            for outcome, table_df, note in welfare_tables:
                pdf.add_table(f'Allocation welfare for {outcome}', table_df,
                              note=note, index_label='Allocation',
                              )


def report_mcf_sense(pdf: PDF, rep_o: 'McfOptPolReport', idx1: int) -> None:
    """Write report on sensitivity analysis."""
    pdf.print_section((idx1,), "MCF Sensitivity Analysis", rep_o.text['sensitivity'], 'red')
    for title, table_df in rep_o.sens_o.report.get('sens_ate_tables') or []:
        pdf.add_table(title, table_df, note='Effects with p-values above 10% are omitted.')
    if rep_o.sens_o.report['sens_plots_iate'] is not None:
        pdf.add_figure_row('Placebo and estimated IATEs', rep_o.sens_o.report['sens_plots_iate'],
                           width=70, height=60, note='\n',
                           )


def create_text(rep_o: 'McfOptPolReport') -> None:
    """Create the dictionary containing all information to be printed."""
    empty_lines_end = 2
    iv = rep_o.iv
    rep_o.text['header'] = header_title(rep_o)
    rep_o.text['general'] = content.general(rep_o)
    if rep_o.mcf_o is not None:
        rep_o.text['mcf_general'] = content.mcf_general(rep_o.mcf_o, empty_lines_end, iv=rep_o.iv)
        rep_o.text['mcf_training'] = content.mcf_training(rep_o.mcf_o, empty_lines_end)
        rep_o.text['mcf_descriptives'] = content.mcf_descriptives(rep_o.mcf_o, empty_lines_end)
        rep_o.text['mcf_forest'] = content.mcf_forest(rep_o.mcf_o, empty_lines_end, iv=rep_o.iv)

        if rep_o.mcf_o.fs_cfg.yes:
            rep_o.text['mcf_feature_selection'] = content.mcf_feature_selection(rep_o.mcf_o,
                                                                                empty_lines_end,
                                                                                iv=rep_o.iv,
                                                                                )
        if rep_o.mcf_o.cs_cfg.type_ > 0:
            rep_o.text['mcf_t_common_support'] = content.mcf_common_support(rep_o.mcf_o,
                                                                            empty_lines_end - 1,
                                                                            train=True,
                                                                            iv=rep_o.iv,
                                                                            )
        if rep_o.mcf_o.lc_cfg.yes:
            rep_o.text['mcf_local_center'] = content.mcf_local_center(rep_o.mcf_o,
                                                                      empty_lines_end,
                                                                      iv=rep_o.iv,
                                                                      )
        if rep_o.mcf_o.gen_tv_cfg.yes:
            rep_o.text['mcf_version'] = content.mcf_version(rep_o.mcf_o.gen_tv_cfg,
                                                            empty_lines_end,
                                                            iv=rep_o.iv,
                                                            )

        # Keep the first effect prediction and latest allocation prediction.
        # Allocation welfare text also refers to the latest allocation run.
        predictions = rep_o.mcf_o.report['predict_list']
        if predictions:
            effect_keys = ('mcf_pred_results', 'mcf_pred_results_global',
                           'mcf_pred_results_local',
                           )
            for prediction in predictions:
                if any(key in prediction for key in effect_keys):
                    rep_o.mcf_o.report.update(prediction.copy())
                    break
            for prediction in reversed(predictions):
                if 'alloc_mcf_pred_results' in prediction:
                    rep_o.mcf_o.report.update(prediction.copy())
                    break
            rep_o.mcf_o.report['predict_list'] = []
        if rep_o.mcf_o.report['analyse_list']:
            sub_dict = rep_o.mcf_o.report['analyse_list'][0].copy()
            rep_o.mcf_o.report.update(sub_dict)
            rep_o.mcf_o.report['analyse_list'] = []

        if any(key in rep_o.mcf_o.report for key in ('mcf_pred_results', 'mcf_pred_results_global',
                                                     'mcf_pred_results_local')
               ):
            rep_o.text['mcf_prediction'] = content.mcf_prediction(
                rep_o.mcf_o, empty_lines_end)
            if rep_o.mcf_o.cs_cfg.type_ > 0:
                rep_o.text['mcf_p_common_support'] = content.mcf_common_support(rep_o.mcf_o,
                                                                                empty_lines_end - 1,
                                                                                train=False,
                                                                                iv=rep_o.iv,
                                                                                )
            rep_o.text['mcf_results'] = content.mcf_results(rep_o.mcf_o,
                                                            empty_lines_end,
                                                            iv=rep_o.iv
                                                            )
            rep_o.text['mcf_ate'] = content.mcf_ate_proc(rep_o.mcf_o, empty_lines_end,
                                                         iv=rep_o.iv, alloc=False,
                                                         )
            if ('fig_gate_local' if iv else 'fig_gate') in rep_o.mcf_o.report:
                rep_o.text['mcf_gate'] = content.mcf_gate(rep_o.mcf_o, empty_lines_end,
                                                          gate_type='gate', iv=rep_o.iv,
                                                          )
            if ('fig_bgate_local' if iv else 'fig_bgate') in rep_o.mcf_o.report:
                rep_o.text['mcf_bgate'] = content.mcf_gate(rep_o.mcf_o, empty_lines_end,
                                                           gate_type='bgate', iv=rep_o.iv,
                                                           )
            if ('fig_cbgate_local' if iv else 'fig_cbgate') in rep_o.mcf_o.report:
                rep_o.text['mcf_cbgate'] = content.mcf_gate(rep_o.mcf_o, empty_lines_end,
                                                            gate_type='cbgate', iv=rep_o.iv,
                                                            )
            if rep_o.mcf_o.p_cfg.bt_yes:
                rep_o.text['mcf_balance'] = content.mcf_balance(empty_lines_end, iv=rep_o.iv)
            if 'iate_text' in rep_o.mcf_o.report:
                rep_o.text['iate_part1'] = content.mcf_iate_part1(rep_o.mcf_o, empty_lines_end,
                                                                  iv=rep_o.iv,
                                                                  )
                rep_o.text['iate_tables'] = rep_o.mcf_o.report.get('iate_tables') or []

            if (rep_o.mcf_o.gen_cfg.with_output
                and rep_o.mcf_o.post_cfg.est_stats
                    and rep_o.mcf_o.gen_cfg.return_iate_sp):
                rep_o.text['mcf_iate_part2'] = content.mcf_iate_analyse(rep_o.mcf_o, 1, iv=rep_o.iv)

        if 'alloc_mcf_pred_results' in rep_o.mcf_o.report:
            rep_o.text['alloc_mcf_prediction'] = content.mcf_prediction(rep_o.mcf_o, empty_lines_end
                                                                        )
            if rep_o.mcf_o.cs_cfg.type_ > 0:
                (rep_o.text['alloc_mcf_p_common_support']
                 ) = content.mcf_common_support(rep_o.mcf_o, empty_lines_end - 1,
                                                train=False, iv=rep_o.iv, alloc=True
                                                )
            rep_o.text['alloc_mcf_results'] = content.mcf_results(rep_o.mcf_o, empty_lines_end,
                                                                  iv=rep_o.iv, alloc=True
                                                                  )
            rep_o.text['alloc_mcf_ate'] = content.mcf_ate_proc(rep_o.mcf_o, empty_lines_end,
                                                               iv=rep_o.iv, alloc=True
                                                               )
            rep_o.text['alloc_welfare_allocations'] = content.mcf_welfare_alloc(rep_o.mcf_o,
                                                                                empty_lines_end
                                                                                )
    if rep_o.opt_o is not None:
        if isinstance(rep_o.opt_o, OptimalPolicyVersions):
            rep_o.text['opt_versions'] = content_optp.opt_versions(rep_o.opt_o, empty_lines_end)
            if isinstance(rep_o.opt_o.optp, list):
                opt_o = rep_o.opt_o.optp[0]
            else:
                opt_o = rep_o.opt_o.optp
        elif isinstance(rep_o.opt_o, OptimalPolicy):
            opt_o = rep_o.opt_o
            rep_o.text['opt_versions'] = None
        else:
            raise ValueError('optp instances used in reporting is an object an from unknown class.')

        # If version estimation, use only the main treatments for reporting here
        rep_o.text['opt_general'] = content_optp.opt_general(opt_o, empty_lines_end)
        rep_o.text.update(optpol_section_text(opt_o, empty_lines_end))
    if rep_o.sens_o is not None:
        rep_o.text['sensitivity'] = content.sensitivity(rep_o.sens_o, empty_lines_end)


def header_title(rep_o: 'McfOptPolReport') -> str:
    """Make header information."""
    txt = 'Modified Causal Forest: '
    methods = []
    if rep_o.mcf:
        methods.append('Estimation')
    if rep_o.sens:
        methods.append('Sensitivity Analysis')
    if rep_o.opt:
        methods.append('Optimal Policy Estimation')

    return txt + ', '.join(methods)


@dataclass(slots=True, kw_only=True)
class ReportCfg:
    """Define key config infos for the pdf report."""

    outfilename: Path
    outpath: Path

    @classmethod
    def from_args(
        cls,
        outputfile: Path | str | None = None,
        outputpath: Path | str | None = None,
            ) -> 'ReportCfg':
        """Get the inputs and transform as needed."""
        # timestamp
        stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # base name (without extension)
        base = 'Report' if outputfile is None else Path(outputfile).stem
        filename = f'{base}_{stamp}.pdf'

        # output directory
        outpath_p = Path.cwd() / 'output' if outputpath is None else Path(outputpath)
        outpath_p = mcf_sys.define_outpath(outpath_p, new_outpath=False)

        return cls(outfilename=outpath_p / filename, outpath=outpath_p)
