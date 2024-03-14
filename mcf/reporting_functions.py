"""Created on Fri Dec  8 13:42:19 2023.

Contains classes and methods needed for mcf and optimal policy reporting.

# -*- coding: utf-8 -*-

@author: MLechner
"""
from datetime import datetime
import os

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from mcf import mcf_general_sys as mcf_sys
from mcf import reporting_content_functions as content


def create_pdf_file(rep_o):
    """Create a pdf report from the text."""
    pdf = PDF()
    pdf.set_title(rep_o.text['header'])
    pdf.set_author("The mcf estimation package")
    # grey: General info; green: mcf; red: sensitivity; blue: optimal policy
    pdf.print_section((1,), "General information",
                      rep_o.text['general'], color='grey')
    idx1 = 2
    if rep_o.mcf_o is not None:
        report_mcf_core(pdf, rep_o, idx1)
        idx1 += 1
    if rep_o.blind_o is not None:
        report_mcf_blind(pdf, rep_o, idx1)
        idx1 += 1
    if rep_o.sens_o is not None:
        report_mcf_sense(pdf, rep_o, idx1)
        idx1 += 1
    if rep_o.opt_o is not None:
        report_optpol(pdf, rep_o, idx1)

    pdf.output(rep_o.gen_dict['outfilename'])


def report_optpol(pdf, rep_o, idx1):
    """Write report on optimal policy allocations."""
    report_allocation = False          # Kept for potential later use
    idx2 = 1
    pdf.print_section((idx1,), "Optimal Policy",
                      rep_o.text['opt_general'], 'blue')
    if rep_o.opt_o.report['training']:
        pdf.print_section((idx1, idx2), "Optimal Policy: Training",
                          rep_o.text['opt_training'], 'blue')
        idx2 += 1
    if rep_o.opt_o.report['allocation'] and report_allocation:
        pdf.print_section((idx1, idx2), "Optimal Policy: Allocation",
                          rep_o.text['opt_allocation'], 'blue')
        idx2 += 1
    if rep_o.opt_o.report['evaluation']:
        general_text, txt_table_list = rep_o.text['opt_evaluation']
        pdf.print_section((idx1, idx2),
                          "Optimal Policy: Evaluation of Allocation(s)",
                          general_text, 'blue')
        for txt_table in txt_table_list:
            pdf.add_table('Evaluation of treatment allocation', txt_table[1],
                          col_width=30, font_size_table=10, note=txt_table[0],
                          index_label='Allocation')


def report_mcf_core(pdf, rep_o, idx1):
    """Write report on mcf training, prediction, and analysis."""
    idx2 = idx3 = idx4 = 1
    pdf.print_section((idx1,), 'MCF estimation', rep_o.text['mcf_general'],
                      'green')
    pdf.print_section((idx1, idx2), 'MCF Training',
                      rep_o.text['mcf_training'], 'greenT')
    pdf.print_section((idx1, idx2, idx3),
                      'Preparation of training data (mcf training)',
                      rep_o.text['mcf_descriptives'], 'greenT')
    idx3 += 1
    if rep_o.mcf_o.fs_dict["yes"]:
        pdf.print_section((idx1, idx2, idx3),
                          "Feature selection (mcf training)",
                          rep_o.text['mcf_feature_selection'], 'greenT')
        idx3 += 1
    if rep_o.mcf_o.cs_dict["type"] > 0:
        pdf.print_section((idx1, idx2, idx3),
                          "Common support (mcf training)",
                          rep_o.text['mcf_t_common_support'], 'greenT')
        idx3 += 1
        title = "Common support plots"
        pdf.add_figure_row(title, rep_o.mcf_o.report['cs_t_figs'][0][1:],
                           width=70, height=60)
    if rep_o.mcf_o.lc_dict["yes"]:
        pdf.print_section((idx1, idx2, idx3),
                          "Local centering (mcf training)",
                          rep_o.text['mcf_local_center'], 'greenT')
        idx3 += 1
    pdf.print_section((idx1, idx2, idx3), "Forest",
                      rep_o.text['mcf_forest'], 'greenT')
    idx2 += 1
    idx3 = 1
    pdf.print_section((idx1,), "MCF estimation", ' ', 'green')
    pdf.print_section((idx1, idx2), "MCF Prediction of Effects",
                      rep_o.text['mcf_prediction'], 'greenP')
    if rep_o.mcf_o.cs_dict["type"] > 0:
        pdf.print_section((idx1, idx2, idx3),
                          "Common support (mcf prediction)",
                          rep_o.text['mcf_p_common_support'], 'greenP')
        idx3 += 1
    idx4 = 1
    pdf.print_section((idx1, idx2, idx3), "Results",
                      rep_o.text['mcf_results'], 'greenP')
    pdf.print_section((idx1, idx2, idx3, idx4), "ATE",
                      rep_o.text['mcf_ate'][0], 'greenP')
    for idx, table_df in enumerate(rep_o.text['mcf_ate'][1]):
        y_name = rep_o.mcf_o.var_dict['y_name'][idx]
        if len(y_name) > 3 and y_name[-3:] == '_LC':
            y_name = y_name[:-3]
        table_df.columns = [name.replace('p_val', 'p-value')
                            for name in table_df.columns]
        table_df.columns = [name.replace('t-val', 't-value')
                            for name in table_df.columns]
        pdf.add_table('ATE for ' + y_name, table_df,
                      col_width=30, note=rep_o.text['mcf_ate'][2])
    idx4 += 1
    if rep_o.mcf_o.p_dict['gate']:
        title = 'GATE'
        pdf.print_section((idx1, idx2, idx3, idx4), "GATE", ' ', 'greenP')
        pdf.add_figure_row(title, rep_o.text['mcf_gate'][0],
                           width=70, height=60, note=rep_o.text['mcf_gate'][1])
        idx4 += 1
    if rep_o.mcf_o.p_dict['bgate']:
        title = 'BGATE'
        pdf.print_section((idx1, idx2, idx3, idx4), "BGATE",
                          rep_o.text['mcf_bgate'][2], 'greenP')
        pdf.add_figure_row(title, rep_o.text['mcf_bgate'][0],
                           width=70, height=60, note=rep_o.text['mcf_bgate'][1])
        idx4 += 1
    if rep_o.mcf_o.p_dict['cbgate']:
        title = 'CBGATE'
        pdf.print_section((idx1, idx2, idx3, idx4), "CBGATE",
                          'All covariates are balanced.', 'greenP')
        pdf.add_figure_row(title, rep_o.text['mcf_cbgate'][0],
                           width=70, height=60, note=rep_o.text['mcf_cbgate'][1]
                           )
        idx4 += 1
    if rep_o.mcf_o.p_dict['iate']:
        pdf.print_section((idx1, idx2, idx3, idx4), "IATE",
                          rep_o.text['iate_part1'], color='greenP')
        idx4 += 1
    if rep_o.mcf_o.p_dict['bt_yes']:
        pdf.print_section((idx1, idx2, idx3, idx4),
                          "Balancing checks (experimental)",
                          rep_o.text['mcf_balance'], 'greenP')
    idx2 += 1
    idx3 = idx4 = 1
    if (rep_o.mcf_o.int_dict['with_output']
        and rep_o.mcf_o.post_dict['est_stats']
        and rep_o.mcf_o.int_dict['return_iate_sp']
            and rep_o.mcf_o.report.get('fig_iate') is not None):
        pdf.print_section((idx1,), "MCF estimation", ' ', 'green')
        pdf.print_section((idx1, idx2), "Analysis of Estimated IATEs",
                          rep_o.text['mcf_iate_part2'][0], 'greenA')
        if rep_o.mcf_o.report['fig_iate']:
            pdf.add_figure_row('IATEs', rep_o.text['mcf_iate_part2'][1],
                               width=70, height=60,
                               note=rep_o.text['mcf_iate_part2'][2])
        knn_table = rep_o.text['mcf_iate_part2'][4]
        if knn_table is not None:
            pdf.print_section(txt=rep_o.text['mcf_iate_part2'][3])
            pdf.print_section(txt=knn_table['obs_cluster_str'])
            note = 'Mean of variable in cluster.'
            pdf.add_table('IATE ', knn_table['IATE_df'],
                          col_width=30, font_size_table=10, note=note)
            pdf.add_table('Potential Outcomes ', knn_table['PotOutcomes_df'],
                          col_width=30, font_size_table=10, note=note)
            pdf.add_table('Features ', knn_table['Features_df'],
                          col_width=30, font_size_table=8,
                          note=note + ' Categorical variables are recoded ' +
                          'to indicator (dummy) variables.')


def report_mcf_blind(pdf, rep_o, idx1):
    """Write report on blinded IATEs."""
    pdf.print_section((idx1,), "MCF Blinder IATEs",
                      'Blinding IATEs is experimental. Some results '
                      'can be found in the output files. ', 'yellow')


def report_mcf_sense(pdf, rep_o, idx1):
    """Write report on sensitivity analysis."""
    pdf.print_section((idx1,), "MCF Sensitivity Analysis",
                      rep_o.text['sensitivity'], 'red')
    if rep_o.sens_o.report['sens_plots_iate'] is not None:
        pdf.add_figure_row('Placebo and estimated IATEs',
                           rep_o.sens_o.report['sens_plots_iate'], width=70,
                           height=60, note='\n')


def create_text(rep_o):
    """Create the dictionary containing all information to be printed."""
    empty_lines_end = 2
    rep_o.text['header'] = header_title(rep_o)
    rep_o.text['general'] = content.general(rep_o)
    if rep_o.mcf_o is not None:
        rep_o.text['mcf_general'] = content.mcf_general(rep_o.mcf_o,
                                                        empty_lines_end)
        rep_o.text['mcf_training'] = content.mcf_training(rep_o.mcf_o,
                                                          empty_lines_end)
        rep_o.text['mcf_descriptives'] = content.mcf_descriptives(
            rep_o.mcf_o, empty_lines_end)
        if rep_o.mcf_o.fs_dict["yes"]:
            rep_o.text['mcf_feature_selection'] = content.mcf_feature_selection(
                rep_o.mcf_o, empty_lines_end)
        if rep_o.mcf_o.cs_dict["type"] > 0:
            rep_o.text['mcf_t_common_support'] = content.mcf_common_support(
                rep_o.mcf_o, empty_lines_end - 1, train=True)
        if rep_o.mcf_o.lc_dict["yes"]:
            rep_o.text['mcf_local_center'] = content.mcf_local_center(
                rep_o.mcf_o, empty_lines_end)
        rep_o.text['mcf_forest'] = content.mcf_forest(
            rep_o.mcf_o, empty_lines_end)

        # When predict and analyse is used more than once, only first occurance
        #  is reported
        if rep_o.mcf_o.report['predict_list']:
            sub_dict = rep_o.mcf_o.report['predict_list'][0].copy()
            rep_o.mcf_o.report.update(sub_dict)
            rep_o.mcf_o.report['predict_list'] = []
        if rep_o.mcf_o.report['analyse_list']:
            sub_dict = rep_o.mcf_o.report['analyse_list'][0].copy()
            rep_o.mcf_o.report.update(sub_dict)
            rep_o.mcf_o.report['analyse_list'] = []

        rep_o.text['mcf_prediction'] = content.mcf_prediction(rep_o.mcf_o,
                                                              empty_lines_end)
        if rep_o.mcf_o.cs_dict["type"] > 0:
            rep_o.text['mcf_p_common_support'] = content.mcf_common_support(
                rep_o.mcf_o, empty_lines_end - 1, train=False)
        rep_o.text['mcf_results'] = content.mcf_results(rep_o.mcf_o,
                                                        empty_lines_end)
        rep_o.text['mcf_ate'] = content.mcf_ate(rep_o.mcf_o, empty_lines_end)
        if rep_o.mcf_o.p_dict['gate']:
            rep_o.text['mcf_gate'] = content.mcf_gate(
                rep_o.mcf_o, empty_lines_end, gate_type='gate')
        if rep_o.mcf_o.p_dict['bgate']:
            rep_o.text['mcf_bgate'] = content.mcf_gate(
                rep_o.mcf_o, empty_lines_end, gate_type='bgate')
        if rep_o.mcf_o.p_dict['cbgate']:
            rep_o.text['mcf_cbgate'] = content.mcf_gate(
                rep_o.mcf_o, empty_lines_end, gate_type='cbgate')
        if rep_o.mcf_o.p_dict['bt_yes']:
            rep_o.text['mcf_balance'] = content.mcf_balance(empty_lines_end)
        if rep_o.mcf_o.p_dict['iate']:
            rep_o.text['iate_part1'] = content.mcf_iate_part1(rep_o.mcf_o,
                                                              empty_lines_end)
        if (rep_o.mcf_o.int_dict['with_output']
            and rep_o.mcf_o.post_dict['est_stats']
                and rep_o.mcf_o.int_dict['return_iate_sp']):
            rep_o.text['mcf_iate_part2'] = content.mcf_iate_analyse(
                rep_o.mcf_o, 1)

    if rep_o.opt_o is not None:
        rep_o.text['opt_general'] = content.opt_general(
            rep_o.opt_o, empty_lines_end)
        if rep_o.opt_o.report['training']:
            rep_o.text['opt_training'] = content.opt_training(
                rep_o.opt_o, empty_lines_end)
        if rep_o.opt_o.report['allocation']:
            rep_o.text['opt_allocation'] = content.opt_allocation(
                rep_o.opt_o, empty_lines_end)
        if rep_o.opt_o.report['evaluation']:
            rep_o.text['opt_evaluation'] = content.opt_evaluation(rep_o.opt_o)

    if rep_o.sens_o is not None:
        rep_o.text['sensitivity'] = content.sensitivity(rep_o.sens_o,
                                                        empty_lines_end)


def header_title(rep_o):
    """Make header information."""
    txt = 'Modified Causal Forest: '
    methods = []
    if rep_o.mcf:
        methods.append('Estimation')
    if rep_o.sens:
        methods.append('Sensitivity Analysis')
    if rep_o.blind:
        methods.append('Blinded IATEs')
    if rep_o.opt:
        methods.append('Optimal Policy Estimation')
    return txt + ', '.join(methods)


class PDF(FPDF):
    """Inherited class from fpdf."""

    def header(self):
        """Define header."""
        # Setting font: helvetica bold 12
        self.set_font("helvetica", "B", 16)
        # Calculating width of title and setting cursor position:
        width = self.get_string_width(self.title) + 6
        self.set_x((210 - width) / 2)
        # Setting colors for frame, background and text: red, green, blue
        self.set_draw_color(0, 128, 0)
        self.set_fill_color(144+30, 238+17, 144+30)
        self.set_text_color(0)
        # Setting thickness of the frame (1 mm)
        self.set_line_width(1)
        # Printing title:
        self.cell(
            width,
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

    def section_title(self, num_tuple, label, color=None):
        """Define chapter title."""
        # Setting font: helvetica 12
        darker = 20
        if len(num_tuple) == 1:
            self.set_font("helvetica", "", 16)
        elif len(num_tuple) == 2:
            self.set_font("helvetica", "", 14)
        elif len(num_tuple) == 3:
            self.set_font("helvetica", "", 12)
        else:
            self.set_font("helvetica", "", 11)
        if len(num_tuple) == 1:
            num = str(num_tuple[0])
        else:
            num = '.'.join([str(s) for s in num_tuple])
        # Setting background color: red, green, blue
        if color == 'blue':
            self.set_fill_color(200, 220, 255)
        elif color == 'green':
            self.set_fill_color(144, 238, 144)
        elif color == 'greenT':
            self.set_fill_color(144-darker, 238-darker, 144-darker)
        elif color == 'greenP':
            self.set_fill_color(144-darker*2, 238-darker*2, 144-darker*2)
        elif color == 'greenA':
            self.set_fill_color(144-darker*3, 238-darker*3, 144-darker*3)
        elif color == 'red':
            self.set_fill_color(255, 127, 127)
        elif color == 'yellow':
            self.set_fill_color(255, 255, 0)
        else:
            self.set_fill_color(200)     # grey
        # Printing section name:
        self.cell(
            0,
            6,
            f"Section {num}: {label}",
            new_x="LMARGIN",
            new_y="NEXT",
            align="L",
            fill=True,
        )
        # Performing a line break:
        self.ln(4)

    def section_body(self, txt):
        """Define chapter body."""
        # Setting font: Helvetica 12
        self.set_font("helvetica", size=12)
        # Printing justified text:
        self.multi_cell(0, h=5, text=txt, align='L')
        # Performing a line break:
        self.ln()

    def print_section(self, num=None, title=None, txt=None, color=None):
        """Print pdf to file."""
        if num is not None and len(num) == 1:
            self.add_page()
        if title is not None:
            self.section_title(num, title, color)
        if txt is not None:
            self.section_body(txt)

    def add_image(self, title, image_path, width=90, height=90):
        """Add figure."""
        self.set_font("helvetica", 'I', size=12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        y_before_image = self.get_y()
        self.image(image_path, x=10, y=y_before_image, w=width, h=height)
        self.ln()
        # Set the current position below the figure
        self.set_y(y_before_image + self.h)

    def add_figure_row(self, title, image_paths, width=80, height=40,
                       note=None):
        """Print figures in pdf."""
        self.set_font("helvetica", 'I', size=12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

        # Calculate the available width for each figure in a row
        available_width = self.w - 20  # 10 units margin on each side
        figure_width = available_width / 2
        # Loop through image paths and add figures in rows
        for i in range(0, len(image_paths), 2):
            # Start a new row
            self.ln()
            # Check page is full
            if self.get_y() + height > self.h:
                self.add_page()

            # Add the first figure in the row
            self.image(image_paths[i], x=10, y=self.get_y(), w=width, h=height)

            # Check if there's a second figure in the row
            if i + 1 < len(image_paths):
                # Add the second figure in the row
                self.image(image_paths[i + 1], x=figure_width + 20,
                           y=self.get_y(), w=width, h=height)

            if self.get_y() + height > self.h:
                self.add_page()
            else:
                self.set_y(self.get_y() + height)

        if note is not None:
            self.set_font("helvetica", size=8)
            self.multi_cell(0, h=5, text=note, align='L')
        else:
            self.ln()

    def add_table(self, title, data, col_width=30, note=None,
                  font_size_table=10, index_label="Comparison"):
        """Add table."""
        col_height = 7
        self.set_font("helvetica", 'I', size=12)
        self.cell(0, col_height, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                  align='C')

        # Calculate available page width
        available_width = self.w - self.l_margin - self.r_margin

        # Check if labels in first (index) column are too long
        max_width = max(len(str(row.name)) for _, row in data.iterrows())
        col_width_idx = max(max_width*1.6, col_width)
        col_width_idx = min(46, col_width_idx)

        # Ensure total column width does not exceed page width
        total_col_width = col_width_idx + len(data.columns) * col_width
        if total_col_width > available_width:
            # Adjust column widths proportionally
            col_width_idx *= available_width / total_col_width
            col_width *= available_width / total_col_width
            font_size_table -= 1

        self.set_font("helvetica", 'I', size=font_size_table)

        # Add index header
        self.cell(col_width_idx, col_height, index_label, border=1, align='L')

        # Add headers
        for col in data.columns:
            self.cell(col_width, col_height, str(col), border=1, align='C')
        self.ln()
        self.set_font("helvetica", size=font_size_table)
        if font_size_table < 10:
            col_height = 5
        # Add data
        for _, row in data.iterrows():
            # Add index cell
            self.cell(col_width_idx, col_height, str(row.name), border=1,
                      align='L')
            # Add data cells
            for col in data.columns:
                self.cell(col_width, col_height, str(row[col]), border=1,
                          align='C')
            self.ln()
        if note is not None:
            self.set_font("helvetica", size=8)
            self.multi_cell(0, h=5, text='Note: ' + note, align='L')
        self.ln()


def gen_init(outputfile, outputpath):
    """Put control variables for reporting into dictionary."""
    dic = {}
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime('%Y_%m_%d_%H_%M')
    outputfile = 'Report' if outputfile is None else outputfile
    outputpath = os.getcwd() + '/out' if outputpath is None else outputpath
    outputfile += '_' + formatted_datetime + '.pdf'
    outputpath = mcf_sys.define_outpath(outputpath, new_outpath=False)
    dic['outfilename'] = outputpath + '/' + outputfile
    return dic
