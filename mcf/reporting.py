from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

from mcf import reporting_functions as rep

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.optpolicy_main import OptimalPolicy, OptimalPolicyVersions


class McfOptPolReport:
    """
    Summary of main specifications and results for MCF estimation and optimal policy learning.

    Parameters
    ----------
        mcf : Instance of the ModifiedCausalForest class or None, optional
            Instance supplying estimation results and descriptive analyses for the report.
            If several runs are stored, use the first effect prediction and the first analyse() run.
            For predict_different_allocations(), use the latest stored run.
            Default is None.

        mcf_sense : Instance of the ModifiedCausalForest class or None, optional
            Contains all information from sensitivity analysis needed for
            reports. The default is None.

        optpol : Instance of OptimalPolicy or OptimalPolicyVersions, or None, optional
            Instance supplying optimal-policy specifications and results for the PDF.
            For OptimalPolicy, include the evaluations stored by evaluate().
            For OptimalPolicyVersions, summarize the fitted allocation rules. Final evaluations
            returned by OptimalPolicyVersions.evaluate() are not included in the PDF; use the
            returned results dictionary and, when enabled, the separate evaluation text output.
            The default is None.

        outputpath : String, pathlib.Path, or None, optional
            Directory in which :meth:`~McfOptPolReport.report` saves the PDF.
            If None, the 'output' subdirectory of the current working directory is used.
            The directory is created if it does not exist. The default is None.

        outputfile : String, pathlib.Path, or None, optional
            Base filename of the PDF created by :meth:`~McfOptPolReport.report`.
            If None, 'Report' is used. For a supplied filename or path, only the filename
            stem is used; any directory component and the final extension are discarded.
            The final filename is '<name>_YYYY_MM_DD_HH_MM_SS.pdf'.
            The timestamp is taken when this McfOptPolReport instance is created.
            The default is None.

    Attributes
    ----------
    version : String
        Version of mcf module used to create the instance.

    <NOT-ON-API>

    gen_cfg : ReportCfg dataclass
        Parameters used to create and save reports.

    mcf_o : ModifiedCausalForest or None
        Instance supplying MCF estimation results.

    opt_o : OptimalPolicy, OptimalPolicyVersions, or None
        Instance supplying optimal-policy results.

    sens_o : ModifiedCausalForest or None
        Instance supplying sensitivity-analysis results.

    text : Dictionary
        Container for generated report text.

    mcf : Boolean
        True when an mcf instance was supplied, regardless of which results it contains.

    opt : Boolean
        True when an optpol instance was supplied, regardless of which results it contains.

    sens : Boolean
        True when an mcf_sense instance was supplied, regardless of which results it contains.

    iv : Boolean
        Initially False. When report() processes an MCF instance, set to whether that instance
        contains a non-None first-stage instrumental-variable model.

    </NOT-ON-API>

    """


    def __init__(self: 'McfOptPolReport', *,
                 mcf: 'ModifiedCausalForest | None' = None,
                 mcf_sense: 'ModifiedCausalForest | None' = None,
                 optpol: 'OptimalPolicy | OptimalPolicyVersions | None' = None,
                 outputpath: Path | str | None = None,
                 outputfile: Path | str | None = None,
                 ) -> None:
        self.gen_cfg = rep.ReportCfg.from_args(outputfile, outputpath)
        self.mcf_o = mcf
        self.opt_o = optpol
        self.sens_o = mcf_sense
        self.mcf = self.mcf_o is not None
        self.opt = self.opt_o is not None
        self.sens = self.sens_o is not None
        self.text = {}
        self.iv = False        # Instrumental variable estimation

        self.version = '0.11.0'

    def report(self):
        """Create a PDF report save file to a user provided location.

        Using instances of the
        :class:`~mcf_main.ModifiedCausalForest` and
        :class:`~optpolicy_main.OptimalPolicy` or
        :class:`~optpolicy_main.OptimalPolicyVersions` classes.

        Returns
        -------
        outpath : Pathlib object
            Name and Location of file in which pdf output is saved.

        """
        mcf_o = self.mcf_o
        try:
            if self.mcf:
                self.mcf_o = copy(mcf_o)
                self.mcf_o.report = mcf_o.report.copy()
                self.iv = self.mcf_o.iv_mcf['firststage'] is not None
            # Step one: Fill the dictionaries
            rep.create_text(self)

            # Step two: Connect the text and figures save as pdf
            rep.create_pdf_file(self)
            print(f'\nReport printed: {self.gen_cfg.outfilename}\n')

            return self.gen_cfg.outfilename
        finally:
            self.mcf_o = mcf_o
