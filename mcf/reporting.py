from mcf import reporting_functions as rep


class McfOptPolReport:
    """
    .. versionadded:: 0.9.0
        Provides reports about the main specification choices and most
        important results of the :class:`~mcf_functions.ModifiedCausalForest`
        and :class:`~optpolicy_functions.OptimalPolicy` estimations.

    Parameters
    ----------
        mcf : Instance of the ModifiedCausalForest class or None, optional
            Contains all information needed for reports. The default is None.

        mcf_sense : Instance of the ModifiedCausalForest class or None, optional
            Contains all information from sensitivity analysis needed for
            reports. The default is None.

        optpol : Instance of the OptimalPolicy class or None, optional
            Contains all information from the optimal policy analysis needed
            for reports. The default is None.

        outputpath : String, Pathlib object, or None, optional
            Path to write the pdf file that is created with the
            :meth:`~McfOptPolReport.report` method. If None, then an
            '/out' subdirectory of the current working directory is used.
            If the latter does not exist, it is created.

        outputfile : String or None, optional
            Name of the pdf file that is created by the
            :meth:`~McfOptPolReport.report` method. If None, 'Reporting' is
            used as name. Any name will always appended by string that contains
            the day and time (measured when the programme ends).

    Attributes
    ----------

    version : String
        Version of mcf module used to create the instance.

    <NOT-ON-API>

    gen_cfg : GenCfg dataclass
        General parameters to compute reports.

    mcf_o : Dictionary
        Content from mcf estimation to compute reports.

    opt_o : Dictionary
        Content from optimal policy allocation to compute reports.

    sens_o : Dictionary
        Content from sensitivity analysis to compute reports.

    blind_o : Dictionary
        Content from partially blinded IATE estimation to compute reports.

    text : Dictionary
        Container for text to compute reports.

    mcf : Boolean
        True if there is anything to report about mcf estimation.

    opt : Boolean
        True if there is anything to report about optimal policy allocation.

    sens : Boolean
        True if there is anything to report about sensitivity analysis.

    blind : Boolean.
        True if there is anything to report about blinded IATE estimation.

    iv : Boolean.
        True if instrumental variable estimation is performed.

    </NOT-ON-API>

    """

    def __init__(self, mcf=None, mcf_sense=None, optpol=None,
                 outputpath=None, outputfile=None):
        self.gen_dict = rep.gen_init(outputfile, outputpath)
        self.mcf_o, self.opt_o, self.sens_o = mcf, optpol, mcf_sense
        self.blind_o = None  # Underlying method deprecated. No reporting.
        self.mcf = self.mcf_o is not None
        self.opt = self.opt_o is not None
        self.sens = self.sens_o is not None
        self.blind = self.blind_o is not None
        self.text = {}
        self.iv = False        # Instrumental variable estimation

        self.version = '0.8.0'

    def report(self):
        """Create a PDF report save file to a user provided location.

        Using instances of the
        :class:`~mcf_functions.ModifiedCausalForest` and
        :class:`~optpolicy_functions.OptimalPolicy` classes.

        Returns
        -------
        outpath : Pathlib object
            Name and Location of file in which pdf output is saved.

        """
        if self.mcf:
            self.iv = self.mcf_o.iv_mcf['firststage'] is not None
        # Step one: Fill the dictionaries
        rep.create_text(self)

        # Step two: Connect the text and figures save as pdf
        rep.create_pdf_file(self)
        print(f'\nReport printed: {self.gen_dict["outfilename"]}\n')
        return self.gen_dict['outfilename']
