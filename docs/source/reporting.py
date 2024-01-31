class McfOptPolReport:
    """Methods and attributes for reporting mcf and optimal policy results.

    Parameters
    ----------
        mcf : Instance of the ModifiedCausalForest class or None, optional
            Contains all information needed for reports. The default is None. 
            
        mcf_blind : Instance of the ModifiedCausalForest class or None, optional
            Contains all information from blinded IATE analysis needed for
            reports. The default is None.      
            
        mcf_sense : Instance of the ModifiedCausalForest class or None, optional
            Contains all information from sensitivity analysis needed for
            reports. The default is None.
            
        optpol : Instance of the OptimalPolicy class or None, optional
            Contains all information from the optimal policy analysis needed
            for reports. The default is None.
            
        outputpath : String or None, optional
            Path to write the pdf file that is created with the reporting
            method. If None, then an '/out' subdirectory of the current working
            directory is used. If the latter does not exist, it is created.
            
        outputfile : String or None, optional
            Name of the pdf file that is created by the reporting method.
            If None, 'Reporting' is used as name. Any name will always appended
            by string that contains the day and time (measured when the
            programme ends).
    """

    def __init__(self, mcf=None, mcf_blind=None, mcf_sense=None, optpol=None,
                 outputpath=None, outputfile=None):
        self.gen_dict = gen_init(outputfile, outputpath)
        self.mcf_o, self.opt_o, self.sens_o = mcf, optpol, mcf_sense
        self.blind_o = mcf_blind
        self.mcf = self.mcf_o is not None
        self.opt = self.opt_o is not None
        self.sens = self.sens_o is not None
        self.blind = self.blind_o is not None
        self.text = {}

    @property
    def xxxx(self):
        """
        Dictionary, parameters to compute (partially) blinded IATEs.
        """
        return self._xxxx
    
    def report(self):
        """Create the content of the report.

        Returns
        -------
        None.

        """
        # Step one: Fill the dictionaries
        rep.create_text(self)

        # Step two: Connect the text and figures save as pdf
        rep.create_pdf_file(self)
        print(f'\nReport printed: {self.gen_dict["outfilename"]}\n')


def gen_init(outputfile, outputpath):
    """Put control variables for reporting into dictionary."""
    dic = {}
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime('%Y_%m_%d_%H_%M')
    outputfile = 'Report' if outputfile is None else outputfile
    outputpath = os.getcwd() + '/out' if outputpath is None else outputpath
    outputfile += '_' + formatted_datetime + '.pdf'
    outputpath = mcf_sys.define_outpath(outputpath, new_outpath=False)
    dic['outfilename'] = outputpath + '/' + outputfile

    return dic
