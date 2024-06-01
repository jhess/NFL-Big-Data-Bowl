class Correlator():
    """Represents a correlation matrix / heatmap for a dataframe.
    
    Examples:
        c = Correlator(df)
        c.corr # shows correlation matrix
        c.heatmap # shows correlation matrix as a heatmap
    """

    def correlation_heatmap(self, dataframe, precision=2, cmap='coolwarm'):
        """Generates a correlation heatmap.

        Args:
            dataframe: the Dataframe object.
            precision: correlation precision to display.
            cmap: cmap which represents the color scheme of the heatmap

        Returns:
            The correlation object (print to display).
        """
        corr = df.corr()
        # Limit number of digits:
        return corr, corr.style.background_gradient(cmap=cmap).set_precision(precision)
    
    def __init__(self, dataframe):
        self.corr = df.corr()
        self.heatmap = correlation_heatmap(dataframe)
        self.df = dataframe
