from freqtrade.strategy import IStrategy

class SampleStrategy(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.01}
    stoploss = -0.05

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
