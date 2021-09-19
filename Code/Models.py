# -*- coding: utf-8 -*-
"""
Obtains historical returns for *any* strategy.

Given: asset returns universe, classification variable, and possible
portfolio weights, in addition to how (and how many) stocks are selected.

Thiago de Oliveira Souza, September 2021
"""

import numpy as np
import pandas as pd


class Models:
    # Models.basePrices = ...
    # Sometimes, it is better to have the general ones as class attribute.
    # Depends on how you use the models.
    '''This is a class for model objects that select stocks based on any given
    classification variable and portfolio weights (both are lagged internally!)
    The input matrices must have the same (ordered) index and the same columns
    (in the same order).
    Functions:
        - save the strategy returns as a dataframe in the .performance 
        attribute.
        - Computes the historical performance of this model in terms
    of cumulative returns, and compares to an equal weighted portfolio in the
    .compare attribute.'''

    # key = defines the instance/model (ex: use as key in a model-dict)
    def __init__(self, key, baseReturns=None, classifVariable=None,
                 baseWeights=None, assetsN=None, name='model', basePrices=None,
                 targetLow=False, benchmark='ew'):

        self.key = key

        # Values needed (prices or returns):
        self.basePrices = basePrices
        self.baseReturns = baseReturns
        self.classifVariable = classifVariable
        self.baseWeights = baseWeights

        # Defaults given:
        self.name = name
        self.targetLow = targetLow
        self.benchmark = benchmark
        self.assetsN = assetsN

        # Output (generated internally):
        self.performance = None
        self.weights = None
        self.ranking = None
        self.selectedAssets = None

    def inconsistentMatrices(self):
        '''Checks if the 3 matrices (returns, weights and classification) have
        the same *exact* fields and indices. If not, returns True.'''

        if (all(self.baseReturns.index == self.baseWeights.index) and
                all(self.baseReturns.index == self.classifVariable.index)):
            if (all(self.baseReturns.columns == self.baseWeights.columns) and
                    all(self.baseReturns.columns
                        == self.classifVariable.columns)):
                return False
            else:
                return True
        else:
            return True

    def calculatePerformance(self, returnsFromPrices=True):
        '''Calculate the performance of the model (one period returns).'''

        if (returnsFromPrices and self.baseReturns is None and
                self.basePrices is not None):

            print('Warning: '
                  'baseReturns automatically calculated from basePrices.')
            self.baseReturns = self.basePrices.pct_change()

        if self.inconsistentMatrices():
            print('Error: I am not calculating anything. '
                  'Indices and columns must be *exactly* the same (in order) '
                  'for baseReturns(prices), classifVariables and baseWeights.')
            return

        # Create ranking based on classification variables
        # *at the end of t* (not lagged!)
        self.ranking = self.classifVariable.rank(axis=1,
                                                 ascending=self.targetLow)

        # mask for assets selected *at the end of t* (not lagged!)
        self.selectedAssets = self.ranking <= self.assetsN

        # Calculate (normalized) portfolio weights *at the end of t*
        self.weights = (self.baseWeights[self.selectedAssets]
                            .div(self.baseWeights[self.selectedAssets]
                                 .sum(axis=1), axis='index'))

        # NOW THERE'S A LAG:
        # (realized) strategy return = w(t-1) * Ret(t)
        self.performance = pd.DataFrame((self.weights.shift(1)
                                         * self.baseReturns)
                                        .sum(axis=1, min_count=1),
                                        columns=[self.name])
        return

    def compareCumulative(self, compareStarts=False,
                          rLabel=False, showGraph=True):
        '''Calculates the cumulative returns of the strategy in this model
        (under the column rLabel, if provided. Otherwise column=model.name).
        So far the only benchmark is the equal weighted portfolio.
        Compares and display a graph if showGraph.'''

        if not compareStarts:
            compareStarts = self.baseReturns.index.min()

        if not rLabel:
            rLabel = self.name

        # Only EW benchmark implemented:

        compare = self.performance.copy()

        if self.benchmark == 'ew':
            compare['EW'] = self.baseReturns.mean(axis=1)

        # Now the cumulative performance (after compareStarts):
        compare = compare.loc[compareStarts:]

        for x in [rLabel, 'EW']:
            compare['Cum. ' + x] = (compare[x].expanding()
                                    .apply(lambda x:
                                           np.expm1(np.log1p(x).sum())))
        self.compare = compare

        if showGraph:
            compare[['Cum. ' + rLabel, 'Cum. EW']].plot(
                title='Cumulative performances')

        return
