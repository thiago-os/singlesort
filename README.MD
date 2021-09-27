The Models class (in Models.py) calculates historical returns for any (one-way sorted) strategy given 3 matrices:
1. Asset returns universe
2. Classification variable
3. Portfolio weights (do not need to add up to one)

Index must be sorted time-series and all matrices must use end of period convention 
(i.e., weights and classification variable are lagged internally).  
The matrices must have identical fields and index.

In addition, inform if we buy high/low values (of BM, for example) and how many target stocks (assetsN). 
Percentile cutoff points are also implemented (between lowQuant and highQuant).

The Jupyter notebook (test.ipynb) has an example and compares the results with the one of a small Excel 
spreadsheet based on fake data (testData.xlsx).