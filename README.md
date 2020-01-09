# Nonlocal low-rank SAR stack despeckling

Code for the paper [Robust nonlocal low-rank SAR time series despeckling considering speckle correlation by total variation regularization](https://ieeexplore.ieee.org/document/9079477).
![alt text](../imgs/methods_comparison.png?raw=true)

## Installation
If necessary create a new Python environment.
Install requirements.
```
pip install -r requirements.txt
```
Install package.
```
pip install .
```
Sphinx documentation can be built using
```
python setup.py build_sphinx
```


## Tests 
Execute ```pytest``` in the root directory to run all unit tests.


## Examples
Examples show the impact of TV regularization along the x or y axes, the benefit of the weighted nuclear norm normalization, and how to use the code for despeckling a stack.


## Citation
```
@article{baier2020nllrtv,
   author  = {G. {Baier} and W. {He} and N. {Yokoya}},
   title   = {Robust nonlocal low-rank {SAR} time series despeckling considering speckle correlation by total variation regularization},
   journal = {IEEE Transactions on Geoscience and Remote Sensing}, 
   year    = {2020},
   month   = {Early access},
   volume  = {Early access},
   number  = {Early access},
   doi     = {10.1109/TGRS.2020.2985400},
   url     = {https://ieeexplore.ieee.org/document/9079477},
}
```
