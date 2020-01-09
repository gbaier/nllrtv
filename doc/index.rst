.. nllrtv documentation master file, created by
   sphinx-quickstart on Thu Jan  9 11:15:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nllrtv's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Check out the examples directory.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



Stack despeckling module
------------------------

.. automodule:: nllrtv.despecks
.. autofunction:: nllrtv.despecks.ks_sim
.. autofunction:: nllrtv.despecks.despecks
.. autofunction:: nllrtv.despecks.despecks_lrtv

Bregman module
--------------

.. automodule:: nllrtv.bregman
.. autofunction:: nllrtv.bregman.tv
.. autofunction:: nllrtv.bregman.l1_tv
.. autofunction:: nllrtv.bregman.l2_tv


Weighted Nuclear Norm module
----------------------------

.. automodule:: nllrtv.wnnm
.. autofunction:: nllrtv.wnnm.rpca
.. autofunction:: nllrtv.wnnm.rpca_wnnm
.. autofunction:: nllrtv.wnnm.rpca_wnnm_tv


Random module
-------------

.. automodule:: nllrtv.random
.. autofunction:: nllrtv.random.multivariate_complex_normal
.. autofunction:: nllrtv.random.exp_decay_coh_mat
.. autofunction:: nllrtv.random.stack
