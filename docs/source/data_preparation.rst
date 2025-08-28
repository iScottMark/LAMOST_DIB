Data Preparation
================

To successfully run the two tutorial Jupyter notebooks included in this project, you need to prepare **two types of data**:

1. **Catalog files**
2. **Spectra files**

Catalogs
--------

1. **Overview**

   Two FITS-format catalog files are required:

   - ``cool_lamost_with_Gaia_ebv.fits`` — catalog of cool stars from LAMOST  
   - ``hot_lamost_with_ebv_snr_gt50.fits`` — catalog of hot stars from LAMOST  

   For details on how these catalogs were generated, please refer to the DATA sections in the corresponding papers:  
   - Cool star catalog: `Ma et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024A%26A...691A.282M/abstract>`_  
   - Hot star catalog: `Ma et al. (2025) <https://ui.adsabs.harvard.edu/abs/2025RAA....25i7001M/abstract>`_  

2. **Download**

   Both catalog files can be downloaded from Zenodo:  
   https://doi.org/10.5281/zenodo.16734593  

3. **Storage location**

   Place the catalog files in the ``inputs`` folder under the project root. The directory structure should look like this:

   .. code-block:: text

      LAMOST_DIB
      |-- inputs
         |-- cool_lamost_with_Gaia_ebv.fits
         |-- hot_lamost_with_ebv_snr_gt50.fits
      |-- dibkit
      |-- scripts
      |-- ...


Spectra
-------

1. **Overview**

   The spectra used in this project are from the public release of **LAMOST DR10 V1.0**.  
   More information can be found at: https://www.lamost.org/dr10/v1.0/  

2. **Download**

   This project **does NOT provide** the entire LAMOST spectra.  
   Users should download spectra directly from the `LAMOST website <https://www.lamost.org/dr10/v1.0/>`_.  

   However, for convenience, we provide a compressed archive containing the demo LAMOST spectra required for the two tutorials:  
   https://doi.org/10.5281/zenodo.16734593  

   The archive file is named: ``demo_lamost_lrs_dr10_v1.0.tar.gz``  

3. **Storage location**

   After extraction, the LAMOST spectra should be placed in the ``datasets`` folder under the project root. The directory structure should look like this:

   .. code-block:: text

      LAMOST_DIB
      |-- inputs
      |-- datasets
         |-- demo_lamost_lrs_dr10_v1.0
            |-- 20120217
                  |-- B5597505
                     |-- spec-55975-B5597505_sp08-204.fits.gz
            |-- 20120403
            |-- ...
      |-- dibkit
      |-- scripts
      |-- ...

   The LAMOST spectra are organized hierarchically by **observation date → sky region → spectrum**, which explains the directory layout above.
