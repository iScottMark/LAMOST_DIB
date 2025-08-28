Quick Start
=================

This is a quick start guide for measuring Diffuse Interstellar Bands (DIBs) in the LAMOST spectra using the ``dibkit`` package.

Installation
------------

**Step 1: Clone or download the repository**

You can clone the repository from GitHub:

.. code-block:: bash

   git clone https://github.com/iScottMark/LAMOST_DIB.git

and then navigate to the cloned directory:

.. code-block:: bash

   cd LAMOST_DIB

Or, you can download the ZIP file from the `GitHub page <https://github.com/iScottMark/LAMOST_DIB>`_, and extract it to your desired location.



**Step 2: Set up the environment**

And then, create a ``mamba`` (recommended) or ``conda`` environment with the required dependencies listed in the ``environment.yaml`` file:

.. code-block:: bash

    mamba env create -f environment.yaml
    mamba activate dib


Or using `conda` with the similar commands:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate dib


Usage
-----

1. Download the necessary data files as introduced in the `Data Preparation <data_preparation.html>`_ page.

2. Run the Jupyter Notebook placed in the ``tutorials`` folder to see how to measure DIBs block by block (`Tutorials <tutorials/index.html>`_ page is converted from these notebooks).
   
3. In debug mode, step into the function defined in the ``dibkit`` package to understand how it works (*I will provide more detailed API documentation and tutorials in the future*).

4. The ``scripts`` folder contains the batch measurement scripts for both cool and hot stellar spectra from LAMOST. Users can customize these scripts to fit their own needs.


Citation
---------

If you use the code in your research, please cite the following paper:

.. code-block:: bibtex

    @ARTICLE{2024A&A...691A.282M,
        author = {{Ma}, Xiao-Xiao and {Chen}, Jian-Jun and {Luo}, A. -Li and {Zhao}, He and {Shi}, Ji-Wei and {Chen}, Jing and {Liang}, Jun-Chao and {Ma}, Shu-Guo and {Qu}, Cai-Xia and {Jiang}, Bi-Wei},
            title = "{Measuring the diffuse interstellar bands at 5780, 5797, and 6614 {\r{A}} in low-resolution spectra of cool stars from LAMOST}",
        journal = {\aap},
        keywords = {dust, extinction, ISM: lines and bands, Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics, Physics - Data Analysis, Statistics and Probability},
            year = 2024,
            month = nov,
        volume = {691},
            eid = {A282},
            pages = {A282},
            doi = {10.1051/0004-6361/202451408},
    archivePrefix = {arXiv},
        eprint = {2409.19539},
    primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&A...691A.282M},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{2025RAA....25i7001M,
        author = {{Ma}, Xiao-Xiao and {Luo}, A. -Li and {Chen}, Jian-Jun and {Chen}, Jing and {Liang}, Jun-Chao},
            title = "{Measurements of the Diffuse Interstellar Bands at 5780, 5797, and 6614 {\r{A}} in the Hot Stellar Spectra of the LAMOST LRS DR10}",
        journal = {Research in Astronomy and Astrophysics},
        keywords = {(ISM:) dust, extinction, Interstellar Medium (ISM), Nebulae, stars: early-type, Astrophysics of Galaxies, Solar and Stellar Astrophysics},
            year = 2025,
            month = sep,
        volume = {25},
        number = {9},
            eid = {097001},
            pages = {097001},
            doi = {10.1088/1674-4527/ade58d},
    archivePrefix = {arXiv},
        eprint = {2506.14346},
    primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2025RAA....25i7001M},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


Change Log
----------

**v1.0 (2025-08-26)**

- Initial release of the ``dibkit`` package, and the tutorials of DIB measurement in the LAMOST spectra.
