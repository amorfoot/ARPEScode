Code contents:

SET 1:
- Initialisation.ipynb
    This is a Jupyter Notebook where you load an ARPES cut which will have dimensions (momentum (size=x), Enenrgy (size=y), Intensity (size=x,y)
    After loading it will allow you to find the Fermi level and the center of the High-Symmetry point

- Band_Collection.ipynb
    This is another Jupyter Notebook where again you load in the an ARPES cut, but also copy over your values of the High-Symmerty point and Fermi level
    Then the you can conevert the cut into k-space
    The you can open the interactive window which allows you plot the position of the bands

- Helpful_functions.py
    This is a small python file including some functions that are imported within Initialisation.ipynb and Band_Collection.ipynb

- Curvature_Kspace_functions.py
    This is another python file including the functions imported by Band_Collection.ipynb which are used to perform the k-space transform and the curvature transform

Set 2:
- ARchPES_v2.ipynb
    This is a Jupyter Notebook. After initially setting the current directory, you can run all the cells and a application should open up.
    This app is used for experiments when wanting to analyse various scans quickly to make decisions on scan alignment and quality.

Set 3:
- Logbook_maker.ipynb
    This is a Jupyter Notebook used to create an excel file which provides all the information of each scan within a directory.
