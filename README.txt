# A Log Likelihood Fit for Extracting Neutrino Oscillation Parameters

This project aims to compute the neutrino mixing angle, squared mass difference and neutrino cross-section using multidimensional minimising techniques including Univariate method, Newton method, Quasi-Newton method and Gradient  method. All optimisation algorithms are developed from scratch. 

Please see the report, `Poon-Anson-CP2020-Project1-Report`, for details on the theoretical background, approach and analysis.

##

Projects.pdf is the project description given by the lecturers.

Poon-Anson-CP2020-Project1-Report.pdf is the submitted report presenting the theoretical background of the neutrino mixing problems, analysis of the different optimisation techniques and the results.

There are two Python files in total: functions_class.py and class_results.py and can be run by Python3.

functions_class.py contains the class where the optimisation algorithms are defined. This file gives no output after running. functions_class.py needs to be run before using class_results.py.

class_results.py contains validations of implemented functions and attempts to the main tasks. It is important to run the first and second cell before running any other cell, which imports functions from libraries and functions_class.py, while the second cell imports the data and creates the object that is later used throughout the rest of this file.

For each question, in class_results.py, it is important to run in the order of cells and cells by cells. For example, one needs to run 3.2 Fit function before running 3.3 Likelihood function. It is also fine to just run the whole class_results.py after running the functions_class.py.

To look at the graphs, it is also important to enlarge the graph to avoid overlapping of axes and titles.

