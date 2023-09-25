# A Log Likelihood Fit for Extracting Neutrino Oscillation Parameters

This project aims to compute the neutrino mixing angle, squared mass difference and neutrino cross-section using multidimensional optimisation techniques including the Univariate method, Newton method, Quasi-Newton method and Gradient  method. All optimisation algorithms are developed from scratch. Since the function is not differentiable, numerical methods are implemented to estimate the first-order and second-order derivatives of the function using the forward-difference scheme.

Please see the report, `Poon-Anson-CP2020-Project1-Report`, for details on the theoretical background, approach and analysis.

## Usage

Before running the scripts, install the required Python libraries:

```
pip install -r requirements.txt
```
To run the script, run the following in the terminal:
```
python class_results.py
```
This would produce all plots for analysis and gives the results of the optimisation (neutrino mixing angle, squared mass difference and neutrino cross-section) from different optimisation algorithms.
## Reports and Descriptions
`Projects.pdf` is the project description given by the lecturers.

`Poon-Anson-CP2020-Project1-Report.pdf` is the submitted report presenting the theoretical background of the neutrino mixing problems, the technical approach and the analysis of the different optimisation techniques and the results.

## Data

The data is simulated data is stored at `https://www.hep.ph.ic.ac.uk/~ms2609/CompPhys/neutrino_data/awp18.txt`.

## Python Scripts

There are two Python scripts:
- `functions_class.py` contains the class NLL is defined with attributes including the expected unoscillated flux of neutrinos of different energies, the observed number of neutrinos of different energies, The distance the neutrinos travel and The list of energies of the neutrinos. This class has methods that calculate the neutrino oscillation probability and negative log-likelihood. All four optimisation algorithms are also defined as methods here
- `class_results.py` contains validations of implemented functions and attempts to the main tasks as described in `Projects.pdf`. This is also where the results and graphs are obtained for the report.



