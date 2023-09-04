# OASIS v1.0.0

[![DOI](https://zenodo.org/badge/685457042.svg)](https://zenodo.org/badge/latestdoi/685457042)

## Description

This is a stable version of the OASIS activity-based framework. This repository includes the simulation code, based on the Python API of the CPLEX solver.

For more details, you can find the technical reports in the **Literature** folder, and the full documentation [here](https://oasis-abm.readthedocs.io/en/latest/).

## Installation guide   

We recommend creating a new environment using the provided requirements file to install the correct packages. 
**NB: You need a valid CPLEX user license to use the solver, which can be obtained [here](https://www.ibm.com/academic/topic/data-science).**

## Tutorial 

An example of how to run the simulator is provided in the ``demo`` notebooks. Your main function should be added in the ``runner`` script, which you can run by typing the following command:
``python runner.py``

## Citation

If you found this repository useful, you can acknowledge the authors by citing:

* Pougala, J., Hillel, T., and Bierlaire M. (2023) *OASIS: Optimisation-based Activity Scheduling with Integrated Simultaneous choice dimensions*, Transportation Research Part C: Emerging Technologies, vol. 155, p. 104291, DOI: [https://doi.org/10.1016/j.trc.2023.104291](https://doi.org/10.1016/j.trc.2023.104291)


