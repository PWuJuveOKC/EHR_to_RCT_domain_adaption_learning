# README

 This folder contains sample simulation code in python for proposed method using Kernel M-learning and Q-learning in the paper titled "Improved Treatment Rules from Randomized Trials
using Evidence from Electronic Health Records".

## Getting Started

Simulation code is included in two files.  Simulation\_ML.py contains code for Kernel M-learning and Simulation\_QL\_RF.py contains code for Q-learning using random forest regression.

### Prerequisites

Install the requirements:

```
Python2: pip install -r requirements.txt
Python3: pip3 install -r rquirements.txt
```

## Running the code
Change to corresponding working directory and run python file using command-line, for example,


```
$ python Simulation_ML.py
```
will run the simulation settings S1, S2, S3 in Kernel M-learning under general testing population. Set ```strat='prob' ```will run S1, S2, S4 in Kernel M-learning.  Set ```unknown='unknown' ``` will run scenario (ii) with an unobserved tailoring.
