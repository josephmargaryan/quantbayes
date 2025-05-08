========================================================================
                             README
========================================================================

Prerequisites
-------------
Create a Python 3 venv and install dependencies:

    python3 -m venv venv
    source venv/bin/activate       # on Windows: venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt

Data files
----------
Make sure `nonlinear_svm_data.csv` sits alongside the `code/` folder (or adjust paths
below to wherever you put it).

Running the experiments
-----------------------

1) SVM with kernels (Questions 1 & 2)

    cd code
    python3 main_q1_svm.py --data nonlinear_svm_data.csv

   This will print out the hinge‐loss tables and save:
     - images/linear_decision.png
     - images/rbf_decision.png

2) Logistic regression on MNIST (Questions 3 – 6)

    cd code
    python3 main_q2_logreg.py

   This will print learning‐rate and batch‐size loss summaries and save:
     - images/loss_const_gamma001.png
     - images/loss_batchsize.png
     - images/loss_diminish.png

Output
------
– All printed values (losses, etc.) gets printed in the terminal
– All figures are saved under `code/images/`

========================================================================

