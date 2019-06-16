# Draft of ML algorhitms with OpenCL

## 1. Logistic regression

Comparison with sklearn implementation with sag solver:
    
    $ python3 benchmarks.py --nsamples 200000 --nfeatures 300
    using  Intel(R) OpenCL HD Graphics
    Training and evaluation on dataset:
        200000  samples
        300  features
    ### LogRegCL, 500 iter ###
        acc:  0.911025 
        time:  33.094815254211426
    ### LogReg with SAG solver, 500 iter ###
        acc:  0.9124875 
        time:  58.88489103317261