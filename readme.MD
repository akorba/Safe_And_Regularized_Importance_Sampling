## Code for the article 'Adaptive Importance Sampling meets Mirror Descent: a Bias-variance tradeoff'

This implementation has been developed and tested with python3.6. 

### Reproducing the figures in the paper

To run the experiments, run:

> python python/blr/main_blr.py

To display the figures, run:

> python python/blr/plot_res_blr.py

#### Dataset

The waveform dataset that we used is from https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+(Version+1).

#### Parallelisation

Our code can be parallelised using joblib. To do so, in main_blr.py replace

```python
#Parallel(nb_cores_used)(delayed(joblib_function)(i) for i in i_list)
for i in i_list:
    joblib_function(i)
```

by

```python
Parallel(nb_cores_used)(delayed(joblib_function)(i) for i in i_list)
#for i in i_list:
#    joblib_function(i)
```