Code submission for 2237 Optimal Auctions through Deep Learning [ICML 2019]

Requirements: Python 2.7 and the following packages: 
1. Tensorflow
2. Numpy
3. Matplotlib
4. Easydict # `pip install easydict` if you don't have it


Running Experiments:
Check <expt_name>_config.py in cfgs folder for setting the hyperparameters, output directory etc.

For I - VII:

To train networks: >>python run_train.py <expt_name>
To run test: >>python run_test.py <expt_name>
To compute baselines: >>python run_baseline.py <expt_name>

Setting        <expt_name>
  (I)      additive_1x2_uniform
  (II)     unit_1x2_uniform_23
  (III)    additive_2x2_uniform
  (IV)     CA_sym_uniform_12
  (V)      CA_asym_uniform_12_15
  (VI)     additive_3x10_uniform
  (VII)    additive_5x10_uniform


For (a) - (f) in Appendix:

To train networks: >>python run_train.py <expt_name>
To run test: >>python run_test.py <expt_name>
To compute baselines: >>python run_baseline.py <expt_name>

Setting       <expt_name>
  (a)    additive_1x2_uniform_416_47
  (b)    additive_1x2_uniform_triangle
  (c)    unit_1x2_uniform
  (d)    additive_1x10_uniform
  (e)    additive_1x2_uniform_04_03
  (f)    unit_2x2_uniform
       
    
For visualizing:
Check visualize_<expt_name>.ipynb 

