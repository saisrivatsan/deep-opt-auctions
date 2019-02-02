# Optimal Auctions through Deep Learning

### Required Packages

- Python 2.7 
- Tensorflow
- Numpy and Matplotlib packages
- Easydict - `pip install easydict`


### Running Experiments in the paper

To run experiments in the paper with with default hyperparameters specified in cfgs/

```
cd regretNet
```

For training the network, testing the mechanism learnt and computing the baselines, run:
```
python run_train.py [setting_name]
python run_train.py [setting_name]
python run_baseline.py [setting_name]
```

setting\_no  |      setting\_name |
 :---:   | :---: |
  (I)    |  additive\_1x2\_uniform |
  (II)   | unit\_1x2\_uniform_23 |
  (III)  | additive\_2x2\_uniform |
  (IV)   | CA\_sym\_uniform\_12 |
  (V)    | CA\_asym\_uniform\_12_15 |
  (VI)   | additive\_3x10\_uniform |
  (VII)  | additive\_5x10\_uniform |
  

 For additional experiments in the appendix, run:
 ```
python run_train_additional.py [setting_name]
python run_train_additional.py [setting_name]
python run_baseline_additional.py [setting_name]
```

 setting\_no  |      setting\_name |
 :---:   | :---: |
  (a) |   additive\_1x2\_uniform\_416\_47
  (b) |   additive\_1x2\_uniform\_triangle
  \(c\) |   unit\_1x2\_uniform
  (d) |  additive\_1x10\_uniform
  (e) |   additive\_1x2\_uniform\_04\_03
  (f) |   unit\_2x2\_uniform


