# Optimal Auctions through Deep Learning

### Getting Started

Install the following packages:
- Python 2.7 
- Tensorflow
- Numpy and Matplotlib packages
- Easydict - `pip install easydict`


### Running Experiments in the paper

Default hyperparameters specified in cfgs/

For training the network, testing the mechanism learnt and computing the baselines, run:
```
cd regretNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```

setting\_no  |      setting\_name |
 :---:   | :---: |
  (I)    |  additive\_1x2\_uniform |
  (II)   | unit\_1x2\_uniform\_23 |
  (III)  | additive\_2x2\_uniform |
  (IV)   | CA\_sym\_uniform\_12 |
  (V)    | CA\_asym\_uniform\_12\_15 |
  (VI)   | additive\_3x10\_uniform |
  (VII)  | additive\_5x10\_uniform |
  

 For additional experiments in the appendix, run:
 ```
python run_train_additional.py [setting_name]
python run_test_additional.py [setting_name]
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


### Running single bidder auctions using RochetNet

```
cd rochetNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```
setting\_no  |      setting\_name |
 :---:  | :---: |
  (a)   |  additive\_1x2\_uniform |
  (b)   |   additive\_1x2\_uniform\_416\_47
  \(c\) |   additive\_1x2\_uniform\_triangle
  (d)   |   additive\_1x2\_uniform\_04\_03
  (e)   |  additive\_1x10\_uniform
  (f)   |   unit\_1x2\_uniform
  (g)   |   unit\_1x2\_uniform\_23
  
  ### Running single item auctions using MyersonNet

```
cd myersonNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```
setting\_no  |      setting\_name |
 :---:  | :---: |
  (a)   |  uniform |
  (b)   |   asymmetric_uniform
  \(c\) |   exponential
  (d)   |   irregular
 
