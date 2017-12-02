# kaggle-porto-seguros-safe-driver-prediction
119th solution for Porto Seguro's Safe Driver Prediction on Kaggle (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction).

Final submission is the rank averaging of the following 4 models (2 original models and 2 kernel models).

- l1_lgb (Public LB: 0.285)
- l1_xgb (Public LB: 0.285)
- [Base on Froza & Pascal single XGB LB (0.284)](https://www.kaggle.com/kueipo/base-on-froza-pascal-single-xgb-lb-0-284)
- [RGF + Target Encoding + Upsampling](https://www.kaggle.com/tunguz/rgf-target-encoding-0-282-on-lb)

## Train the models and predict 
To train the original models and predict, run:

```
python l1_lgb.py
python l1_xgb.py
```

## Average the predictions
To average the predictions, run:

```
python l2_rank_avg.py
```
