# Statistical Analysis

## McNemar test on test-set predictions

```csv
model_a,model_b,b01,b10,n_discordant,p_value
autoencoder,cnn,14869,25317,40186,0.0
linear_svm,lstm,1035,6643,7678,0.0
decision_tree,lstm,335,11131,11466,0.0
decision_tree,linear_svm,289,5477,5766,0.0
cnn_lstm_attention,random_forest,41586,1319,42905,0.0
cnn_lstm_attention,lstm,36422,6820,43242,0.0
cnn_lstm_attention,linear_svm,40558,5348,45906,0.0
cnn_lstm_attention,decision_tree,41544,1146,42690,0.0
cnn_lstm,random_forest,8193,249,8442,0.0
cnn_lstm,lstm,695,3416,4111,0.0
cnn_lstm,linear_svm,3890,1003,4893,0.0
cnn_lstm,decision_tree,8353,278,8631,0.0
linear_svm,random_forest,5318,261,5579,0.0
cnn_lstm,cnn_lstm_attention,5503,37826,43329,0.0
cnn,lstm,30141,5031,35172,0.0
cnn,linear_svm,35162,4444,39606,0.0
cnn,decision_tree,36701,795,37496,0.0
cnn,cnn_lstm_attention,2227,6719,8946,0.0
cnn,cnn_lstm,32319,4488,36807,0.0
autoencoder,random_forest,25688,361,26049,0.0
autoencoder,lstm,16581,1919,18500,0.0
autoencoder,linear_svm,21889,1619,23508,0.0
autoencoder,decision_tree,25854,396,26250,0.0
autoencoder,cnn_lstm_attention,16340,31280,47620,0.0
autoencoder,cnn_lstm,19185,1802,20987,0.0
cnn,random_forest,36510,735,37245,0.0
lstm,random_forest,10988,323,11311,0.0
decision_tree,random_forest,252,383,635,0.0
```

## One-way ANOVA on baseline cross-validation folds

```csv
test,statistic,p_value,groups,folds_per_group
one_way_anova,21812.9848,0.0,3,5
```

## Wilcoxon signed-rank tests on baseline cross-validation folds

```csv
model_a,model_b,statistic,p_value,n_pairs
decision_tree,linear_svm,0.0,0.0625,5
linear_svm,random_forest,0.0,0.0625,5
decision_tree,random_forest,1.0,0.125,5
```