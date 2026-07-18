# Step 11 Test Evaluation

The table below summarizes the test-set performance of all evaluated models.

```csv
model,accuracy,precision,recall,f1_score,attack_precision,attack_recall,attack_f1_score,macro_f1_score,roc_auc,attack_roc_auc
decision_tree,0.9626,0.9438,0.9952,0.9689,0.9928,0.9167,0.9532,0.961,0.9815,0.9815
random_forest,0.9613,0.9413,0.9959,0.9678,0.9938,0.9127,0.9515,0.9597,0.9833,0.9833
linear_svm,0.9122,0.8832,0.9791,0.9287,0.9654,0.818,0.8856,0.9072,0.9369,0.9369
cnn_lstm,0.8841,0.8761,0.9337,0.904,0.8973,0.8144,0.8538,0.8789,0.9068,0.9068
lstm,0.8576,0.8554,0.9103,0.882,0.8614,0.7836,0.8206,0.8513,0.8918,0.8918
cnn,0.6135,0.843,0.4161,0.5572,0.5205,0.891,0.6571,0.6071,0.7747,0.7747
cnn_lstm_attention,0.5698,0.8826,0.3043,0.4526,0.4909,0.9431,0.6457,0.5491,0.6726,0.6726
autoencoder,0.7151,0.7253,0.8248,0.7719,0.6948,0.5609,0.6207,0.6963,0.7464,0.7464
```