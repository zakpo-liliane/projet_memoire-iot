# Step 10 Hyperparameter Tuning

This step tunes the deep learning models on the validation split.
Balance method: smote. Loss function: binary_crossentropy.

## Best configs

```csv
model,best_config,attack_f1_score,filters1,filters2,lstm_units,attention_heads,attention_key_dim,dense_units,dropout,learning_rate
cnn_lstm_attention,cfg1,0.7159,16,32,16,2,16,16,0.2,0.001
```

## Search results

```csv
model,accuracy,precision,recall,f1_score,attack_precision,attack_recall,attack_f1_score,macro_f1_score,roc_auc,attack_roc_auc,train_seconds,config_name,param_filters1,param_filters2,param_lstm_units,param_attention_heads,param_attention_key_dim,param_dense_units,param_dropout,param_learning_rate
cnn_lstm_attention,0.6258,0.8439,0.3087,0.452,0.577,0.9429,0.7159,0.584,0.6831,0.6831,22.4421,cfg1,16,32,16,2,16,16,0.2,0.001
cnn_lstm_attention,0.5,0.0,0.0,0.0,0.5,1.0,0.6667,0.3333,0.6181,0.6181,38.1486,cfg2,32,64,32,4,16,32,0.3,0.001
```