# Deep Learning Step 9 Summary

Ce bloc couvre l'implémentation des modeles deep learning du memoire.

## Modeles

- CNN: extraction de motifs locaux sur les features normalisées.
- LSTM: apprentissage de dependances séquentielles sur le vecteur de features.
- Autoencoder: detection d'anomalies à partir des erreurs de reconstruction sur le trafic normal.
- CNN + LSTM: modele hybride combinant convolution et memoire sequentielle.
- CNN + LSTM + Attention: extension hybride integrant un mecanisme d'attention multi-tete.

## Comparaison validation

```csv
model,accuracy,precision,recall,f1_score,roc_auc,train_seconds,attack_precision,attack_recall,attack_f1_score,macro_f1_score,attack_roc_auc,threshold
cnn,0.9523,0.9335,0.9889,0.9604,0.9765,682.3671,0.9829,0.9009,0.9401,0.9502,0.9765,
cnn_lstm,0.92,0.8955,0.9772,0.9346,0.9468,172.3452,0.9632,0.8397,0.8972,0.9159,0.9468,
lstm,0.9041,0.8834,0.963,0.9215,0.9295,566.1252,0.9404,0.8213,0.8768,0.8992,0.9295,
cnn_lstm_attention,0.5019,0.6792,0.0072,0.0142,0.6158,203.5504,0.501,0.9966,0.6668,0.3405,0.6158,
autoencoder,0.5846,0.5845,1.0,0.7377,0.9145,20.4531,1.0,0.0005,0.001,0.3694,0.9145,0.0212
```

## Critere de selection

Le meilleur modele est retenu sur la base de `attack_f1_score` pour privilegier la detection des attaques.