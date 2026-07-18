# EDA Summary

- Nombre de lignes : 685671
- Nombre de colonnes : 80
- Nombre de variables numeriques : 76
- Nombre de variables categorielles : 4
- Distribution binaire label1 : {'benign': 400672, 'attack': 284999}
- Top features a forte variance : ['network_packets_all_count', 'network_packets_dst_count', 'network_tcp-flags-rst_count', 'network_tcp-flags-fin_count', 'network_tcp-flags-ack_count', 'network_tcp-flags-syn_count', 'network_tcp-flags-psh_count', 'network_window-size_max', 'network_window-size_avg', 'network_packets_src_count']

Fichiers generes :
- label1_distribution.csv / .png
- label2_distribution.csv / .png
- label3_distribution.csv / .png
- label4_distribution.csv / .png
- numeric_summary.csv
- top_numeric_histograms.png
- top_feature_correlation_heatmap.png
- label1_binary_distribution.png