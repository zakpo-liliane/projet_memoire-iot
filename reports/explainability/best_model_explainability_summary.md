# Explainability Summary

The final decision tree model was analyzed using tree-based feature importance and permutation importance on the held-out test set.

## Top features

```csv
feature,tree_importance,permutation_importance_mean,permutation_importance_std
network_tcp-flags-fin_count,0.000111,0.215771,0.000592
network_packets_dst_count,0.0,0.066853,0.000548
log_data-types_count,0.001169,0.056916,0.000631
network_ip-flags_min,3.3e-05,0.055058,0.000549
network_ip-length_min,0.001,0.039752,0.000464
network_ports_dst_count,0.0,0.035507,0.000321
network_ports_src_count,0.0,0.035432,0.000258
network_time-delta_avg,0.019567,0.034817,0.00041
log_data-ranges_std_deviation,0.0,0.033631,0.00042
network_window-size_std_deviation,0.037355,0.031901,0.0002
network_ips_all_count,0.000258,0.028118,0.000222
network_ip-flags_max,0.0,0.021571,0.000227
network_tcp-flags-ack_count,0.0,0.018573,0.000246
network_tcp-flags_min,0.000464,0.01788,0.000263
log_data-ranges_avg,0.0,0.008162,0.000108
network_packet-size_avg,0.000131,0.004472,0.000152
network_tcp-flags-psh_count,0.0,0.004131,7.3e-05
log_messages_count,0.00214,0.003152,0.000137
network_payload-length_std_deviation,0.0,0.003137,8.3e-05
log_data-ranges_min,0.0,0.003125,7.8e-05
```