## Profile individual seqlens
```
python benchmarks/bench_batch_attention.py
python benchmarks/plotting/plot_persistent_individual.py
```

## Profile all individual seqlens
```
bash benchmarks/plot_individual_seqlens.sh
```

## Profile throughput trends (fix prefill seqlen and vary decode seqlen)
```
bash benchmarks/plot_seqlen_trends.sh
```
