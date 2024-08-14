The postprocessing contains three operations: clearing small objects, analysis and plotting figures. All the inputs and outputs are expected to be stored in the directory used in prediction.

## Clear small objects

```bash
sbatch clear_control.sh
sbatch clear_rheb.sh
```


## Analysis

```bash
sbatch control.sh
sbatch rheb.sh
```

## Plot figures

This stage is prune to frequently update.