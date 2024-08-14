Run *python predict.py --help* to show parameters. The outputfile will be used for postprocessing. Create a directory to store the outputfile and other output files generated during postprocessing is recommended.

### Run prediction on both control and rheb neurons on ARC:

```bash
sbatch control.sh
sbatch rheb.sh
```