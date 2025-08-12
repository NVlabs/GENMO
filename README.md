# GENMO: A GENeralist Model for Human MOtion


## Setup

Please see [installation](docs/INSTALL.md) for details.


## Training
Locally train
```bash
python tools/train_v2.py exp=genmo/genmo_lg exp_name_var=localtest
```

Submit jobs to slurm
```bash
python tools/train_slurm.py -c genmo/genmo_lg -v 'test' -u ${USER} -a nvr_lpr_digitalhuman -g 2
```


## Demo
```bash
python tools/demo/demo_slam.py --video example.mp4 --output_root outputs/demo
```
