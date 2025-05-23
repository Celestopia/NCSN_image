Implementation of NCSNv2 on image generation tasks.



## Project Setting

To run the project, you should download `model_weights` from [link,placeholder] and put it in the root directory of the project.

The Project structure is:
```sh
root
├── configs
│   ├── celeba.yml
│   └── cifar10.yml
├── evaluation
│   ├── inception.py
│   └── metrics.py
├── exp # directory created to save experiment data
│   ├── exp1
│   ├── exp2
│   ├── exp3
│   └── ...
├── model_weights # neural network weights and inception statistics
│   ├── celeba
|   |   ├── best_checkpoint_with_denoising.pth
|   |   └── celeba_test_fid_stats.npz
│   ├── cifar10
|   |   ├── best_checkpoint_with_denoising.pth
|   |   └── fid_stats_cifar10_train.npz
│   └── hub
|       └── checkpoints
|           └── pt_inception-2015-12-05-6726825d.pth
├── models
│   ├── ema.py
│   ├── layers.py
│   ├── refinenet.py
│   └── normalization.py
├── utils
│   ├── __init__.py
│   ├── format.py
│   └── log.py
├── dynamics.py
├── main.py
└── README.md
```

Command line arguments for `main.py` are as follows:

- `--config`: **Required**  
  Specifies the path to the configuration file, which is placed in the `configs` directory.  
  **Example**: `--cifar10.yml`

- `--seed`: **Optional**  
  Sets the random seed to ensure reproducibility of the experiment.  
  **Default**: `1234`  
  **Example**: `--seed 4321`

- `--exp`: **Optional**  
  Specifies the path for saving experiment-related data.  
  **Default**: `exp`  
  **Example**: `--exp ./experiment_data`

- `--comment`: **Optional**  
  Adds a comment string for the experiment to record additional information.  
  **Default**: `''` (empty string)  
  **Example**: `--comment 'This is a test experiment'`

- `--exp_name`: **Optional**  
  Specifies the name of the experiment.  
  **Default**: `default`  
  **Example**: `--exp_name my_experiment`

- `--exp_dir_suffix`: **Optional**  
  Adds a suffix string to the experiment directory to distinguish different experiments.  
  **Default**: `None`  
  **Example**: `--exp_dir_suffix _v2`

- `--k_p`: **Optional**  
  Coefficient for Proportional Gain.  
  **Default**: `None`  
  **Example**: `--k_p 0.1`

- `--k_i`: **Optional**  
  Coefficient for Integral Gain.  
  **Default**: `None`  
  **Example**: `--k_i 0.01`

- `--k_d`: **Optional**  
  Coefficient for Differential Gain.  
  **Default**: `None`  
  **Example**: `--k_d 0.001`

- `--k_i_decay`: **Optional**  
  Decay rate for Integral Gain.  
  **Default**: `None`  
  **Example**: `--k_i_decay 0.99`

- `--k_d_decay`: **Optional**  
  Decay rate for Differential Gain.  
  **Default**: `None`  
  **Example**: `--k_d_decay 0.98`

- `--n_steps_each`: **Optional**  
  Number of sampling steps per noise level.  
  **Default**: `None`  
  **Example**: `--n_steps_each 100`

- `--num_classes`: **Optional**  
  Number of noise levels.  
  **Default**: `None`  
  **Example**: `--num_classes 10`

For example, to sample CIFAR10 images with $k_p=2.0$, $k_i=0.5$, $k_d=4.5$, $100$ noise levels, and $1$ step per noise level, run
```sh
python main.py --config cifar10.yml --exp_name k_p=2.0_k_i=0.5_k_d=4.5_100x1_steps --k_p 2.0 --k_i 0.5 --k_d 4.5 --num_classes 100 --n_steps_each 1
```



