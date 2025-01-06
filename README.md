# Quantitative Evaluation of Foundation Models

Python framework for evaluating Foundation Models (FM).  

## Documentation

- Latest: https://github.com/nasa-nccs-hpda/qefm-core/blob/main/README.md

# qefm-core

This framework consists of a container that hosts the dependencies required for an extendable collection of Models.  Snapshots of model source code are also captured along with supporting inference scripts.  Configuration files that specify runtime parameters such as data paths and model tunings are also included.  Example runs illustrate how to invoke scripts that execute model inferences.

NOTE:  The initial version of this project is deployed with restrictions:
1) The container can be deployed on any platform with Singularity or Docker; however, associated model checkpoints and statistics file are not included.
3) In order to run the canned Python/Bash scripts, the user must log into the Discover cluster and execute the runtime scripts described below.
4) All paths reflect a static Discover enviroment, referencing both fully-specified and relative paths to the input data.
5) To change default parameters, a copy of the runtime scripts should be made by the user and modified accordingly.
6) Scripts and configuration files, which are hard-coded with parameters that invoke a very specific Discover invocation, have typically originated in the the separate Model projects and tweaked to run in this environment.
7) Each FM is entirely independent and has a unique runtime signature.
8) Output formats vary across FMs.
9) Runtime assistance can be nominally supported by the development team, but FM model architecture expertise is not provided.

## Objectives

- Library to process FMs using GPU and CPU parallelization.
- Machine Learning and Deep Learning inference applications.
- Example scripts for a quick AI/ML start with your own data.

## Contributors

- **Jordan Alexis Caraballo-Vega**: [jordan.a.caraballo-vega@nasa.gov](mailto:jordan.a.caraballo-vega@nasa.gov)
- **Glenn Tamkin**: [glenn.s.tamkin@nasa.gov](mailto:glenn.s.tamkin@nasa.gov)
- **Jian Li**: [jian.li@nasa.gov](mailto:jian.li@nasa.gov)

---
# <b> User Guide </b>

This User Guide reflects instructions for running inference scripts on Discover only.

##  1. <b> FM Ensemble </b>

|Name|Pretrain|Resolution|Channels | Parameters|
|---|---|---|---|---|
|SatVision-TOA-GIANT|MODIS-TOA-100-M|128x128|14|3B|

### Accessing the Model

Model Repository: [HuggingFace](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128)

#### **Clone the Model Checkpoint**

1. Load `git-lfs`:
```bash
  $ salloc --gres=gpu:1 --mem=60G --time=1:00:00 --partition=gpu_a100 --constraint=rome --ntasks-per-node=1 --cpus-per-task=10
  $ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-ensemble.sh  qefm-core-20241229-sandbox
```
```bash
  git lfs install
```

2. Run the Ensemble:
```bash
  $ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-ensemble.sh  qefm-core-20241229-sandbox
```

<b> Note: Using SSH authentication </b>

Ensure SSH keys are configured. Troubleshooting steps:
- Check SSH connection:
```bash
	ssh -T git@hf.co # If reports back as anonymous follow the next steps
```
- Add your SSH key:
```bash
	eval $(ssh-agent)
	ssh-add ~/.ssh/your-key # Path to your SSH key
```

## <b> Running SatVision-TOA Pipelines </b>

### <b> Command-Line Interface (CLI) </b>

To run tasks with **PyTorch-Caney**, use the following command:

```bash
$ python pytorch-caney/pytorch_caney/ptc_cli.py --config-path <Path to config file>
```

### <b> Common CLI Arguments </b>
| Command-line-argument | Description                                         |Required/Optional/Flag | Default  | Example                  |
| --------------------- |:----------------------------------------------------|:---------|:---------|:--------------------------------------|
| `-config-path`                  | Path to training config                                   | Required | N/A      |`--config-path pytorch-caney/configs/3dcloudtask_swinv2_satvision_gaint_test.yaml`         |
| `-h, --help`               | show this help message and exit                  | Optional | N/a      |`--help`, `-h` |


### <b> Examples </b>

**Run 3D Cloud Task with Pretrained Model**:
```shell
$ python pytorch-caney/pytorch_caney/ptc_cli.py --config-path pytorch-caney/configs/3dcloudtask_swinv2_satvision_giant_test.yaml
```
**Run 3D Cloud Task with baseline model**:
```shell
$ python pytorch-caney/pytorch_caney/ptc_cli.py --config-path pytorch-caney/configs/3dcloudtask_fcn_baseline_test.yaml
```

**Run SatVision-TOA Pretraining from Scratch**:
```shell
$ python pytorch-caney/pytorch_caney/ptc_cli.py --config-path pytorch-caney/configs/mim_pretrain_swinv2_satvision_giant_128_onecycle_100ep.yaml
```

## **3. Using Singularity for Containerized Execution**

**Shell Access**

```bash
$ singularity shell --nv -B <DRIVE-TO-MOUNT-0> <PATH-TO-CONTAINER>
Singularity> export PYTHONPATH=$PWD:$PWD/pytorch-caney
```

**Command Execution**
```bash
$ singularity exec --nv -B <DRIVE-TO-MOUNT-0>,<DRIVE-TO-MOUNT-1> --env PYTHONPATH=$PWD:$PWD/pytorch-caney <PATH-TO-CONTAINER> COMMAND
```

### **Example**

Running the 3D Cloud Task inside the container:

```bash
$ singularity shell --nv -B <DRIVE-TO-MOUNT-0> <PATH-TO-CONTAINER>
Singularity> export PYTHONPATH=$PWD:$PWD/pytorch-caney
Singularity> python pytorch-caney/pytorch_caney/ptc_cli.py --config-path pytorch-caney/configs/3dcloudtask_swinv2_satvision_giant_test.yaml
```

---
## Installation

The following library is intended to be used to accelerate the development of data science products for remote sensing satellite imagery, or other applications. `pytorch-caney` can be installed by itself, but instructions for installing the full environments are listed under the `requirements` directory so projects, examples, and notebooks can be run.

**Note:** PIP installations do not include CUDA libraries for GPU support. Make sure NVIDIA libraries are installed locally in the system if not using conda/mamba.

```bash
module load singularity # if a module needs to be loaded
singularity build --sandbox pytorch-caney-container docker://nasanccs/pytorch-caney:latest
```
