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
6) Scripts and configuration files, which are hard-coded with parameters that invoke a very specific Discover invocation, have typically originated in the separate Model projects and tweaked to run in this environment.
7) Each FM is entirely independent and has a unique runtime signature.
8) Output formats vary across FMs.
9) Runtime assistance can be nominally supported by the development team, but FM model architecture expertise is not provided.

## Objectives

- Library to process FMs using GPU and CPU parallelization.
- Machine Learning and Deep Learning inference applications.
- Example scripts for a quick AI/ML start with your own data.

## Contributors

- **Glenn Tamkin**: [glenn.s.tamkin@nasa.gov](mailto:glenn.s.tamkin@nasa.gov)
- **Jian Li**: [jian.li@nasa.gov](mailto:jian.li@nasa.gov)
- **Jordan Alexis Caraballo-Vega**: [jordan.a.caraballo-vega@nasa.gov](mailto:jordan.a.caraballo-vega@nasa.gov)
---
# <b> User Guide </b>

This User Guide reflects instructions for running inference scripts on Discover only.

## <b> Running QEFM Foundation Models Inference scripts </b>

Allocate a GPU before running the inference scripts:

### <b> GPU Allocation (CLI) </b>

```bash
  $ salloc --gres=gpu:1 --mem=60G --time=1:00:00 --partition=gpu_a100 --constraint=rome --ntasks-per-node=1 --cpus-per-task=10
```

### <b> Command-Line Interface (CLI) </b>

To run all of the Foundation Model tasks with **qefm-core** in one script, use the following command:

```bash
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-ensemble.sh <Container name>
```

To run a specific Foundation Model task with **qefm-core**, use the following command:

```bash
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-<Foundation model name>.sh <Container name>
```

### <b> Common CLI Arguments </b>
| Command-line-argument | Description                                         |Required/Optional/Flag | Default  | Example                  |
| --------------------- |:----------------------------------------------------|:---------|:---------|:--------------------------------------|
| `<Container name>`                  | Name of Singularity container image (or sandbox)                                | Required | N/A      |`qefm-core.sif`         |
| `<Foundation Model name>`                  | Short title of Foundation Model                               | Required | N/A      |`ensemble`, `aurora`, `fourcastnet`, `graphcast`, `pangu`,`privthi`  |

### <b> Examples </b>

**Run Inference for **All** Foundation Models**:
```shell
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-ensemble.sh qefm-core.sif
```
**Run Inference for Aurora Foundation Model**:
```shell
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-aurora.sh qefm-core.sif
```
**Run Inference for Fourcastnet Foundation Model**:
```shell
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-fourcastnet.sh qefm-core.sif
```
**Run Inference for GraphCast Foundation Model**:
```shell
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-graphcast.sh qefm-core.sif
```
**Run Inference for Pangu Foundation Model**:
```shell
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-pangu.sh qefm-core.sif
```
**Run Inference for Privthi Foundation Model**:
```shell
$ /discover/nobackup/projects/QEFM/qefm-core/tests/fm-privthi.sh qefm-core.sif
```
