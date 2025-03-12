# Quantitative Evaluation of Foundation Models

Python framework for evaluating Foundation Models (FM).  

## Installation

- Discover node (e.g,. Discover14) is needed for outbound access and setup rather than GPU
- Requires 12G to install the FM container (can take over 40 minutes to download)
- Instructions assume Singularity as container host
- Usage leverages existing FM artifacts on Discover (e.g., checkpoints, statistics files)
- Tested using a100 GPUs with 60G memory

```bash
module load singularity
cd <install_dir>
git clone https://github.com/nasa-nccs-hpda/qefm-core.git
cd qefm-core/qefm/models
ln -sf /discover/nobackup/projects/QEFM/qefm-core/qefm/models/checkpoints .

cd <install_dir>   (e.g., cd ../../../..)
mkdir containers
cd containers
singularity pull docker://nasanccs/qefm-core:latest
```

## Documentation

- See User Guide to run QEFM inference with default data for multiple FMs: https://github.com/nasa-nccs-hpda/qefm-core/blob/main/README.md

- Shortcut to run Aurora FM with default inputs:

```bash
cd <install_dir/qefm-core>   
./tests/fm-inference.sh qefm-core_latest.sif aurora
```

## Sample Session
```bash
### Create installation directory and retrieve FM support code:

<user>@discover14:/home/gtamkin$ mkdir <install_dir>
<user>@discover14:/home/gtamkin$ cd <install_dir>
<user>@discover14:<install_dir>$ module load singularity
<user>@discover14:<install_dir>$ git clone https://github.com/nasa-nccs-hpda/qefm-core.git
Cloning into 'qefm-core'...
remote: Enumerating objects: 1213, done.
remote: Counting objects: 100% (327/327), done.
remote: Compressing objects: 100% (238/238), done.
remote: Total 1213 (delta 180), reused 133 (delta 65), pack-reused 886 (from 1)
Receiving objects: 100% (1213/1213), 48.61 MiB | 3.74 MiB/s, done.
Resolving deltas: 100% (392/392), done.
Updating files: 100% (470/470), done.
<user>@discover14:<install_dir>$ ls -alt
total 0
drwx------ 10 gtamkin ilab 230 Mar 12 04:10 qefm-core
drwx------  3 gtamkin ilab  23 Mar 12 04:09 .
drwx------ 19 gtamkin ilab 330 Mar 12 03:48 ..
<user>@discover14:<install_dir>$ ls qefm-core/
CONTRIBUTING.md  data  docs  INSTALL.md  LICENSE  notebooks  packages.txt  qefm  README.md  requirements  tests

### Link to QEFM artifacts:
<user>@discover14:<install_dir>$ cd qefm-core/qefm/models
<user>@discover14:<install_dir>/qefm-core/qefm/models$ ln -sf /discover/nobackup/projects/QEFM/qefm-core/qefm/models/checkpoints .
<user>@discover14:<install_dir>/qefm-core/qefm/models$ ls  checkpoints/aurora/
aurora-0.25-pretrained.ckpt

### Return to install_directory and create container dir:

<user>@discover14:<install_dir>/qefm-core/qefm/models/checkpoints$ cd ../../../..
<user>@discover14:<install_dir>$ mkdir containers
<user>@discover14:<install_dir>$ cd containers/

### Pull container:
<user>@discover14:<install_dir>$ time singularity pull docker://nasanccs/qefm-core:latest
INFO:    Converting OCI blobs to SIF format
WARNING: 'nodev' mount option set on /lscratch, it could be a source of failure during build process
INFO:    Starting build...
Getting image source signatures
Copying blob 857cc8cb19c0 done  
Copying blob 857cc8cb19c0 done  
.. . . . . . 
2025/03/12 04:00:02  info unpack layer: sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
2025/03/12 04:00:02  info unpack layer: sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
INFO:    Creating SIF file...

real	41m20.069s
user	53m35.945s
sys	2m31.113s

<user>@discover14:<install_dir>/containers$ ls
qefm-core_latest.sif
<user>@discover14:<install_dir>/containers$ du -sh qefm-core_latest.sif 
12G	qefm-core_latest.sif
<user>@discover14:<install_dir>/containers$ cd ../qefm-core

Test Aurora with container on GPU:
<user>@discover12:/home/gtamkin$ salloc --gres=gpu:1 --mem=60G --time=1:00:00 --partition=gpu_a100 --constraint=rome --ntasks-per-node=1 --cpus-per-task=10
<user>@warpa003:<install_dir>/qefm-core$ ./tests/fm-inference.sh  qefm-core_latest.sif aurora
Inference: /discover/nobackup/projects/QEFM/qefm-core/tests/fm-aurora.sh /discover/nobackup/projects/QEFM/qefm-core qefm-core-sandbox
Aurora: time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core/qefm /discover/nobackup/projects/QEFM/qefm-core/../containers/qefm-core-sandbox python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMAurora/predictions-for-ERA5.py
WARNING: underlay of /etc/localtime required more than 50 (99) bind mounts
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (533) bind mounts
Static variables downloaded!
Surface-level variables downloaded!
Atmospheric variables downloaded!
Preparing a Batch
Loading and Running the Model
model = Aurora(
  (encoder): Perceiver3DEncoder(
    (surf_mlp): MLP(
      (net): Sequential(
        (0): Linear(in_features=512, out_features=2048, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=2048, out_features=512, bias=True)
        (3): Dropout(p=0.0, inplace=False)
      )
    )
    (surf_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (pos_embed): Linear(in_features=512, out_features=512, bias=True)
    (scale_embed): Linear(in_features=512, out_features=512, bias=True)
    (lead_time_embed): Linear(in_features=512, out_features=512, bias=True)
    (absolute_time_embed): Linear(in_features=512, out_features=512, bias=True)
    (atmos_levels_embed): Linear(in_features=512, out_features=512, bias=True)
    (surf_token_embeds): LevelPatchEmbed(
      (weights): ParameterDict(
          (10u): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (10v): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (2t): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (lsm): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (msl): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (slt): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (z): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
      )
      (norm): Identity()
    )
    (atmos_token_embeds): LevelPatchEmbed(
      (weights): ParameterDict(
          (q): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (t): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (u): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (v): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
          (z): Parameter containing: [torch.FloatTensor of size 512x1x2x4x4]
      )
      (norm): Identity()
    )
    (level_agg): PerceiverResampler(
      (layers): ModuleList(
        (0): ModuleList(
          (0): PerceiverAttention(
            (to_q): Linear(in_features=512, out_features=512, bias=False)
            (to_kv): Linear(in_features=512, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=512, bias=False)
          )
          (1): MLP(
            (net): Sequential(
              (0): Linear(in_features=512, out_features=2048, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=2048, out_features=512, bias=True)
              (3): Dropout(p=0.0, inplace=False)
            )
          )
          (2-3): 2 x LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
  )
  (backbone): Swin3DTransformerBackbone(
    (time_mlp): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=True)
      (1): SiLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
    )
    (encoder_layers): ModuleList(
      (0): Basic3DEncoderLayer(
        (blocks): ModuleList(
          (0-5): 6 x Swin3DTransformerBlock(
            (norm1): AdaptiveLayerNorm(
              (ln): LayerNorm((512,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=1024, bias=True)
              )
            )
            (attn): WindowAttention(
              dim=512, window_size=(2, 6, 12), num_heads=8
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (drop_path): Identity()
            (norm2): AdaptiveLayerNorm(
              (ln): LayerNorm((512,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=1024, bias=True)
              )
            )
            (mlp): MLP(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging3D(
          (reduction): Linear(in_features=2048, out_features=1024, bias=False)
          (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): Basic3DEncoderLayer(
        (blocks): ModuleList(
          (0-9): 10 x Swin3DTransformerBlock(
            (norm1): AdaptiveLayerNorm(
              (ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=2048, bias=True)
              )
            )
            (attn): WindowAttention(
              dim=1024, window_size=(2, 6, 12), num_heads=16
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (drop_path): Identity()
            (norm2): AdaptiveLayerNorm(
              (ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=2048, bias=True)
              )
            )
            (mlp): MLP(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging3D(
          (reduction): Linear(in_features=4096, out_features=2048, bias=False)
          (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): Basic3DEncoderLayer(
        (blocks): ModuleList(
          (0-7): 8 x Swin3DTransformerBlock(
            (norm1): AdaptiveLayerNorm(
              (ln): LayerNorm((2048,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=4096, bias=True)
              )
            )
            (attn): WindowAttention(
              dim=2048, window_size=(2, 6, 12), num_heads=32
              (qkv): Linear(in_features=2048, out_features=6144, bias=True)
              (proj): Linear(in_features=2048, out_features=2048, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (drop_path): Identity()
            (norm2): AdaptiveLayerNorm(
              (ln): LayerNorm((2048,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=4096, bias=True)
              )
            )
            (mlp): MLP(
              (fc1): Linear(in_features=2048, out_features=8192, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=8192, out_features=2048, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (decoder_layers): ModuleList(
      (0): Basic3DDecoderLayer(
        (blocks): ModuleList(
          (0-7): 8 x Swin3DTransformerBlock(
            (norm1): AdaptiveLayerNorm(
              (ln): LayerNorm((2048,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=4096, bias=True)
              )
            )
            (attn): WindowAttention(
              dim=2048, window_size=(2, 6, 12), num_heads=32
              (qkv): Linear(in_features=2048, out_features=6144, bias=True)
              (proj): Linear(in_features=2048, out_features=2048, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (drop_path): Identity()
            (norm2): AdaptiveLayerNorm(
              (ln): LayerNorm((2048,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=4096, bias=True)
              )
            )
            (mlp): MLP(
              (fc1): Linear(in_features=2048, out_features=8192, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=8192, out_features=2048, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (upsample): PatchSplitting3D(
          (lin1): Linear(in_features=2048, out_features=4096, bias=False)
          (lin2): Linear(in_features=1024, out_features=1024, bias=False)
          (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): Basic3DDecoderLayer(
        (blocks): ModuleList(
          (0-9): 10 x Swin3DTransformerBlock(
            (norm1): AdaptiveLayerNorm(
              (ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=2048, bias=True)
              )
            )
            (attn): WindowAttention(
              dim=1024, window_size=(2, 6, 12), num_heads=16
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (drop_path): Identity()
            (norm2): AdaptiveLayerNorm(
              (ln): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=2048, bias=True)
              )
            )
            (mlp): MLP(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (upsample): PatchSplitting3D(
          (lin1): Linear(in_features=1024, out_features=2048, bias=False)
          (lin2): Linear(in_features=512, out_features=512, bias=False)
          (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): Basic3DDecoderLayer(
        (blocks): ModuleList(
          (0-5): 6 x Swin3DTransformerBlock(
            (norm1): AdaptiveLayerNorm(
              (ln): LayerNorm((512,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=1024, bias=True)
              )
            )
            (attn): WindowAttention(
              dim=512, window_size=(2, 6, 12), num_heads=8
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (drop_path): Identity()
            (norm2): AdaptiveLayerNorm(
              (ln): LayerNorm((512,), eps=1e-05, elementwise_affine=False)
              (ln_modulation): Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=1024, bias=True)
              )
            )
            (mlp): MLP(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
  )
  (decoder): Perceiver3DDecoder(
    (level_decoder): PerceiverResampler(
      (layers): ModuleList(
        (0): ModuleList(
          (0): PerceiverAttention(
            (to_q): Linear(in_features=1024, out_features=1024, bias=False)
            (to_kv): Linear(in_features=1024, out_features=2048, bias=False)
            (to_out): Linear(in_features=1024, out_features=1024, bias=False)
          )
          (1): MLP(
            (net): Sequential(
              (0): Linear(in_features=1024, out_features=2048, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=2048, out_features=1024, bias=True)
              (3): Dropout(p=0.0, inplace=False)
            )
          )
          (2-3): 2 x LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (surf_heads): ParameterDict(
        (10u): Object of type: Linear
        (10v): Object of type: Linear
        (2t): Object of type: Linear
        (msl): Object of type: Linear
      (10u): Linear(in_features=1024, out_features=16, bias=True)
      (10v): Linear(in_features=1024, out_features=16, bias=True)
      (2t): Linear(in_features=1024, out_features=16, bias=True)
      (msl): Linear(in_features=1024, out_features=16, bias=True)
    )
    (atmos_heads): ParameterDict(
        (q): Object of type: Linear
        (t): Object of type: Linear
        (u): Object of type: Linear
        (v): Object of type: Linear
        (z): Object of type: Linear
      (q): Linear(in_features=1024, out_features=16, bias=True)
      (t): Linear(in_features=1024, out_features=16, bias=True)
      (u): Linear(in_features=1024, out_features=16, bias=True)
      (v): Linear(in_features=1024, out_features=16, bias=True)
      (z): Linear(in_features=1024, out_features=16, bias=True)
    )
    (atmos_levels_embed): Linear(in_features=1024, out_features=1024, bias=True)
  )
)
Plot the results 
Matplotlib created a temporary cache directory at /tmp/matplotlib-v2ppo7_1 because the default path (/home/gtamkin/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
pred = 0 Batch(surf_vars={'2t': tensor([[[[247.7672, 247.8064, 247.7757,  ..., 247.8876, 247.8563, 247.8380],
           ...,
           [197472.5625, 197473.9844, 197478.0781,  ..., 197475.1094,
            197479.6875, 197486.8594],
           [197456.2188, 197463.7500, 197463.7188,  ..., 197464.8438,
            197465.7500, 197457.5469],
           [197442.5625, 197446.9375, 197435.3125,  ..., 197448.3125,
            197436.9688, 197439.7969]]]]])}, metadata=Metadata(lat=tensor([ 90.0000,  89.7500,  89.5000,  89.2500,  89.0000,  88.7500,  88.5000,
         88.2500,  88.0000,  87.7500,  87.5000,  87.2500,  87.0000,  86.7500,
           ...,
        -86.7500, -87.0000, -87.2500, -87.5000, -87.7500, -88.0000, -88.2500,
        -88.5000, -88.7500, -89.0000, -89.2500, -89.5000, -89.7500]), lon=tensor([0.0000e+00, 2.5000e-01, 5.0000e-01,  ..., 3.5925e+02, 3.5950e+02,
        3.5975e+02]), time=(datetime.datetime(2023, 1, 1, 18, 0),), atmos_levels=(1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50), rollout_step=2))
31.43user 6.19system 2:10.95elapsed 28%CPU (0avgtext+0avgdata 12282084maxresident)k
0inputs+96outputs (7major+705577minor)pagefaults 0swaps
```