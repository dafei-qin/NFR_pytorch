# NFR

This is the official implementation of the paper 'Neural Face Rigging for Animating and Retargeting Facial Meshes in the Wild'

## [Project Page](https://dafei-qin.github.io/NFR/) | [Paper](https://arxiv.org/abs/2305.08296)

## Testing Setup

1. Create an environment called NFR
```shell
conda create -n NFR python=3.9
conda activate NFR
```

2. Recommand mamba to accelerate the installation process
```shell
conda install mamba -c conda-forge
```


3. Install necessary packages via mamba
```shell
mamba install pytorch=1.12.1 cudatoolkit=11.3 pytorch-sparse=0.6.15 pytorch3d=0.7.1 cupy=11.3 numpy=1.23.5 -c pytorch -c conda-forge -c pyg -c pytorch3d
```

4. Install necessary packages via pip
```shell
pip install potpourri3d trimesh open3d transforms3d libigl robust_laplacian vedo
```

5. Run!

```shell
python test_user.py -c config/test.yml
```

### Using your customized data

You can test your own mesh. Remeber to first align it using the align.blend!

## Training

The training module will be released later. 

## Citation


```
@inproceedings{qin2023NFR,
          author = {Qin, Dafei and Saito, Jun and Aigerman, Noam and Groueix Thibault and Komura, Taku},
          title = {Neural Face Rigging for Animating and Retargeting Facial Meshes in the Wild},
          year = {2023},
          booktitle = {SIGGRAPH 2023 Conference Papers},
      }
```