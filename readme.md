# NFR

This is the official implementation of the paper 'Neural Face Rigging for Animating and Retargeting Facial Meshes in the Wild'

## [Project Page](https://dafei-qin.github.io/NFR/) | [Paper](https://arxiv.org/abs/2305.08296)

Why try NFR?

NFR can transfer facial animations to any customized face mesh, even with different topology, without any labor for manual rigging or data capturing. For facial meshes obtained from any source, you can quickly retarget exising animations onto the mesh and see the results in real-time.

## Testing Setup

**This release is tested under Ubuntu 20.04, with a RTX 4090 GPU. Other GPU models with CUDA should be OK as well.** 

**The testing module utilizes [vedo](https://vedo.embl.es/) for interactive visualization. Thus a display is required.**

**Windows is currently not supported unless you manually install the pytorch3d package following [their official guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).**

1. Create an environment called NFR
```shell
conda create -n NFR python=3.9
conda activate NFR
```

2. Recommend mamba to accelerate the installation process
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

5. Download the preprocess data and the pretrained model here: [Google Drive](https://drive.google.com/file/d/1cXXeU3AtpoGEVz2mhlWTSG1dEbAtCmD1/view?usp=sharing). Place them in the root directory of this repo.

6. Run!

```shell
python test_user.py -c config/test.yml
```

7. Interactive visualization

Here's the plot when you successfully run the script. You can interact with the sliders and buttons to change the expression of the source mesh, and manually adjust the expression by FACS-like codes.
![](fig/vedo.jpg)

- Zone 0: The source mesh
- Zone 1: The target mesh (with source mesh's expression transferred)
- Zone 2: The source mesh under ICT Blendshape space
- Zone 3: Interactive buttons and sliders
    - Buttons:
        - code_idx: input (0-52) the FACS code index to the terminal
        - input/next/random: change the source expression index
        - iden: change the source identity
    - Sliders:
        - AU scale: Change the intensity of the FACS code specified by *code_idx*
        - scale: Scale uniformly the target mesh
        - x/y/z shift: Shift the target mesh

### Pre-processed facial animation sequences
Currently we have two pre-processed facial animation sequences, one from [ICT](https://github.com/ICT-VGL/ICT-FaceKit) and another from [Multiface](https://github.com/facebookresearch/multiface). You can swith between them by changing the `dataset` and `data_head` variables in the `config/test.yml` file. 

### Using your customized data

You can test with your own mesh as the target. This has two requirement:
1. There should be no mouth/eye/nose sockets and eye balls inside the face. Otherwise bad deformations may occur on those area. 
2. The mouth and eyes need to be cut for correct global solving. Please refer to the preprocessed meshes in the `test-mesh` folder as examples.
3. Remember to roughly align your mesh to the examples in blender via the align.blend file!

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

## Acknowledgement

This project uses code from [ICT](https://github.com/ICT-VGL/ICT-FaceKit), [Multiface](https://github.com/facebookresearch/multiface), [Diffusion-Net](https://github.com/nmwsharp/diffusion-net), data from ICT and Multiface, testing mesh templates from ICT, Multiface, [COMA](https://github.com/anuragranj/coma), [FLAME](https://flame.is.tue.mpg.de/), [MeshTalk](https://github.com/facebookresearch/meshtalk). Thank you!