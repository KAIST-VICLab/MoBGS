<div><h2>[AAAI'26] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video</h2></div>
<br>

**[Minh-Quan Viet Bui](https://quan5609.github.io/)<sup>\*</sup>, [Jongmin Park](https://sites.google.com/view/jongmin-park)<sup>\*</sup>, [Juan Luis Gonzalez Bello](https://sites.google.com/view/juan-luis-gb/home)<sup></sup>, [Jaeho Moon](https://sites.google.com/view/jaehomoon)<sup></sup>, [Jihyong Oh](https://cmlab.cau.ac.kr/)<sup>†</sup>, [Munchurl Kim](https://www.viclab.kaist.ac.kr/)<sup>†</sup>** 
<br>
<br>
\*Co-first authors (equal contribution), †Co-corresponding authors
<p align="center">
        <a href="https://kaist-viclab.github.io/mobgs-site/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/pdf/2504.15122" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2504.15122-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/MoBGS">
</p>

<p align="center" width="100%">
    <img src="https://github.com/KAIST-VICLab/MoBGS/blob/main/assets/architecture.png?raw=tru"> 
</p>


## ⚙️ Environmental Setups
Please refer to the environment installation of [SplineGS](https://github.com/KAIST-VICLab/SplineGS.git), and install gsplat, simple-knn as
```sh
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
pip install -e submodules/simple-knn
```
## 📁 Data Preparations
### We follow the evaluation setup from [DyBluRF](https://github.com/huiqiang-sun/DyBluRF). Download our preprocessed dataset [here](https://github.com/KAIST-VICLab/MoBGS/releases/tag/dataset) and arrange them as follows:
```bash
MoBGS/data/stereo
    ├── basketball
    │   
    │   
    │   
    │   
    ├── ...
    └── street
```
## 🚀 Get Started
### Training
```sh
python train.py -s data/stereo/seesaw/dense/ --port 6969 --expname "seesaw" --configs arguments/stereo/seesaw.py
```
#### Metrics Evaluation
```sh
python eval.py -s data/stereo/seesaw/dense/ --port 6018 --expname "seesaw" --configs arguments/stereo/seesaw.py --checkpoint output/seesaw/point_cloud/iteration_10000

python metrics.py --datadir data/stereo/seesaw/dense/ --scene_name seesaw --output_dir output
```

## Acknowledgments
- This work was supported by Institute of Information and communications Technology Planning and Evaluation (IITP) grant funded by the Korean Government [Ministry of Science and ICT (Information and Communications Technology)] (Project Number: RS-2022-00144444, Project Title: Deep Learning Based Visual Representational Learning and Rendering of Static and Dynamic Scenes, 100%).

## ⭐ Citing MoBGS

If you find our repository useful, please consider giving it a star ⭐ and citing our research papers in your work:
```bibtex
@InProceedings{bui2025mobgs,
    author    = {Bui, Minh-Quan Viet and Park, Jongmin and Bello, Juan Luis Gonzalez and Moon, Jaeho and Oh, Jihyong and Kim, Munchurl},
    title     = {MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video},
    booktitle = {AAAI},
    year      = {2026},
}
```


## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=KAIST-VICLab/MoBGS&type=Date)](https://www.star-history.com/#KAIST-VICLab/MoBGS&Date)
