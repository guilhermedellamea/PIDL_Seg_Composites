<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
<!--
  <a href="https://github.com/guilhermedellamea/PIDL_Seg_Composites">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>-->

<h3 align="center">Physics Informed Self-Supervised Segmentation of Composite Materials</h3>

  <p align="center">
    This work presents the application of a Physics Informed Deep Learning models for both surrogate modelling and segmentation of composite materials.  
    <br />
    <a href="https://github.com/guilhermedellamea/PIDL_Seg_Composites"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/guilhermedellamea/PIDL_Seg_Composites/issues">Report Bug</a>
    ·
    <a href="https://github.com/guilhermedellamea/PIDL_Seg_Composites/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#citation">Citation</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


### Citation

Our paper is available on [SSRN](https://ssrn.com/abstract=4807639). Please cite our work with
```sh
@misc{basso_della_mea_physics_2024,
	title = {Physics {Informed} {Self}-{Supervised} {Segmentation} of {Composite} {Materials}},
	author = {Basso Della Mea, Guilherme and Ovalle, Cristian and Laiarinandrasana, Lucien and Decencière, Etienne and Dokladal, Petr},
	month = apr,
	year = {2024}}
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
Our released implementation is tested on:
* Ubuntu 20.04
* Python 3.10.3
* PyTorch 2.3.1
* NVIDIA CUDA 12.1


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Clone our project folder
* Create and lunch your conda environment with
  ```sh
  conda create -n PIDL_Seg_Composites python=3.10.3
  conda activate PIDL_Seg_Composites
  ```
<!--### Installation-->
* Install dependencies
    ```sh
  pip install -r requirements.txt
  ```
  Note: for Pytorch CUDA installation follow https://pytorch.org/get-started/locally/.
  
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Dataset
The dataset used in our experiments can be downloaded [here]() and should be put into the data folder.

### Training
To train a model you can use the `train.py` script provided. In this implementation the momentum conservation, segmentation and $\mathcal{L}_\phi$ training phases are split to accelerate the training.

### Testing 
The trained models can be tested with the `test.py` script.


<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/my_feature`)
3. Commit your Changes (`git commit -m 'Add my_feature'`)
4. Push to the Branch (`git push origin feature/my_feature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT (or other) License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Guilherme Della Mea - guilherme.basso_della_mea@minesparis.psl.eu

Project Link: [https://github.com/guilhermedellamea/PIDL_Seg_Composites](https://github.com/guilhermedellamea/PIDL_Seg_Composites)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 945304 – Cofund AI4theSciences hosted by PSL University.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/guilhermedellamea/PIDL_Seg_Composites.svg?style=for-the-badge
[contributors-url]: https://github.com/guilhermedellamea/PIDL_Seg_Composites/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/guilhermedellamea/PIDL_Seg_Composites.svg?style=for-the-badge
[forks-url]: https://github.com/guilhermedellamea/PIDL_Seg_Composites/network/members
[stars-shield]: https://img.shields.io/github/stars/guilhermedellamea/PIDL_Seg_Composites.svg?style=for-the-badge
[stars-url]: https://github.com/guilhermedellamea/PIDL_Seg_Composites/stargazers
[issues-shield]: https://img.shields.io/github/issues/guilhermedellamea/PIDL_Seg_Composites.svg?style=for-the-badge
[issues-url]: https://github.com/guilhermedellamea/PIDL_Seg_Composites/issues
[license-shield]: https://img.shields.io/github/license/guilhermedellamea/PIDL_Seg_Composites.svg?style=for-the-badge
[license-url]: https://github.com/guilhermedellamea/PIDL_Seg_Composites/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: www.linkedin.com/in/guilherme-basso-della-mea-837b4b147