# Welcome to RadiViz Toolkit

RadiViz Toolkit is a Python program designed to generate radiomics feature maps from medical images, such as CT, PET, MRI, US, X-ray. The RadiViz supports users to save them locally in different formats, such as EPS, PDF, PNG and so on. This toolkit leverages the power of radiomics to extract valuable features from medical images, aiding in various research and clinical applications.

<!-- ## Table of Contents

- [Features](#Features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license) -->

## Features

- Generate radiomics feature maps from medical images.
- Support for multiple image types (e.g., Original, Wavelet, LoG, Square and so on).
- Configurable feature extraction parameters.
- Save generated feature maps locally in a specified directory with various figure formats.

## Quick Start (1 mins)

```
conda create --name radiviz python==3.9
conda activate radiviz
```
Then install related dependencies.

```
pip install -r requirements.txt
```
If the installation runs slowly, you can try the following method

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## RadiViz Feature visualization example

### Patient-level feature visualization

<p align="center">
  <img src="https://github.com/zhenweishi/FM-LCT/assets/17007301/610b8ab1-3495-47f5-87c9-e9e741716bce" width="450" height="430">
  <img src="https://github.com/zhenweishi/FM-LCT/assets/17007301/9cdf2dfd-4dfc-452a-884e-b9143c3e0fcd" width="450" height="430">
</p>

### Feature-level visualization (examples)


![RadiViz_feature](https://github.com/zhenweishi/FM-LCT/assets/17007301/9648b31a-7618-4d81-a567-a5ca5e5fd0e3)


## License

This project is freely available to browse, download, and use for scientific and educational purposes as outlined in the [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/).

## Disclaimer

FM-BCMRI is still under development. Although we have tested and evaluated the workflow under many different situations, it may have errors and bugs unfortunately. Please use it cautiously. If you find any, please contact us and we would fix them ASAP.

## Main Developers
 - [Dr. Zhenwei Shi](https://github.com/zhenweishi) <sup/>1, 2
 - MSc. Zhitao Wei <sup/>2, 3
 - MD. Zaiyi Liu <sup/>1, 2
 

<sup>1</sup> Department of Radiology, Guangdong Provincial People's Hospital (Guangdong Academy of Medical Sciences), Southern Medical University, China <br/>
<sup>2</sup> Guangdong Provincial Key Laboratory of Artificial Intelligence in Medical Image Analysis and Application, China <br/>
<sup>3</sup> Institute of Computing Science and Technology, Guangzhou University, China <br/>

## Contact
We are happy to help you with any questions. Please contact Dr Zhenwei Shi.

We welcome contributions to RadiViz Toolkit. 
