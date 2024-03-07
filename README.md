# BASE: Brain Age Standardized Evaluation

This is the repository for the paper:
**BASE: Brain Age Standardized Evaluation**, NeuroImage, vol. 285, p. 120469, Jan. 2024, 
doi: 10.1016/j.neuroimage.2023.120469. 

Avaliable at: [https://doi.org/10.1016/j.neuroimage.2023.120469](https://doi.org/10.1016/j.neuroimage.2023.120469)

### What is BASE? 
BASE is a standardized evaluation protocol for *deep learning brain age models* comprised of:
(i) a standardized T1w MRI dataset including multi-site, new unseen site, test-retest, and longitudinal datasets, 
along with (ii) an evaluation metrics and statistics.
The model evaluation  involves four tasks: (1) comparison of the performance of DL models and/or the comparative 
evaluation of the impact of model training strategies, (2) performance evaluation on seen/unseen dataset with possibly 
new preprocessing, (3) reproducibility on test-rest data and (4) consistency evaluation on longitudinal datasets,
as depicted in the figure below.

![BASE scheme](https://ars.els-cdn.com/content/image/1-s2.0-S1053811923006183-gr1_lrg.jpg "BASE scheme")


## Table of  contents

* [Prerequisites](#Prerequisites)
* [Models](#models)
   * [Model weights](#model-weights)
   * [Model inference](#model-inference)
* [Datasets](#datasets)
  * [Datasplit](#datasplit)
  * [Preprocesing](#prepocessing)
* [Model evaluation](#model-evaluation)
  * [Compute metrics](#compute-metrics)
  * [Statistical analysis](#statistical-analysis)
* [Radar plot](#radar-plot)
* [Citation](#citation)


## Prerequisites
To run this code, set the variable `project_root_path` to the root project folder in the `src/config.py` :
```
    project_root_path = '/path/to/project'
```
The following prerequisites are required:

* **Python 3.6 or higher**. 
To install the required libraries run the following command in the terminal:
```{bash}
    pip3 install -r requirements.txt
```
* **Docker** is required for utilising our preprocessing pipeline. The preprocessing pipeline will be made available shortly.
* **R 4.3.0** for running the statistical analysis, with the following packages: 
  * lme4 (version 1.1-34)
  * lmerTest (version 3.1-3)
  * dplyr (version 1.1.2)
  * emmeans (version 1.8.7)
  * ggradar (version 0.2)

To install the required R packages, run the following command in R:
```{R}
    install.packages("dplyr")
```

## Models

The four CNN architectures used in this research are:
* **Model 1** by [Cole et al. (2017)](https://www.sciencedirect.com/science/article/pii/S1053811917306407), 
Predicting brain age with deep learning from raw imaging data results in a reliable and heritable biomarker
* **Model 2** by [Huang at al. (2017)](https://ieeexplore.ieee.org/document/7950650), Age estimation from brain MRI images using deep learning
* **Model 3** by [Ueda at al. (2019)](https://ieeexplore.ieee.org/document/8759392), An age estimation method using 3D-CNN for brain MRI images
* **Model 4** by [Peng at al. (2021)](https://www.sciencedirect.com/science/article/pii/S1361841520302358), 
Accurate brain age prediction with lightweight deep neural networks
  * GitHub: we used the authors [implementation](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain) (accessed in 2020-03)

## Model training 
This repository containes the weights of the pre-trained models and the code for model inference. The training code used  
to train the models on the multi-site dataset is not included in this repository. However, all the hyperparameters, 
as well as the loss function, learning rate schedule, and the optimizer are available in the `./src/models`
directory, contained in each individual model's file.
The MONAI library can be used for random augmentations. 

### Model weights
The `./BASE_models` directory contains 20 sets of pre-trained model weights (4 ✕ architecture, 5 ✕ random weight initialization),
pretrained on multi-site dataset, where each file name consists of the following information:
```
    model{model_number}-{autor}-seed_{random_seed}-{number_of_epochs}ep.tar
```
where:
* `model_number` is the number of the model (1, 2, 3, or 4),
* `autor` is the name of the first author of the paper,
* `random_seed` is the random seed used for the model weight initialization,
* `number_of_epochs` is the number of epochs used for training. 

### Model inference

Each of the four models can be used for inference, as demonstrated in the `./src/models` files. The input should be 
a 3D T1w MR image, and the output is the predicted brain age. The inference code is available in the following files:
* `./src/models/model1_cole.py` for Model 1
* `./src/models/model2_huang.py` for Model 2
* `./src/models/model3_ueda.py` for Model 3
* `./src/models/model4_peng.py` for Model 4

## Datasets

This research uses 9 public datasets for a wide evaluation. The datasets descirbed in Table 1 of the paper.
The list of the datasets used in this research is as follows:
* **Multisite dataset**:
  * [ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
  * [ADNI](https://adni.loni.usc.edu/)
  * [CC-359](https://sites.google.com/view/calgary-campinas-dataset/download)
  * [Cam-CAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
  * [FCON1000](https://fcon_1000.projects.nitrc.org/indi/enhanced/neurodata.html)
  * [IXI](https://brain-development.org/ixi-dataset/)
  * [OASIS-2](https://www.oasis-brains.org/)
  
* **New site** (+ new preprocessing):
  * [UK Biobank](https://www.ukbiobank.ac.uk/)
  
* **Test-retest dataset**:
  * [OASIS-1](https://www.oasis-brains.org/)
  
* **Longitudinal dataset**:
  * [UK Biobank](https://www.ukbiobank.ac.uk/)

### Datasplit 

To assure the reproducibility of the results, the exact IDs of each datasplit are available in the `./src/datasets_split` 
directory. In the process of quality control, the IDs of some subjects were removed from the datasets. 

The multi-site dataset (`multisite_datasplit.csv`) dataset was split into training, validation, and test sets. The rest 
of the datasets were used only for testing. 

### Preprocesing

<span style="color:blue"> NOTE: a docker container with the preprocessing pipeline 
will be made available shortly. </span>

## Analysis
<span style="color:red"> NOTE: The csv files in ./predictions directory contain **GENERATED** data. Researchers should obtain the datasets 
from the original sources in accordance with the license agreement.</span>

### Compute metrics

For each of four evaluation tasks, the metrics are computed and saved in the `./src/analysis` directory:
* **Task 1**: Comparison of the performance of DL models and/or the comparative evaluation of the impact of model training strategies
  * `./src/analysis/01_multisite_dataset.py`
* **Task 2**: Performance evaluation on seen/unseen dataset with possibly new preprocessing
  * `./src/analysis/02_unseens_site_dataset.py`
* **Task 3**: Reproducibility on test-rest data
  * `./src/analysis/03_test_retest.py`
  * `./src/analysis/03_test-retest_ICC.html` (R code for ICC computation)
* **Task 4**: Consistency evaluation on longitudinal datasets
  * `./src/analysis/04_longitudinal_dataset.py` 


### Statistical analysis
The statistical analysis is performed using linear mixed-effects models (LMMs) in R. The R code is available in the file
```
./src/analysis/LMM_analysis.html
``` 

## Radar plot
The principal results of BASE are visualized in the form of a **radar plot**. Values closer 
to the plot’s center indicate better performance, therefore a tighter envelope indicates a better overall performance. 
The radar plot was generated using R and the `ggradar` package. The R code is available in 
```
    ./src/analysis/radar_plot.html
```

![BASE radar ](https://ars.els-cdn.com/content/image/1-s2.0-S1053811923006183-gr2_lrg.jpg "BASE output")

## Citation
When using our evaluation protocol, please cite the following paper:

**L. Dular and Ž. Špiclin, “BASE: Brain Age Standardized Evaluation,” NeuroImage, vol. 285, p. 120469, Jan. 2024, doi: 10.1016/j.neuroimage.2023.120469.**

For BibTeX:
```

@article{dular_base_2024,
	title = {{BASE}: {Brain} {Age} {Standardized} {Evaluation}},
	volume = {285},
	issn = {1053-8119},
	shorttitle = {{BASE}},
	url = {https://www.sciencedirect.com/science/article/pii/S1053811923006183},
	doi = {10.1016/j.neuroimage.2023.120469},
	urldate = {2024-02-26},
	journal = {NeuroImage},
	author = {Dular, Lara and Špiclin, Žiga},
	month = jan,
	year = {2024},
	keywords = {Brain age, Reproducibility, Accuracy, UK biobank, Robustness, Consistency, Deep regression, Evaluation},
	pages = {120469},
}
```
