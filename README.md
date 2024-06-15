
# Oracle-Bone-Script-Recognition: Step by Step Demo

==============================

## 1. Project Background

We referenced two projects: one [(Deep-Learning-for-Oracle-Bone-Script-Recognition)](https://github.com/cuicaihao/Deep-Learning-for-Oracle-Bone-Script-Recognition/tree/master) used Qt to create the GUI but the model is not very accuracte, while the other one [(HUST-OBC)](https://github.com/Pengjie-W/HUST-OBC/tree/main?tab=readme-ov-file#validation) used ResNet50 and has a high reconition rate. We promoted the interface of the first one, and optimized and amended the second one to use the Validation part. As a result, we got an intelligent and  accurate model. Once we write an OBC (oracle bones character), it shows us what's the Chinese the OBC represents. (As the picture shows)
![APP SAMPLE IMAGE](figures/zhengchang.png)
Besides, out model is very sensitive that even we write abstractly, which makes it hard to be identified, it can still show us the answer.   
![APP SAMPLE IMAGE](figures/chou.png)

## 2. Starting Requirements

- conda 4.12.0
- Python 3.7, 3.8
- We suggest using [Anaconda](https://www.anaconda.com/) for the installation of Python. Or you can just install the [miniconda](https://docs.conda.io/en/latest/miniconda.html) package which save a lot of space on your hard drive.

## 3. Tutorial Step by Step

### Step 1: Download the project

Download the project using git clone command.

```bash
cd PROJECT_DIR
git clone 
https://github.com/zhz5687/Transformer-OBS-Recognition
```
There's another part of the program that't too big to put here, you can download [here](https://pan.baidu.com/s/1XlUjOg7S51yKuPkrKkOpKw?pwd=1234).

### Step 2: Create Environment 

We prepared the environment required for you. You can downoad it [here]( https://pan.baidu.com/s/1N2FAqHrLoz_Ol8q7ZtkGkQ?pwd=1234)
and import it to Anaconda.
When you run the program, you need to run it in this environment.
```bash
conda activate Transformer
```

### Step 3: Download the Dataset

We use the [HUST](https://arxiv.org/html/2401.15365v1) dataset, which can be download [here](https://figshare.com/articles/dataset/HUST-OBS/25040543). After downloading it, move to the project file you download from github.


### Step 4: Train or Test your OBS model
You can use [train.py](train.py) for fine-tuning or retraining. Once the model is downloaded, you can use [test.py](test.py) to validate the test set with an accuracy of 94.3%. [log.csv](log.csv) records the changes in training set accuracy and test set accuracy for each epoch. 
[Validation_label.json](Validation_label.json) stores the relationship between classification IDs and dataset category IDs.

### Step 5: Try to run the program

Now you can try to run the program using command.
```bash
python gui.py
```

## 4 Introduction of the interface

![APP SAMPLE IMAGE](figures/4.png)

Once you load this interface, you can just write the OBC that you want to identify on the "输入" part, and click the "运行" button. Then the interface will show like that:

![APP SAMPLE IMAGE](figures/3.png)

The prediction ID means the ID in the model. If the accuracy shows 1.000, it means the program is very sure about the outcome. On the contrast, while it is 0.0000, it means it can't make sure at all. There are also several other possibilities beneath it, and the numbers after these characters are not their ID, but their number in the dataset. And you want to identify another OBC, you can click the "清空" botton, and write another OBC.

## 5 Summary
This repository is inspired by the most recent DeepMind's work [Predicting the past with Ithaca](https://www.deepmind.com/blog/predicting-the-past-with-ithaca), I did not dig into the details of the work due to limited resources.

I think the work is very interesting and I want to share my experience with the readers by trying a different language like Oracle Bone Scripts. It is also a good starter example for myself to revisit the pytorch deep learning packages and the qt-gui toolboxes.

I will be very grateful if you can share your experience with more readers. If you like this repository, please upvote/star it.

If you find the repository useful, please consider donating to the Standford Rural Area Education Program (<https://sccei.fsi.stanford.edu/reap/>): Policy change and research to help China's invisible poor.

Thank you.

## License

The MIT License (MIT), Copyright (c) 2022, Caihao Cui

## Planning Tasks

- [ ] Enhance the data sample with image transform operations to make the model robust to various input, e.g., rotate image with [-15,15] degrees.
- [ ] Upgrade the ConvNet model and find better solutions (model structure, loss function, etc).
- [ ] Upgrade the Chinese-to-English translator, maybe built a Transformer model to translate sentence from chinese to english; Feel free to check my early work on blog [the-annotated-transformer-english-to-chinese-translator](https://cuicaihao.com/the-annotated-transformer-english-to-chinese-translator/).
- [ ] Upgrade the Input drawing,  adding features of OCR and Character Detection function allow use scanning the large image.

## Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

### Reference

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Qt for Python](https://wiki.qt.io/Qt_for_Python)
- [Chinese-Traditional-Culture/JiaGuWen](https://github.com/Chinese-Traditional-Culture/JiaGuWen)
- [Website of the Oracle Bone Script Index](https://chinese-traditional-culture.github.io/JiaGuWen/)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
