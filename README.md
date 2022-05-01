#  Jigsaw puzzles - Deep Learning
## Authors: Lorenzo Mandelli
#### Università degli Studi di Firenze | UiT The Arctic University of Norway 

---
![](https://img.shields.io/github/contributors/divanoletto/Jigsaw_puzzles-Deep_Learning?color=light%20green) ![](https://img.shields.io/github/repo-size/divanoletto/Jigsaw_puzzles-Deep_Learning)
---

1 | 2 | 3 | 4 | 5 | 6 | 7| 8
:-------------------------:|:-------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:
![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/0.png) | ![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/1.png)  |  ![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/2.png)  |  ![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/3.png)  |  ![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/4.png)  |  ![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/5.png)  |  ![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/6.png)  |  ![](https://github.com/divanoLetto/Jigsaw_puzzles-Deep_Learning/blob/main/images/7.png)


## Introduction

In this problem we solve the Jigsaw Puzzles problem of ordering shuffled patches of an image. <br/>
A nerural network has been created modifying a VGG11 which, given the shuffled images, learns to recognize the permutation and allows to restore the initial image. The problem so has been considered as a classification problem in which it is necessary to distinguish in which of the possible permutations the image has been perturbed. <br\> 
To be able to reach a high level of accuracy starting from the data set provided, an initialization phase of the weights and a phase of data augmentation were carried out.<br\>
Read Report/report.pdf for all the details.

## Usage

In order to use run the code run the following istruction:

```
$ python 3_b.py 
```

The file *settings/config.cfg* contains the project settings as batch size and dataset path.
The following code allows to test the trained network against a never seen dataset by a fine-tuning operations into 9 classes. 

```
$ python 3_c.py 
```
