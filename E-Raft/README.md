# E-RAFT: Dense Optical Flow from Event Cameras

<p align="center">
  <a href="https://youtu.be/dN8fl7-XfNw">
    <img src="http://rpg.ifi.uzh.ch/eraft/eraft_thumbnail_play.png" alt="E-RAFT" width="400"/>
  </a>
</p>

This is the code for the paper **E-RAFT: Dense Optical Flow from Event Cameras** by [Mathias Gehrig](https://magehrig.github.io/), Mario Millhäusler, [Daniel Gehrig](https://danielgehrig18.github.io/) and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html).

## Download

Download the network checkpoints and place them in the folder ```checkpoints/```:


[Checkpoint trained on DSEC](https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar)

[Checkpoint trained on MVSEC 20 Hz](https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/mvsec_20.tar)

[Checkpoint trained on MVSEC 45 Hz](https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/mvsec_45.tar)


## Installation
Please install [conda](https://www.anaconda.com/download).
Then, create new conda environment with python3.7 and all dependencies by running
```
conda env create --file environment.yml
```

## Datasets
### DSEC
The DSEC dataset for optical flow can be downloaded [here](https://dsec.ifi.uzh.ch/dsec-datasets/download/).
We prepared a script [download_dsec_test.py](download_dsec_test.py) for your convenience.
It downloads the dataset directly into the `OUTPUT_DIRECTORY` with the expected directory structure.
```python
download_dsec_test.py OUTPUT_DIRECTORY
```

## Experiments
### DSEC Dataset
For the evaluation of our method with warm-starting, execute the following command:
```
python3 main.py --path <path_to_dataset>
```
For the evaluation of our method **without** warm-starting, execute the following command:
```
python3 main.py --path <path_to_dataset> --type standard
```

### Arguments
```--path``` : Path where you stored the dataset

```--dataset``` : Which dataset to use: ([dsec]/mvsec)

```--type``` : Evaluation type ([warm_start]/standard)

```--frequency``` : Evaluation frequency of MVSEC dataset ([20]/45) Hz

```--visualize``` : Provide this argument s.t. DSEC results are visualized. MVSEC experiments are always visualized.

```--num_workers``` : How many sub-processes to use for data loading (default=0)
