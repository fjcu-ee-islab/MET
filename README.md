This repository contains the official code from __Multi-modal event transformer__. 


### Repository requirements

Step 1. First, install Anaconda, then create an environment using the command line with the following commands:
```
conda create --name evt python=3.7.10
conda activate evt
pip install -r requirements.txt
```

Step 2. After creating the environment, activate it using:
```
conda activate evt
```

### Dataset prepare
The datasets must be downloaded from their source and place it in the `./datasets` directory:
 - DVS128: https://research.ibm.com/interactive/dvsgesture/
 - SL-Animals-DVS: http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database


### Event-based motion feature extraction

Visual motion estimation: To execute parallel_main.m in Matlab, use the following command:
```
run('parallel_main.m')
```

Optical flow estimation (E-Raft):

Step 1. Install Anaconda, then create an environment using the command line with the following command:
```
conda env create --file environment.yml
```

Step 2. Activate the environment created in the previous step with:
```
conda activate eRaft
```

Step 3. Use `aedat_to_h5.py` in each dataset to create the necessary `event.h5`, `image_timestamps.txt`, and `test_forward_flow_timestamps.csv` files for E-Raft.

Step 4. Run the eRaft main program using the command:
```
python3 main.py --path data/DVS_Gesture -v
```


### Dataset preprocessing
Preprocess each dataset using the scripts in the `./dataset_script` directory.

If the dataset is [DvsGesture], [DvsGesture_VM], or [DvsGesture_OF], preprocessing involves two steps:

[DvsGesture]:
```
python dvs128_split_dataset.py
python dvs128.py 
```

[DvsGesture_VM]:
```
python dvs128_VM_split_dataset.py
python dvs128_VM.py
```
[DvsGesture_OF]:
```
python dvs128_OF_split_dataset.py
python dvs128_OF.py
```

For datasets such as [SL_Animals], [SL_Animals_VM], [SL_Animals_OF], [UCF11], [UCF11_VM], [UCF11_OF], [IITM], [IITM_VM], [IITM_OF], [Fall detection dataset], [FDD_VM], [FDD_OF], only the corresponding preprocessing script for each dataset needs to be executed.


### MMET training

Open train.py and adjust the corresponding pretrain model and weights based on the dataset being used. 
After verifying everything is correct, you can execute MMET Training with the command:
```
python train.py
```


### MMET evaluating

Use the command to perform testing on the trained models in the ./pretrained_models/tests/ directory. 
Evaluation results include FLOPs, parameters, average activated patches, average processing time, and validation accuracy.
```
python evalution_states.py
```

### Reference
```bibtex
@article{orchard2014bioinspired,
  title={Bioinspired visual motion estimation},
  author={Orchard, Garrick and Etienne-Cummings, Ralph},
  journal={Proceedings of the IEEE},
  volume={102},
  number={10},
  pages={1520--1536},
  year={2014},
  publisher={IEEE}
}

@inproceedings{orchard2013spiking,
  title={A spiking neural network architecture for visual motion estimation},
  author={Orchard, Garrick and Benosman, Ryad and Etienne-Cummings, Ralph and Thakor, Nitish V},
  booktitle={2013 IEEE Biomedical Circuits and Systems Conference (BioCAS)},
  pages={298--301},
  year={2013},
  organization={IEEE}
}

@InProceedings{Gehrig3dv2021,
  author = {Mathias Gehrig and Mario Millh\"ausler and Daniel Gehrig and Davide Scaramuzza},
  title = {E-RAFT: Dense Optical Flow from Event Cameras},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2021}
}
```
