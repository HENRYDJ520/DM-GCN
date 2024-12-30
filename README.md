# DM-GCN

This is the code repository for the paper “Dynamic Multi-stream Graph Neural Networks for Efficient Interactive Action Recognition”.

# Prerequisites

- Python == 3.8
- PyTorch == 1.12.1
- CUDA == 11.6

# Hardware Configuration

Our hardware configuration for this experiment comprises an Intel Core i9-12900K processor, featuring 24 cores and a base clock speed of 3.9 GHz, along with 64GB of DDR4 RAM and two NVIDIA 3090 GPUs.

# Data Preparation

### Download datasets.

#### There are 1 datasets to download:

- NTU RGB+D 120 Skeleton

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   2. Extract above files to `./data/nturgbd_raw`
   
### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 120 dataset:

```
 cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing

### Training

- Change the config file depending on what you want.
- twograph is the PI_GCN
- Hypergnn is the DIH_GCN
- alltransformer is the GI_GCN

```
# Example: training on NTU RGB+D 120 XSub in 3 streams
python main.py --config config_in/alltransformer.yaml
python main.py --config config_in/Hypergnn.yaml
python main.py --config config_in/twograph.yaml
```

```
# Example: training on NTU RGB+D 120 XSet in 3 streams
python main.py --config config_in/alltransformer_view.yaml
python main.py --config config_in/Hypergnn_view.yaml
python main.py --config config_in/twograph_view.yaml
```

### Testing

- Fusion of 3 streams using the pso algorithm
- Copy the file named “epoch70_test_score.pkl” generated from the training of the three streams to the “. /two_person_score/” directory, and change the file name to ‘score.pkl’.

```
# Example: Streams fused on a subset of NTU-RGB+D 120 interaction dataset on XSub
python enu_two_graph.py --dataset 120/xsub --joint_dir two_person_score/hypergnn --bone_dir two_person_score/two_graph --joint_motion_dir two_person_score/alltransformer
```


