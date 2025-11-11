
# SynTeX-FL: Cross-Modal Text Transfer in Federated Learning for Medical Visual Question Answering


## Environment

Install Package: Create conda environment

```bash
conda env create --file environment.yaml
```

To activate the conda environment, run the command below.
```bash
conda activate syntex
```

Clients 1 and 2 consist of the X-ray dataset.
Clients 3 and 4 consist of the X-ray dataset.
Clients 5 and 6 consist of the X-ray dataset.

If you want to adjust the number of clients, update the client variable in both the config.py and train.py files, and make sure to set the M_out and C_out variables in train.py to match the desired number of MRI and CT clients, respectively.


### Prepare data
> NOTE: to access the `LLaVA-Med` dataset family, appropriate credentials are required.
1. LLaVA-Med dataset : Download the entire LLaVA-Med from the [LLaVA-Med](https://github.com/microsoft/LLaVA-Med?tab=readme-ov-file). After downloading, use the llava_med_instruct_fig_captions.json file to filter the images based on the domain key. Select only the images where the domain value is set to 'true' for either MRI or CT. Next, separate the filtered images into training and validation sets for each modality.

1. For each subset (train/val) and modality (CT/MRI), generate a CSV file using the pair_id and fig_caption fields from llava_med_instruct_fig_captions.json. Save these files in the corresponding text folders under train and val directories:

Your final directory structure should look like this:
```
data/
├── train/
│   ├── xray/
│   │   ├── 1477034_f2-kjim-7-1-68-12.jpg
│   │   ├── 1666957_f1-kjim-6-2-79-5.jpg
│   │   ├── ...
│   │   ├── 1742253_f5-kjim-6-1-27-5.jpg
│   ├── mri/
│   │   ├── 1807369_f2-kjim-6-2-85-6.jpg
│   │   ├── 9876666_F2.jpg
│   │   ├── ...
│   │   ├── 10917126_F1.jpg
│   ├── mri/
│   │   ├── 8038146_f1-kjim-9-1-43-8.jpg
│   │   ├── 10444016_F2.jpg
│   │   ├── ...
│   │   ├── 11752922_F3.jpg
│   ├── text/
│   │   ├── pmc-15m-xray-train.csv
│   │   ├── pmc-15m-ct-train.csv
│   │   ├── pmc-15m-mri-train.csv
├── val/
│   ├── ct/
│   │   ├── 2486849_f1-kjim-4-2-174-13.jpg
│   │   ├── 20167080_F2.jpg
│   │   ├── ...
│   │   ├── 20191022_F1.jpg
│   ├── ct/
│   │   ├── 9159044_f1-kjim-12-1-80-15.jpg
│   │   ├── 9876756_F1.jpg
│   │   ├── ...
│   │   ├── 11752955_F15.jpg
│   ├── mri/
│   │   ├── 17427647_F3.jpg
│   │   ├── 17951214_figure3.jpg
│   │   ├── ...
│   │   ├── 18306482_F1.jpg
│   ├── text/
│   │   ├── pmc-15m-xray-val.csv
│   │   ├── pmc-15m-ct-val.csv
│   │   ├── pmc-15m-mri-val.csv
```


### Train the model
Run the python script below:
```bash
python train.py
```
Before running, check the config.py variable `TRAIN_DIR`, `VAL_DIR`, 


### Validation the model
After model training, run the validation script below:"
```bash
python valid.py
```