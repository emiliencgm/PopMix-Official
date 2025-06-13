# Exploring Explicit Debiasing with Augmentation for Contrastive Collaborative Filtering

## Published as a conference paper at IJCNN2024
doi: 10.1109/IJCNN60899.2024.10651019

## Notes
- In order to reduce the file size, precalculated results are removed from it. They will be automatically recalculated before training for our method. Please be patient and wait.
- The file `run.py` provides a simple example of running the program. To run our proposed method:
    - First, enter the `code` directory.
    - Then enter the command `python run.py --dataset amazon-book --method PopMix --device 0 --visual 0 --temp_tau 0.09 --lambda1 0.1` to test our method on the Amazon-Book dataset. 
    - Experimental data will be uploaded to WandB for analysis. If you do not want to upload data to WandB, please uncomment the line `os.environ['WANDB_MODE'] = 'dryrun'` in the `main.py` file.
    - If you wish to perform some of the visualizations we demonstrated in the paper during this training process, please modify the parameter to `--visual 1`. 
    - If you have multiple GPUs, you can change the parameter `--device` to select the ID number of the GPU on your device.
- To run a baseline method:
    - First, enter the `baselines` directory.
    - Then enter the command `python run.py --dataset amazon-book --method BC --device 0 --visual 0 --temp_tau 0.09` to test the BC loss on the Amazon-Book dataset. 
