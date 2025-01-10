
#python ../run/run_training1.py 3d_fullres nnUNetTrainerV2 Task022_Synapse 0 --npz

python ../inference/predict.py -i /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_raw/nnUNet_raw_data/Task022_Synapse/imagesTs/ -o /remote-home/hhhhh/Seg/Pos/pos_models_out/output22/  -f 0  -m /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task022_Synapse/nnUNetTrainerV2__nnUNetPlansv2.1/

python ../inference/calculate_Synapse22.py
