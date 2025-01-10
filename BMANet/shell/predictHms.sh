
#python ./bmanet/run/run_training.py 3d_fullres nnUNetTrainerV2 Task021_Synapse 0 --npz

# BTCV20
python ../inference/predict.py -i /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_raw/nnUNet_raw_data/Task020_BTCV/imagesTs/ -o /remote-home/hhhhh/Seg/Pos/pos_models_out/output20/  -f 0  -m /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task020_BTCV/BctvTrainerV2__nnUNetPlansv2.1/
python ../inference/calculate_BTCV.py

## Synapse21
#python ../inference/predict.py -i /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_raw/nnUNet_raw_data/Task021_Synapse/imagesTs/ -o /remote-home/hhhhh/Seg/Pos/pos_models_out/output21/  -f 0  -m /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task021_Synapse/BctvTrainerV2__nnUNetPlansv2.1/
#python ../inference/calculate_Synapse.py
#
## Synapse22
#python ../inference/predict.py -i /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_raw/nnUNet_raw_data/Task022_Synapse/imagesTs/ -o /remote-home/hhhhh/Seg/Pos/pos_models_out/output22/  -f 0  -m /remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task022_Synapse/BctvTrainerV2__nnUNetPlansv2.1/
#python ../inference/calculate_Synapse22.py

