export nnUNet_raw_data_base="/BMANet/DATASET/nnUNet_raw" 
export nnUNet_preprocessed="/BMANet/DATASET/nnUNet_preprocessed" 
export RESULTS_FOLDER="/BMANet/DATASET/nnUNet_trained_models"
export CUDA_VISIBLE_DEVICES=0


#Then first check the data format for errors and preprocess,
cd /BMANet/experiment_planning
python nnUNet_plan_and_preprocess.py -t 017 --verify_dataset_integrity  # 017 is your task name id

#train
cd /BMANet/run/
python run_training.py 3d_fullres Btcv12TrainerV2 Task017_pros 0 --npz

#predict
cd /BMANet/inference
python predict.py -i /BMANet/DATASET/nnUNet_raw/nnUNet_raw_data/Task017_pros/imagesTs/ -o /BMANet/DATASET/pos_models_out/outPros/ -f 0 -m /BMANet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task017_pros/Btcv12TrainerV2__nnUNetPlansv2.1/
