#!/bin/bash

export TOKENIZERS_PARALLELISM=false

GEN_SCRIPT_PATH=generate.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=data/MileBench
MODEL_CONFIG_PATH=configs/model_configs.yaml

gpu_num=4

for model in llava-v1.5-7b; do
    for dataset_name in action_localization action_prediction action_sequence object_existence object_interaction moving_attribute object_shuffle egocentric_navigation moving_direction counterfactual_inference state_change character_order scene_transition ALFRED CLEVR-Change DocVQA IEdit MMCoQA MultiModalQA nuscenes OCR-VQA SlideVQA Spot-the-Diff TQA WebQA WikiVQA GPR1200 TextNeedleInAHaystack ImageNeedleInAHaystack; do

        # Set batch size: max(int(batch_image/n_img),1)
        if [ ${dataset_name} = "MMCoQA" ] || [ ${dataset_name} = "NeedleInAHaystack" ] || [ ${dataset_name} = "GPR1200" ]
        then
            BATCH_SIZE=1
        else
            BATCH_SIZE=24
        fi

        mkdir -p logs/${model}

        # Start generating
        accelerate launch --config_file ./configs/accelerate_configs.yaml \
            --main_process_port 29500  \
            --num_machines 1 \
            --machine_rank 0 \
            --num_processes ${gpu_num} \
            --deepspeed_multinode_launcher standard \
            \
            ${GEN_SCRIPT_PATH} \
            --data_dir ${DATA_DIR} \
            --dataset_name ${dataset_name}  \
            --model_name ${model} \
            --output_dir outputs \
            --batch-image ${BATCH_SIZE} \
            --model_configs ${MODEL_CONFIG_PATH} \
            --overwrite \
            > logs/${model}/${dataset_name}.log

        # Start evaluating
        python ${EVAL_SCRIPT_PATH} \
            --data-dir ${DATA_DIR} \
            --dataset ${dataset_name} \
            --result-dir outputs/${model} \
            >> logs/${model}/${dataset_name}.log

        # ############################## Combined to 1 image ###########################
        # # Start generating
        # accelerate launch --config_file ./configs/accelerate_configs.yaml \
        #     --main_process_port 29500  \
        #     --num_machines 1 \
        #     --machine_rank 0 \
        #     --num_processes ${gpu_num}  \
        #     --deepspeed_multinode_launcher standard \
        #     \
        #     ${GEN_SCRIPT_PATH} \
        #     --data_dir ${DATA_DIR} \
        #     --dataset_name ${dataset_name}  \
        #     --model_name ${model} \
        #     --output_dir outputs_combine_1 \
        #     --batch-image ${BATCH_SIZE} \
        #     --model_configs ${MODEL_CONFIG_PATH} \
        #     --overwrite \
        #     --combine_image 1 \
        #     > logs/${model}/${dataset_name}_combine_1.log

        # # Start evaluating
        # python ${EVAL_SCRIPT_PATH} \
        #     --data-dir ${DATA_DIR} \
        #     --dataset ${dataset_name} \
        #     --result-dir outputs_combine_1/${model} \
        #     >> logs/${model}/${dataset_name}_combine_1.log
    done
done
