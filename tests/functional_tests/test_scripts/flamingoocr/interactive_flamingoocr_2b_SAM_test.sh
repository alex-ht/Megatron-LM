#! /bin/bash

# These paths should be modified by user to use their source code and output dir
CHECKPOINT_PATH=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingoocr_2b_SAM_test/checkpoints
TENSORBOARD_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingoocr_2b_SAM_test/tensorboard_logs
SCRIPTS_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/source/megatron-lm-flamingo-tests

DATA_PATH=/lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data/flamingo_data
mkdir /workspace/data
ln -s $DATA_PATH /workspace/data/flamingo_data

USE_CORE=0
USE_TE=0
MBS=1
GBS=8
MAX_STEPS=50
NUM_NODES=1
TP_SIZE=1
PP_SIZE=1
VISUAL_ARCH="SAM_L"
VISUAL_TYPE="sam"
VISUAL_PATH=${DATA_PATH}/SAM_L_16
IMG_H=1024
IMG_W=1024
SAM_RANDINIT=1

bash ./tests/functional_tests/test_scripts/flamingoocr/finetune_flamingoocr_distributed_test.sh DATA_PATH=$DATA_PATH CHECKPOINT_PATH=$CHECKPOINT_PATH TENSORBOARD_DIR=$TENSORBOARD_DIR SCRIPTS_DIR=$SCRIPTS_DIR USE_TE=$USE_TE TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE VP_SIZE=$VP_SIZE NUM_NODES=$NUM_NODES MAX_STEPS=$MAX_STEPS USE_CORE=$USE_CORE MBS=$MBS GBS=$GBS VISUAL_ARCH=$VISUAL_ARCH VISUAL_TYPE=$VISUAL_TYPE VISUAL_PATH=$VISUAL_PATH IMG_H=$IMG_H IMG_W=$IMG_W SAM_RANDINIT=$SAM_RANDINIT

python tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py /lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingoocr_2b_SAM_test/tensorboard_logs flamingoocr_2b_SAM_test

export EXPECTED_METRICS_FILE=tests/functional_tests/test_results/flamingoocr/flamingoocr_2b_SAM_test.json

export LOGS_DIR=/lustre/fsw/adlr/adlr-nlp/jbarker/next-llm/output/flamingoocr_2b_SAM_test/tensorboard_logs

pytest tests/functional_tests/python_test_utils/test_ci_pipeline.py