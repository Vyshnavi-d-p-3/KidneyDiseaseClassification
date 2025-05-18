#!/usr/bin/env python
import sys
import argparse
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_02b_prepare_effnet_model import PrepareEffnetModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_03b_model_training_effnet import ModelTrainingEffnetPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

# Mapping of stage names to pipeline classes
PIPELINES = {
    'stage_01_data_ingestion':          DataIngestionTrainingPipeline,
    'stage_02_prepare_base_model':      PrepareBaseModelTrainingPipeline,
    'stage_02b_prepare_effnet_model':   PrepareEffnetModelTrainingPipeline,  # <-- note the full class name
    'stage_03_model_training':          ModelTrainingPipeline,
    'stage_03b_model_training_effnet':  ModelTrainingEffnetPipeline,
    'stage_04_model_evaluation':        EvaluationPipeline,
}


def main():
    parser = argparse.ArgumentParser(
        description='Run a specific stage of the kidney CT-scan pipeline'
    )
    parser.add_argument(
        'stage',
        choices=PIPELINES.keys(),
        help='Stage to execute'
    )
    args = parser.parse_args()
    stage = args.stage
    pipeline_cls = PIPELINES[stage]
    pipeline = pipeline_cls()

    try:
        logger.info(f'>>> Starting {stage} <<<')
        pipeline.main()
        logger.info(f'>>> Completed {stage} <<<')
    except Exception as e:
        logger.exception(f'Error running {stage}')
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()
