from .coco_detection import CocoDetectionEvaluator, MyDetectionEvaluator


def build_evaluator(cfg, dataset):
    if cfg.evaluator.name == 'CocoDetectionEvaluator':
        return CocoDetectionEvaluator(dataset)
    elif cfg.evaluator.name == 'MyDetectionEvaluator':
        return MyDetectionEvaluator(dataset)
    else:
        raise NotImplementedError
