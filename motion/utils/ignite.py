import torch
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Engine, Events
from torch.optim import Optimizer


def set_default_tb_train_logging(tb_logger: TensorboardLogger, trainer: Engine, optimizer: Optimizer,
                                 model: torch.nn.Module):
    """
    Logs standard information to tensorboard during training.
    """
    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        **{
            'tag': 'training',
            'output_transform': lambda loss: {"loss": loss[0]}
        }
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED(every=10),
        **{
            'optimizer': optimizer,
            'param_name': 'lr',
            'tag': 'training'
        }
    )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        log_handler=WeightsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        log_handler=GradsScalarHandler(model)
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=GradsHistHandler(model)
    )
