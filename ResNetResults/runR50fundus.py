import numpy as np
import torch
import os
from argparse import ArgumentParser

from ResNet50 import ResNet50
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    # ------------------------
    # 1 INIT MODEL
    # ------------------------
    print("model")
    model = ResNet50(False)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------

#     logger = TestTubeLogger(
#     save_dir='./lightning_logs',
#     version=39  # An existing version with a saved checkpoint
#     )

    print("trainer")

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_best_only=False,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='ResNet50RanditRun1'
    )

    trainer = Trainer(max_nb_epochs=50, 
                      early_stop_callback=None, 
                      gpus=4, distributed_backend='ddp', 
                      checkpoint_callback=checkpoint_callback, 
                      fast_dev_run=False)

#     trainer = Trainer(logger=logger,  default_save_path='./lightning_logs')

    '''
    The trainer stopped early becasue the validation set performed very well on epoch 9 for version 39,
    so I disabled the early stop callback to prevent this and get a full training/to satisfactory convergence.
    '''





    # ------------------------
    # 3 START TRAINING
    # ------------------------
    print("fitting")
    trainer.fit(model)


if __name__ == '__main__':
#     # ------------------------
#     # TRAINING ARGUMENTS
#     # ------------------------
#     # these are project-wide arguments
#     root_dir = os.path.dirname(os.path.realpath(__file__))
#     parent_parser = ArgumentParser(add_help=False)

#     # each LightningModule defines arguments relevant to it
#     parser = ResNet50.add_model_specific_args(parent_parser, root_dir)
#     hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main()
