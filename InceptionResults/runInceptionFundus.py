import numpy as np
import torch
import os
from argparse import ArgumentParser

from InceptionV3 import InceptionV3
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    # ------------------------
    # 1 INIT MODEL 
    # ------------------------
    print("model")
    model = InceptionV3(True)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    
#     logger = TestTubeLogger(
#     save_dir='./lightning_logs',
#     version=39  # An existing version with a saved checkpoint
#     )
    checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_best_only=False,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='InceptionV3_Trans')
    
    print("trainer")
    trainer = Trainer(max_nb_epochs=50, 
                      checkpoint_callback=checkpoint_callback, 
                      early_stop_callback=None, 
                      gpus=3, 
                      distributed_backend='ddp')
#     trainer = Trainer(logger=logger,  default_save_path='./lightning_logs')
    
    
    
    
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
