import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from NeuralSystems import configure_system

hyperparameter_defaults = dict(
    system = "DentalRecurentVariationalSegmentor",
    criterion = "CrossEntropyCriterion",
    optimizer = torch.optim.Adam,
    lr = 1e-3,
    scheduler = "ReduceLROnPlateau",
    batch_size = 32,
    epochs = 200,
    patience = 20,
    data_path='/home/kpinigin/AnteriorClearDentalDataset512px/',
    resolution = 256,
    num_workers = 32,
)

def main():
    system = configure_system(hyperparameter_defaults["system"])(hyperparameter_defaults)
    logger = TensorBoardLogger('experiments_logs', name=str(hyperparameter_defaults['system'])

                                               + "_" + str(system.model.__class__.__name__)
                                               + "_" + str(hyperparameter_defaults['criterion'])
                                               + "_" + str(hyperparameter_defaults['scheduler'])
                               )

    early_stop = EarlyStopping(monitor="valid_iou", mode="max", verbose=True, patience=hyperparameter_defaults["patience"])
    model_checkpoint = ModelCheckpoint(monitor="valid_iou", mode="max", verbose=True, filename='Model-{epoch:02d}-{valid_iou:.5f}', save_top_k=3, save_last=True)
    trainer = pl.Trainer(
        gpus=[0,1],
        plugins=DDPPlugin(find_unused_parameters=True),
        max_epochs=hyperparameter_defaults['epochs'],
        logger = logger,
        check_val_every_n_epoch=1,
        accelerator='ddp',
        callbacks= [early_stop, model_checkpoint],
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        deterministic=True,
    )

    trainer.fit(system)
    trainer.test(system)

if __name__ == '__main__':
    main()