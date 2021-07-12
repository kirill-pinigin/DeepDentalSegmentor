import torch
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from skimage.color import label2rgb
import numpy as np
import cv2
import monai
from DentalSegmentationDataset import DentalSegmentationDataset, DentalSegmentationDetectionDataset, INPUT_DIMENSION, OUTPUT_DIMENSION, MIN_RECT_POINTS_DIMENSION
from NeuralCriterions import configure_criterion
from NeuralMetrics import IntersectionOverUnion, SpatialMetric, SSIM
from NeuralModels import *

def configure_system(system_querry : str):
    system_collection = {"DentalSegmentor" : DentalSegmentor
        , "DentalJointDetectionSegmentor" : DentalJointDetectionSegmentor
        , "DentalVariationalSegmentor" : DentalVariationalSegmentor
        , 'DentalRecurentVariationalSegmentor': DentalRecurentVariationalSegmentor
                         }

    return system_collection[system_querry]


class DentalSegmentor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = UNet(in_channels=INPUT_DIMENSION, out_channels=OUTPUT_DIMENSION)
        self.criterion = configure_criterion(hparams["criterion"])
        self.metric = IntersectionOverUnion()
        self.hparams = hparams
        self.data_table = {'train_loss' :  [], 'train_iou' :  [], 'valid_loss' :  [] , 'valid_iou' :  [] }
        self.batch_size = hparams['batch_size']
        self.test_counter = int(0)
        self.make_datasets()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(),
                                     lr=self.hparams['lr'])

        if self.hparams['scheduler'] == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max', factor=0.5, min_lr=1e-7, verbose=True)

            scheduler = {
                'scheduler': lr_scheduler,
                'reduce_on_plateau': True,
                'monitor': 'valid_iou'
            }

        elif self.hparams['scheduler'] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7, verbose=True)

        elif self.hparams['scheduler'] == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=5, epochs=self.hparams['epochs'], verbose=True)

        return [optimizer], [scheduler]

    def evaluating_step(self, batch, batch_idx):
        img, mask = batch[0], batch[1]
        result = self(img)
        loss = self.criterion(result, mask)
        accuracy = self.metric(result, mask)
        return result, loss, accuracy

    def training_step(self, batch, batch_idx):
        _, loss, accuracy = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "loss": loss,
            "iou": accuracy,
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,logger= True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        _, loss, accuracy = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "valid_loss": loss,
            "valid_iou": accuracy,
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def test_step(self, batch, batch_idx):
        self.test_counter +=1
        result, loss, accuracy = self.evaluating_step(batch, batch_idx)
        output = torch.argmax(result.detach(), dim=1).squeeze(0).byte().cpu().numpy()
        self.visualize(batch[0], output, accuracy)
        batch_dictionary = {
            "test_loss": loss,
            "test_iou": accuracy,
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Train", avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'train_epoch_loss': avg_loss, 'train_epoch_iou': avg_iou})
        self.data_table['train_loss'].append(float(avg_loss))
        self.data_table['train_iou'].append(float(avg_iou))

    def validation_epoch_end(self, outputs):
        valid_avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        valid_avg_iou = torch.stack([x['valid_iou'] for x in outputs]).mean()
        print(" \n IoU/Valid = {}".format(valid_avg_iou))
        self.logger.experiment.add_scalar("Loss/Valid", valid_avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Valid", valid_avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'valid_epoch_loss': valid_avg_loss, 'valid_epoch_iou': valid_avg_iou})
        self.data_table['valid_loss'].append(float(valid_avg_loss))
        self.data_table['valid_iou'].append(float(valid_avg_iou))

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=32, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=32, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

    def visualize(self, image, output, accuracy):
        print(" output.mean() ", output.mean())
        img = torchvision.utils.make_grid(image.cpu().squeeze(0)).mul(float(255)).clamp(0,255).byte().permute(1,2,0).numpy()
        grid = label2rgb(output, img, bg_label=0, alpha=0.5, colors=None)
        cv2.imwrite("./"+self.logger.log_dir + "/IoU__"+ str(accuracy.item()) + "__"+ f"{self.test_counter:05d}" + "test_image.png",  (grid * float(255)).astype(np.uint8))
        grid = torch.from_numpy(grid).permute(2,0,1).float().mul(255.0).clamp(0.0,255.0).byte()
        self.logger.experiment.add_image(f"{self.test_counter:05d}" +"_test_image", grid, self.test_counter)

    def make_datasets(self):
        self.trainset = DentalSegmentationDataset(self.hparams['data_path'], split='train', resolution = self.hparams['resolution'])
        self.validset = DentalSegmentationDataset( self.hparams['data_path'], split='valid', resolution = self.hparams['resolution'])


class DentalJointDetectionSegmentor(DentalSegmentor):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = JointDetectionUNet(in_channels=INPUT_DIMENSION, out_channels=OUTPUT_DIMENSION, points_dimension=MIN_RECT_POINTS_DIMENSION * 2)
        self.points_criterion = torch.nn.MSELoss()
        self.spatial_metric = SpatialMetric()

    def evaluating_step(self, batch, batch_idx):
        img, mask = batch[0], batch[1]
        segments, points = self(img)
        seg_loss = self.criterion(segments, mask)
        seg_accuracy = self.metric(segments, mask)
        length = batch[2].size(0)
        point_loss = self.points_criterion(points.view(length, -1), batch[2])
        point_accuracy = self.spatial_metric(points.view(length, -1), batch[2])
        loss = seg_loss  + point_loss
        accuracy_dictionary = {
            "iou": seg_accuracy,
            "mae": point_accuracy,
        }
        return segments, loss, accuracy_dictionary

    def training_step(self, batch, batch_idx):
        _, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "loss": loss,
            "iou": accuracy_dictionary["iou"],
            "mae": accuracy_dictionary["mae"],
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,logger= True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        _, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "valid_loss": loss,
            "valid_iou": accuracy_dictionary["iou"],
            "valid_mae": accuracy_dictionary["mae"],
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def test_step(self, batch, batch_idx):
        self.test_counter +=1
        result, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "test_loss": loss,
            "test_iou": accuracy_dictionary["iou"],
            "test_mae": accuracy_dictionary["mae"],
        }

        output = torch.argmax(result.detach(), dim=1).squeeze(0).byte().cpu().numpy()
        self.visualize(batch[0], output, accuracy_dictionary["iou"])
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Train", avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'train_epoch_loss': avg_loss, 'train_epoch_iou': avg_iou})
        self.data_table['train_loss'].append(float(avg_loss))
        self.data_table['train_iou'].append(float(avg_iou))

    def validation_epoch_end(self, outputs):
        valid_avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        valid_avg_iou = torch.stack([x['valid_iou'] for x in outputs]).mean()
        valid_avg_mae = torch.stack([x['valid_mae'] for x in outputs]).mean()
        print(" \n IoU = {}".format(valid_avg_iou))
        print(" \n Spatial Accuracy ,percents = {}".format(valid_avg_mae*100.0))
        self.logger.experiment.add_scalar("Loss/Valid", valid_avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Valid", valid_avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'valid_epoch_loss': valid_avg_loss, 'valid_epoch_iou': valid_avg_iou})
        self.data_table['valid_loss'].append(float(valid_avg_loss))
        self.data_table['valid_iou'].append(float(valid_avg_iou))

    def make_datasets(self):
        self.trainset = DentalSegmentationDetectionDataset(self.hparams['data_path'], split='train', resolution = self.hparams['resolution'])
        self.validset = DentalSegmentationDetectionDataset(self.hparams['data_path'], split='valid', resolution = self.hparams['resolution'])

class DentalVariationalSegmentor(DentalSegmentor):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = VariationalUNet(in_channels=INPUT_DIMENSION, out_channels=OUTPUT_DIMENSION)
        self.reconstruction_criterion = torch.nn.BCELoss()
        self.spatial_metric = SpatialMetric()
        #self.ssim_metric = SSIM(INPUT_DIMENSION)

    def evaluating_step(self, batch, batch_idx):
        img, mask = batch[0], batch[1]
        segments, reconstruction, kl = self(img)
        seg_loss = self.criterion(segments, mask)
        seg_accuracy = self.metric(segments, mask)
        reconstruction_loss = self.reconstruction_criterion(reconstruction, img)

        loss = seg_loss  + reconstruction_loss + 1e-3*kl
        reconstruction_accuracy = self.spatial_metric(reconstruction, img)
        #ssim = self.ssim_metric(reconstruction, img)

        accuracy_dictionary = {
            "iou": seg_accuracy,
            "mae": reconstruction_accuracy,
            #"ssim": ssim,
        }
        return segments, loss, accuracy_dictionary

    def training_step(self, batch, batch_idx):
        _, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "loss": loss,
            "iou": accuracy_dictionary["iou"],
            "mae": accuracy_dictionary["mae"],
            #"ssim": accuracy_dictionary["ssim"],
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,logger= True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        _, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "valid_loss": loss,
            "valid_iou": accuracy_dictionary["iou"],
            "valid_mae": accuracy_dictionary["mae"],
            #"valid_ssim": accuracy_dictionary["ssim"],
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def test_step(self, batch, batch_idx):
        self.test_counter +=1
        result, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "test_loss": loss,
            "test_iou": accuracy_dictionary["iou"],
            "test_mae": accuracy_dictionary["mae"],
            #"test_ssim": accuracy_dictionary["ssim"],
        }

        output = torch.argmax(result.detach(), dim=1).squeeze(0).byte().cpu().numpy()
        self.visualize(batch[0], output, accuracy_dictionary["iou"])
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Train", avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'train_epoch_loss': avg_loss, 'train_epoch_iou': avg_iou})
        self.data_table['train_loss'].append(float(avg_loss))
        self.data_table['train_iou'].append(float(avg_iou))

    def validation_epoch_end(self, outputs):
        valid_avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        valid_avg_iou = torch.stack([x['valid_iou'] for x in outputs]).mean()
        valid_avg_mae = torch.stack([x['valid_mae'] for x in outputs]).mean()
        #valid_avg_ssim = torch.stack([x['valid_ssim'] for x in outputs]).mean()
        print(" \n IoU = {}".format(valid_avg_iou))
        print(" \n Spatial Accuracy ,percents = {}".format(valid_avg_mae*100.0))
        #print(" \n Similarity ,percents = {}".format(valid_avg_ssim * 100.0))
        self.logger.experiment.add_scalar("Loss/Valid", valid_avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Valid", valid_avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'valid_epoch_loss': valid_avg_loss, 'valid_epoch_iou': valid_avg_iou})
        self.data_table['valid_loss'].append(float(valid_avg_loss))
        self.data_table['valid_iou'].append(float(valid_avg_iou))


class DentalRecurentVariationalSegmentor(DentalVariationalSegmentor):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = RecurentVariationalUNet(in_channels=INPUT_DIMENSION, out_channels=OUTPUT_DIMENSION)
        self.reconstruction_criterion = torch.nn.BCELoss()
        self.spatial_metric = SpatialMetric()

    def evaluating_step(self, batch, batch_idx):
        img, mask = batch[0], batch[1]
        segments, reconstruction, kl = self(img)
        seg_loss = self.criterion(segments, mask)
        seg_accuracy = self.metric(segments, mask)
        reconstruction_loss = self.reconstruction_criterion(reconstruction, img)

        loss = seg_loss  + reconstruction_loss + 1e-3*kl
        reconstruction_accuracy = self.spatial_metric(reconstruction, img)
        #ssim = self.ssim_metric(reconstruction, img)

        accuracy_dictionary = {
            "iou": seg_accuracy,
            "mae": reconstruction_accuracy,
            #"ssim": ssim,
        }
        return segments, loss, accuracy_dictionary

    def training_step(self, batch, batch_idx):
        _, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "loss": loss,
            "iou": accuracy_dictionary["iou"],
            "mae": accuracy_dictionary["mae"],
            #"ssim": accuracy_dictionary["ssim"],
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,logger= True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        _, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "valid_loss": loss,
            "valid_iou": accuracy_dictionary["iou"],
            "valid_mae": accuracy_dictionary["mae"],
            #"valid_ssim": accuracy_dictionary["ssim"],
        }
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def test_step(self, batch, batch_idx):
        self.test_counter +=1
        result, loss, accuracy_dictionary = self.evaluating_step(batch, batch_idx)
        batch_dictionary = {
            "test_loss": loss,
            "test_iou": accuracy_dictionary["iou"],
            "test_mae": accuracy_dictionary["mae"],
            #"test_ssim": accuracy_dictionary["ssim"],
        }

        output = torch.argmax(result.detach(), dim=1).squeeze(0).byte().cpu().numpy()
        self.visualize(batch[0], output, accuracy_dictionary["iou"])
        self.log_dict(batch_dictionary, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Train", avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'train_epoch_loss': avg_loss, 'train_epoch_iou': avg_iou})
        self.data_table['train_loss'].append(float(avg_loss))
        self.data_table['train_iou'].append(float(avg_iou))

    def validation_epoch_end(self, outputs):
        valid_avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        valid_avg_iou = torch.stack([x['valid_iou'] for x in outputs]).mean()
        valid_avg_mae = torch.stack([x['valid_mae'] for x in outputs]).mean()
        #valid_avg_ssim = torch.stack([x['valid_ssim'] for x in outputs]).mean()
        print(" \n IoU = {}".format(valid_avg_iou))
        print(" \n Spatial Accuracy ,percents = {}".format(valid_avg_mae*100.0))
        #print(" \n Similarity ,percents = {}".format(valid_avg_ssim * 100.0))
        self.logger.experiment.add_scalar("Loss/Valid", valid_avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("IoU/Valid", valid_avg_iou, self.current_epoch)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log_dict({'valid_epoch_loss': valid_avg_loss, 'valid_epoch_iou': valid_avg_iou})
        self.data_table['valid_loss'].append(float(valid_avg_loss))
        self.data_table['valid_iou'].append(float(valid_avg_iou))