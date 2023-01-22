import os
import sys
import shutil
import glob
import logging

import mlflow
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.metrics import AverageMeter, MetricCalculator

logging.basicConfig(level=logging.INFO)

torch.manual_seed(84)

OUT_DIR = "outputs"
WEIGHTS = "weights"
CONFIG = "train_configs.yml"


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = self.model.criterion
        self.optimizer = self.model.optimizer
        self.scheduler = self.model.scheduler
        self.accum_iter = config.accum_iter
        self.res_path, self.weights_path = self.create_result_dir()
        self.dump_configs()
        logging.getLogger().addHandler(logging.FileHandler(os.path.join(self.res_path, "log.txt")))

        self.scaler = GradScaler()

    def create_result_dir(self):
        arch_name = self.config.experiment_name
        fold = self.config.fold
        dir_name = f"{arch_name}"
        dir_path = os.path.join(OUT_DIR, dir_name)
        dirs_num = len(glob.glob(f"{dir_path}/*"))
        dir_path = os.path.join(dir_path, f"{dirs_num}_{fold}")
        os.makedirs(dir_path, exist_ok=True)
        weights_path = os.path.join(dir_path, WEIGHTS)
        os.makedirs(weights_path, exist_ok=True)
        return dir_path, weights_path

    def dump_configs(self):
        path = os.path.join(self.res_path, CONFIG)
        try:
            shutil.copy(self.config["self_path"], path)
        except:
            pass

    def run_epoch(self, model, loader, train=True, scale_factor=None, ema=False):
        if train:
            model.train()
        else:
            model.eval()
        running_loss = AverageMeter()
        metrics = MetricCalculator()
        tk1 = tqdm(loader, total=int(len(loader)))

        self.optimizer.zero_grad()
        batch_idx = 0
        for data in tk1:
            if scale_factor is not None:
                data["img"] = F.interpolate(data["img"], scale_factor=scale_factor)
            if train:
                with autocast():
                    loss, outputs = self.model.iteration(data, train)
                    if isinstance(loss, dict):
                        loss["loss"] /= self.accum_iter
                        self.scaler.scale(loss["loss"]).backward()
                        loss["loss"] *= self.accum_iter
                    else:
                        loss /= self.accum_iter
                        self.scaler.scale(loss).backward()
                        loss *= self.accum_iter
                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                if (
                    "scheduler_batch_step" in self.config
                    and self.config["scheduler_batch_step"]
                ):
                    self.scheduler.step()
            else:
                with torch.no_grad():
                    loss, outputs = self.model.iteration(data, train, ema=ema)
            batch_idx += 1
            labels = data["target"] if isinstance(data, dict) else data[1]

            running_loss.add(loss, labels.shape[0])

            pred, labels = self.model.binarize(outputs, labels, self.config["thresh"])
            metrics.add(pred, labels, data["pred_id"])

            if batch_idx >= self.config.epoch_len:
                break

        result = metrics.get()
        loss = running_loss.get()
        if isinstance(loss, dict):
            for metric in loss:
                result[metric] = loss[metric]
        else:
            result["loss"] = loss
        return result

    def run(self, train_loader, valid_loader, loaders=[]):
        experiment_id = mlflow.set_experiment(self.config["experiment_name"])
        with mlflow.start_run(experiment_id=experiment_id):
            usefull_config = {
                key: self.config[key] for key in self.config if "transform" not in key
            }
            mlflow.log_params(usefull_config)
            mlflow.log_param("criterion", self.criterion)

            train_tr, test_tr = self.model.get_transform_dicts()
            for tr in train_tr:
                mlflow.log_param(tr, train_tr[tr])
            for tr in test_tr:
                mlflow.log_param("eval_" + tr, test_tr[tr])

            num_epochs = self.config["epochs"]
            for epoch in range(num_epochs):
                logging.info(f"Epoch: {epoch}.    Train:")

                scale_factor = None
                if len(self.config.progressive_resize) > epoch:
                    scale_factor = self.config.progressive_resize[epoch]
                train_res = self.run_epoch(
                    self.model, train_loader, train=True, scale_factor=scale_factor
                )
                self._print_result("Train", train_res)
                self.model.ema_update()
                logging.info("Validation:")
                val_res = self.run_epoch(
                    self.model, valid_loader, train=False, scale_factor=scale_factor
                )
                self._print_result("Val", val_res)

                # ema_val_res = self.run_epoch(
                #     self.model, valid_loader, train=False, scale_factor=scale_factor, ema=True
                # )
                # self._print_result("Val ema", ema_val_res)

                if self.config["scheduler_step"]:
                    self.scheduler.step(val_res["F1"])

                self._save_model(epoch, train_res, val_res)

                self._log_to_mlflow(epoch, train_res)
                self._log_to_mlflow(epoch, val_res, "val_")

                for i, eloader in enumerate(loaders):
                    eval_res = self.evaluate(eloader)
                    self._log_to_mlflow(epoch, eval_res, f"val_{i + 1}_")

    def evaluate(self, loader):
        eval_res = self.run_epoch(self.model, loader, train=False)
        self._print_result("Eval", eval_res)
        return eval_res

    def _log_to_mlflow(self, epoch, result, prefix=""):
        for key in result:
            if isinstance(result[key], dict):
                nested_prefix = prefix + key + "_"
                self._log_to_mlflow(epoch, result[key], nested_prefix)
            else:
                mlflow.log_metric(prefix + key, result[key], step=epoch)

    def _print_result(self, stage, result_dict):
        result = "".join(f"{key}:   {value}   " for key, value in result_dict.items() if not isinstance(value, dict))
        logging.info(f"{stage}: " + result)

        for key in result_dict:
            if isinstance(result_dict[key], dict):
                self._print_result(stage + "_" + key, result_dict[key])

    def _save_model(self, epoch, train_res, val_res):
        train_loss = train_res["F1"]
        val_loss = val_res["F1"]
        path = f"{self.weights_path}/{self.config['experiment_name']}_{epoch}_{train_loss}_{val_loss}.pth"
        self.model.save(path)
