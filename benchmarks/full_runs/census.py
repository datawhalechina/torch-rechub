"""Train Census-Income models on the published train, val and test CSVs."""

import torch

from benchmarks.datasets import build_census_multitask_dataset
from benchmarks.models import build_multitask_model
from torch_rechub.models.multi_task import AITM
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


def run(data, output, models, device, epochs=None):
    params = {"MMOE": {"n_expert": 8}, "SharedBottom": {"bottom_dims": [117]}, "PLE-2": {"n_level": 2}}
    results = {}
    for name in models:
        torch.manual_seed(2022)
        dataset = build_census_multitask_dataset({"path": data}, 16 if name == "ESMM" else 4, name, sample=False)
        if name == "AITM":
            model = AITM(dataset.features, 2, bottom_params={"dims": [32, 16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
        else:
            model_name = "PLE" if name == "PLE-2" else name
            model = build_multitask_model({"name": model_name, "params": params.get(name, {})}, dataset)

        train, val, test = DataGenerator(dataset.x_train, dataset.y_train).generate_dataloader(x_val=dataset.x_val, y_val=dataset.y_val, x_test=dataset.x_test, y_test=dataset.y_test, batch_size=1024)
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        trainer = MTLTrainer(
            model,
            task_types=dataset.task_types,
            optimizer_params={
                "lr": 0.001,
                "weight_decay": 0.0001
            },
            n_epoch=epochs or 1,
            earlystop_patience=epochs or 30,
            device=device,
            model_path=str(directory),
        )
        trainer.fit(train, val)
        scores = trainer.evaluate(trainer.model, test)
        results[name] = dict(zip(["AUC[income]", "AUC[marital status]", "AUC[joint]"], map(float, scores)))
    return results
