"""The mtl_utils module, it is used to provide a funciton 
to get shared layers and task layers in multi-task model.
Available function:
- shared_task_layers: get shared layers and task layers in multi-task model
Authors: Qida Dong, dongjidan@126.com
"""
import torch
from ..models.multi_task import ESMM, MMOE, SharedBottom, PLE, AITM


def shared_task_layers(model):
	shared_layers = list(model.embedding.parameters())
	task_layers = None
	if isinstance(model, SharedBottom):
		shared_layers += list(model.bottom_mlp.parameters())
		task_layers = list(model.towers.parameters()) + list(model.predict_layers.parameters())
	elif isinstance(model, MMOE):
		shared_layers += list(model.experts.parameters())
		task_layers = list(model.towers.parameters()) + list(model.predict_layers.parameters())
		task_layers += list(model.gates.parameters())
	elif isinstance(model, PLE):
		shared_layers += list(model.cgc_layers.parameters())
		task_layers = list(model.towers.parameters()) + list(model.predict_layers.parameters())
	elif isinstance(model, AITM):
		shared_layers += list(model.bottoms.parameters())
		task_layers = list(model.info_gates.parameters()) + list(model.towers.parameters()) + list(model.aits.parameters())
	else:
		raise ValueError(f'this model {model} is not suitable for MetaBalance Optimizer')
	return shared_layers, task_layers

