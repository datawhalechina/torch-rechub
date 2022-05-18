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

# share ali meta:test auc: [0.6122230867729563, 0.6102749439999999]
# share ali adam:test auc: [0.6042237572680785, 0.59811668]
# mmoe ali meta:test auc: [0.5516007157418209, 0.5518165456]
# mmoe ali adam:test auc: [0.5398094929187571, 0.5879860592]
# ple ali meta:test auc: [0.7634381816288129, 0.5400777488]
# ple ali adam:test auc: [0.6046847571408265, 0.5892243184]
# aitm ali meta:test auc: [0.5394383038128273, 0.5424584976]
# aitm ali adam:test auc: [0.5424497743198506, 0.5270031264000001]