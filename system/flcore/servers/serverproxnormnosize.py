import copy

import torch

from flcore.clients.clientprox import clientProx
from flcore.servers.serverprox import FedProx


class FedProxSimNormNoSize(FedProx):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.sim_tau = args.sim_tau
        print(f"Similarity temperature (sim_tau): {self.sim_tau}")
        print("Aggregation override: normalized similarity without sample-size weighting")

    def _model_delta_vector(self, client_model):
        delta_chunks = []
        for global_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            delta = (client_param.data - global_param.data).reshape(-1)
            delta_chunks.append(delta)
        return torch.cat(delta_chunks)

    def _compute_similarity_weights(self):
        delta_vectors = [self._model_delta_vector(model) for model in self.uploaded_models]
        normalized_deltas = [delta_vec / (torch.norm(delta_vec) + 1e-12) for delta_vec in delta_vectors]

        consensus = torch.zeros_like(normalized_deltas[0])
        for norm_delta in normalized_deltas:
            consensus += norm_delta
        consensus = consensus / (torch.norm(consensus) + 1e-12)

        sim_scores = []
        for delta_vec in delta_vectors:
            delta_unit = delta_vec / (torch.norm(delta_vec) + 1e-12)
            sim_scores.append(torch.dot(delta_unit, consensus).item())

        raw_weights = [float(torch.exp(torch.tensor(self.sim_tau * sim_score)).item()) for sim_score in sim_scores]
        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            uniform_weight = 1.0 / len(self.uploaded_models)
            return [uniform_weight for _ in self.uploaded_models]
        return [weight / weight_sum for weight in raw_weights]

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        adapted_weights = self._compute_similarity_weights()

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for weight, client_model in zip(adapted_weights, self.uploaded_models):
            self.add_parameters(weight, client_model)
