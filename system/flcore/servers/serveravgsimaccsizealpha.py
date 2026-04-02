import torch

from flcore.servers.serveravgsimacc import FedAvgSimAcc


class FedAvgSimAccSizeAlpha(FedAvgSimAcc):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.size_alpha = args.size_alpha
        print(f"Aggregation override: (sample-size ^ {self.size_alpha}) x similarity x accuracy")

    def _compute_joint_weights(self):
        adjusted_base_weights = [float(base_weight ** self.size_alpha) for base_weight in self.uploaded_weights]
        adjusted_sum = sum(adjusted_base_weights)
        if adjusted_sum <= 0:
            adjusted_base_weights = [1.0 / len(self.uploaded_weights) for _ in self.uploaded_weights]
        else:
            adjusted_base_weights = [weight / adjusted_sum for weight in adjusted_base_weights]

        delta_vectors = [self._model_delta_vector(model) for model in self.uploaded_models]
        consensus = torch.zeros_like(delta_vectors[0])
        for adjusted_weight, delta_vec in zip(adjusted_base_weights, delta_vectors):
            consensus += delta_vec * adjusted_weight

        consensus_norm = torch.norm(consensus) + 1e-12
        sim_scores = []
        for delta_vec in delta_vectors:
            denom = (torch.norm(delta_vec) * consensus_norm) + 1e-12
            sim_scores.append(torch.dot(delta_vec, consensus).item() / denom.item())

        raw_weights = []
        for client_id, adjusted_weight, sim_score in zip(self.uploaded_ids, adjusted_base_weights, sim_scores):
            client_acc = self.client_acc_map.get(client_id, 0.0)
            exponent = self.sim_tau * sim_score + self.acc_tau * client_acc
            raw_weights.append(adjusted_weight * float(torch.exp(torch.tensor(exponent)).item()))

        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            equal_weight = 1.0 / len(self.uploaded_models)
            return [equal_weight for _ in self.uploaded_models]

        return [weight / weight_sum for weight in raw_weights]
