import torch

from flcore.servers.serveravgsimacc import FedAvgSimAcc


class FedAvgSimAccNoSize(FedAvgSimAcc):
    def __init__(self, args, times):
        super().__init__(args, times)
        print("Aggregation override: similarity x accuracy (without sample-size weight)")

    def _compute_joint_weights(self):
        delta_vectors = [self._model_delta_vector(model) for model in self.uploaded_models]
        consensus = torch.zeros_like(delta_vectors[0])
        uniform_weight = 1.0 / len(delta_vectors)
        for delta_vec in delta_vectors:
            consensus += delta_vec * uniform_weight

        consensus_norm = torch.norm(consensus) + 1e-12
        sim_scores = []
        for delta_vec in delta_vectors:
            denom = (torch.norm(delta_vec) * consensus_norm) + 1e-12
            sim_scores.append(torch.dot(delta_vec, consensus).item() / denom.item())

        raw_weights = []
        for client_id, sim_score in zip(self.uploaded_ids, sim_scores):
            client_acc = self.client_acc_map.get(client_id, 0.0)
            exponent = self.sim_tau * sim_score + self.acc_tau * client_acc
            raw_weights.append(float(torch.exp(torch.tensor(exponent)).item()))

        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            equal_weight = 1.0 / len(self.uploaded_models)
            return [equal_weight for _ in self.uploaded_models]

        return [weight / weight_sum for weight in raw_weights]
