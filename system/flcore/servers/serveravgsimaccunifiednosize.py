import torch

from flcore.servers.serveravgsimaccunified import FedAvgSimAccUnified


class FedAvgSimAccUnifiedNoSize(FedAvgSimAccUnified):
    def __init__(self, args, times):
        super().__init__(args, times)
        print("Aggregation override: similarity x relative accuracy (without sample-size weight)")

    def _compute_joint_weights(self):
        gamma_t = self._accuracy_gamma()
        sim_scores = self._compute_normalized_similarity_scores()

        raw_weights = []
        for client_id, sim_score in zip(self.uploaded_ids, sim_scores):
            relative_acc = self.client_relative_acc_ema_map.get(client_id, 0.0)
            exponent = self.sim_tau * sim_score + self.acc_tau * gamma_t * relative_acc
            raw_weights.append(float(torch.exp(torch.tensor(exponent)).item()))

        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            equal_weight = 1.0 / len(self.uploaded_models)
            return [equal_weight for _ in self.uploaded_models]

        return [weight / weight_sum for weight in raw_weights]
