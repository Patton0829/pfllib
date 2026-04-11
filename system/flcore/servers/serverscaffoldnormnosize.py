import copy
import random

import torch

from flcore.clients.clientscaffold import clientSCAFFOLD
from flcore.servers.serverscaffold import SCAFFOLD


class SCAFFOLDSimNormNoSize(SCAFFOLD):
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

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_models = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_models.append(copy.deepcopy(client.model))

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
        adapted_weights = self._compute_similarity_weights()

        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for server_param in global_model.parameters():
            server_param.data = server_param.data.clone()

        # Keep SCAFFOLD's control-variate update unchanged.
        # Only the model update uses similarity-based reweighting.
        for cid, weight in zip(self.uploaded_ids, adapted_weights):
            dy, dc = self.clients[cid].delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() * weight * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / self.num_clients
        self.global_model = global_model
        self.global_c = global_c
