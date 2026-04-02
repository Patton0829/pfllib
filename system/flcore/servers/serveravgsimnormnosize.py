import copy
import time

import torch

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class FedAvgSimNormNoSize(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.sim_tau = args.sim_tau

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Similarity temperature (sim_tau): {self.sim_tau}")
        print("Consensus direction: normalized client updates")
        print("Aggregation: similarity only (no sample-size weighting)")
        print("Finished creating server and clients.")

        self.Budget = []

    def _model_delta_vector(self, client_model):
        delta_chunks = []
        for global_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            delta = (client_param.data - global_param.data).reshape(-1)
            delta_chunks.append(delta)
        return torch.cat(delta_chunks)

    def _compute_similarity_weights(self):
        delta_vectors = [self._model_delta_vector(model) for model in self.uploaded_models]
        normalized_deltas = []
        for delta_vec in delta_vectors:
            normalized_deltas.append(delta_vec / (torch.norm(delta_vec) + 1e-12))

        consensus = torch.zeros_like(normalized_deltas[0])
        for norm_delta in normalized_deltas:
            consensus += norm_delta
        consensus = consensus / (torch.norm(consensus) + 1e-12)

        sim_scores = []
        for delta_vec in delta_vectors:
            delta_unit = delta_vec / (torch.norm(delta_vec) + 1e-12)
            sim_scores.append(torch.dot(delta_unit, consensus).item())

        raw_weights = []
        for sim_score in sim_scores:
            raw_weights.append(float(torch.exp(torch.tensor(self.sim_tau * sim_score)).item()))

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

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(verbose=self.should_print_round(i))

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
