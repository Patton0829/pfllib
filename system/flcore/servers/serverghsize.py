import copy
import time

import torch

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class FedGHSize(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.gh_conflict_threshold = args.gh_conflict_threshold
        self.gh_eps = 1e-12

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Gradient conflict threshold: cos < {self.gh_conflict_threshold}")
        print("Aggregation: pairwise conflict projection + sample-size weighted averaging")
        print("Finished creating server and clients.")

        self.Budget = []

    def _model_delta_vector(self, client_model):
        delta_chunks = []
        for global_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            delta = (client_param.data - global_param.data).reshape(-1)
            delta_chunks.append(delta)
        return torch.cat(delta_chunks)

    def _compute_projected_deltas(self):
        delta_vectors = [self._model_delta_vector(model) for model in self.uploaded_models]
        projected_deltas = [delta.clone() for delta in delta_vectors]

        for i in range(len(delta_vectors)):
            for j in range(i + 1, len(delta_vectors)):
                g_i = delta_vectors[i]
                g_j = delta_vectors[j]
                norm_i = torch.norm(g_i)
                norm_j = torch.norm(g_j)

                if norm_i <= self.gh_eps or norm_j <= self.gh_eps:
                    continue

                cosine = torch.dot(g_i, g_j) / (norm_i * norm_j + self.gh_eps)
                if cosine.item() >= self.gh_conflict_threshold:
                    continue

                proj_i = torch.dot(projected_deltas[i], g_j) / (torch.dot(g_j, g_j) + self.gh_eps)
                proj_j = torch.dot(projected_deltas[j], g_i) / (torch.dot(g_i, g_i) + self.gh_eps)
                projected_deltas[i] = projected_deltas[i] - proj_i * g_j
                projected_deltas[j] = projected_deltas[j] - proj_j * g_i

        return projected_deltas

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        projected_deltas = self._compute_projected_deltas()
        weighted_delta = sum(
            weight * delta for weight, delta in zip(self.uploaded_weights, projected_deltas)
        )

        self.global_model = copy.deepcopy(self.global_model)
        cursor = 0
        for param in self.global_model.parameters():
            numel = param.data.numel()
            delta_slice = weighted_delta[cursor:cursor + numel].reshape(param.data.shape)
            param.data += delta_slice
            cursor += numel

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
