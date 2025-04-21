import re
import ast
import matplotlib.pyplot as plt

class ScoreEvaluator:
    def __init__(self):
        self.hw_constraints = {
            'flash_size': 128 * 1024,  # 1MB in bytes
            'cpu_frequency': 180_000_000          # 168 MHz
        }

        self.accuracy_scores = []
        self.size_scores = []
        self.speed_scores = []

    def _estimate_flops(self, params):
        seq_length = 32
        h = params['hidden_size']
        l = params['num_layers']
        ff_dim = params['ff_dim']

        attn_flops = seq_length * seq_length * h * 2
        ffn_flops = seq_length * h * ff_dim * 2

        return (attn_flops + ffn_flops) * l

    def _estimate_model_size(self, params):
        h = params['hidden_size']
        l = params['num_layers']
        ff_dim = params['ff_dim']
        v = params['vocab_size']
        bits = params['quantization_bits']

        embedding_params = v * h
        transformer_params = l * (4 * h * h + 2 * h * ff_dim)

        total_params = embedding_params + transformer_params
        return (total_params * bits) / 8  # in bytes

    def process_log_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        architecture = None
        score = None

        for line in lines:
            if "architecture:" in line:
                architecture = ast.literal_eval(re.search(r"\{.*\}", line).group(0))


            elif "Score:" in line:    # for RL
                score = float(re.search(r"Score: ([0-9.]+)", line).group(1))
            # elif "score:" in line:      # for BO
            #     score = float(re.search(r"score: ([0-9.]+)", line).group(1))
                model_size = self._estimate_model_size(architecture)
                flops = self._estimate_flops(architecture)

                size_score = 1 - model_size / self.hw_constraints['flash_size']
                inf_time = flops / self.hw_constraints['cpu_frequency']
                speed_score = 1 - inf_time / 100

                accuracy = 2 * (score - 0.25 * size_score - 0.25 * speed_score)

                self.accuracy_scores.append(accuracy)
                self.size_scores.append(size_score)
                self.speed_scores.append(speed_score)

    def get_scores(self):
        return self.accuracy_scores, self.size_scores, self.speed_scores

    def plot_scores(self):
        iterations = list(range(1, len(self.accuracy_scores) + 1))

        plt.figure(figsize=(12, 6))
        plt.plot(iterations[:75], self.accuracy_scores[:75], label='Accuracy Score', color='blue', marker='o', linestyle='-')
        # plt.plot(iterations, self.size_scores, label='Size Score', color='green', marker='s', linestyle='--')
        plt.plot(iterations[:75], self.speed_scores[:75], label='Speed Score', color='red', marker='^', linestyle='-.')

        # plt.title("RL-NAS 50 iterations  Config 4", fontsize=16)
        plt.title("RL-NAS via PPO: 50 iterations Config 4", fontsize=16)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(iterations[::2])
        # plt.ylim(0.98, 1.00)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.savefig("bo_nas_50_iters_5_epochs_config4.png", dpi=300)
        plt.savefig("rl_nas_50_iters_config4.png", dpi=300)



if __name__ == "__main__":
    evaluator = ScoreEvaluator()
    evaluator.process_log_file("runs/20250419_130511/search.log")

    acc_scores, sz_scores, spd_scores = evaluator.get_scores()
    evaluator.plot_scores()
