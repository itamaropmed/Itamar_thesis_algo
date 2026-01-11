import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import optuna
import copy
import os
import csv
from collections import defaultdict

# ------------------------------------------------------------------------------
# 0. gpu setup (Mac MPS / cuda / CPU)
# ------------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Success: Using Apple M-Series GPU (MPS)")
else:
    device = torch.device("cpu")
    print(">>> Warning: Using CPU")

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------------------
# 1. environment
# ------------------------------------------------------------------------------

class PatientSchedulingEnv:
    def __init__(self, data, stochastic=True):
        self.meta = data['meta']
        self.patients = data['patients']
        self.num_patients = len(self.patients)
        self.day_limit = self.meta['day_limit']
        self.num_days = 5
        self.num_rooms = self.meta['num_rooms']
        self.num_doctors = self.meta['num_doctors']
        self.stochastic = stochastic

        self.p_map = {p['id']: p for p in self.patients}
        self.ids = [p['id'] for p in self.patients]

        self.state = None
        self.mask = None
        self.unscheduled = set()
        self.schedule_log = []
        self.room_usage = None
        self.doc_usage = None

    def reset(self):
        self.unscheduled = set(self.ids)
        self.schedule_log = []
        self.room_usage = np.zeros((self.num_days, self.num_rooms, self.day_limit), dtype=bool)
        self.doc_usage = np.zeros((self.num_days, self.num_doctors, self.day_limit), dtype=bool)
        return self._get_observation()

    def _get_observation(self):
        feats = []
        mask = []
        for pid in self.ids:
            p = self.p_map[pid]
            is_done = 1.0 if pid not in self.unscheduled else 0.0
            norm_dur = p['duration'] / 600.0
            norm_rooms = len(p['compatible_rooms']) / self.num_rooms
            norm_docs = len(p['compatible_doctors']) / self.num_doctors

            feats.append([norm_dur, norm_rooms, norm_docs, is_done])
            mask.append(1.0 if pid in self.unscheduled else 0.0)

        return torch.tensor(feats, dtype=torch.float32).to(device), torch.tensor(mask, dtype=torch.float32).to(device)

    def step(self, action_idx):
        pid = self.ids[action_idx]
        if pid not in self.unscheduled:
            return self._get_observation(), -5.0, True, {}  # Penalty for invalid

        p = self.p_map[pid]
        est_dur = p['duration']

        # stochastic duration (Paper Feature)
        if self.stochastic:
            noise = random.normalvariate(0, est_dur * 0.1)
            actual_dur = max(15, int(est_dur + noise))
        else:
            actual_dur = est_dur

        booked = False
        for d in range(self.num_days):
            if booked: break
            comp_rooms = p['compatible_rooms']
            comp_docs = p['compatible_doctors']

            for t in range(self.day_limit - actual_dur + 1):
                valid_r = -1
                for r in comp_rooms:
                    if not self.room_usage[d, r, t:t + actual_dur].any():
                        valid_r = r
                        break
                if valid_r == -1: continue

                valid_doc = -1
                for doc in comp_docs:
                    if not self.doc_usage[d, doc, t:t + actual_dur].any():
                        valid_doc = doc
                        break
                if valid_doc == -1: continue

                self.room_usage[d, valid_r, t:t + actual_dur] = True
                self.doc_usage[d, valid_doc, t:t + actual_dur] = True
                booked = True
                self.unscheduled.remove(pid)
                self.schedule_log.append({
                    'pid': pid, 'day': d, 'room': valid_r, 'start': t, 'end': t + actual_dur
                })
                break

        reward = 1.0 if booked else -1.0  # Reward for scheduling
        if not booked:
            self.unscheduled.remove(pid)  # Skip if impossible

        done = (len(self.unscheduled) == 0)
        return self._get_observation(), reward, done, {}


# ------------------------------------------------------------------------------
# 2. model: GNN Policy with Dropout (Regularization)
# ------------------------------------------------------------------------------

class GNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.0):
        super(GNNPolicy, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(p=dropout_rate)  # Added Dropout

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask):
        h = F.relu(self.embedding(x))
        for layer in self.layers:
            h = h + F.relu(layer(h))
            h = self.dropout(h)  # apply dropout after graph layers

        scores = self.actor(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        probs = F.softmax(scores, dim=0)

        global_h = torch.mean(h, dim=0)
        value = self.critic(global_h)

        return probs, value


# ------------------------------------------------------------------------------
# 3. ppo agent (with entropy & save best)
# ------------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, input_dim, hidden_dim, lr, num_layers, dropout, ent_coef, gamma=0.99, clip_eps=0.2):
        self.policy = GNNPolicy(input_dim, hidden_dim, num_layers, dropout).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef  # Entropy coefficient for exploration
        self.mse_loss = nn.MSELoss()

    def select_action(self, state, mask):
        probs, value = self.policy(state, mask)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def update(self, memory):
        states, masks, actions, old_log_probs, rewards, values, dones = zip(*memory)

        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done: discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns).to(device)
        old_log_probs = torch.stack(list(old_log_probs)).detach().to(device)
        actions = torch.tensor(actions).to(device)
        values = torch.stack(list(values)).detach().squeeze().to(device)

        # Advantage Normalization (Stabilizer)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):
            # Batch update (simplified to whole episode batch)
            # In production, use mini-batches if memory is tight
            policy_loss_sum = 0
            value_loss_sum = 0

            for i in range(len(states)):
                probs, val = self.policy(states[i], masks[i])
                dist = Categorical(probs)
                new_log_prob = dist.log_prob(actions[i])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - old_log_probs[i])
                advantage = advantages[i]

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage

                # PPO Loss + Entropy Bonus (to prevent premature convergence)
                p_loss = -torch.min(surr1, surr2) - self.ent_coef * entropy
                v_loss = 0.5 * self.mse_loss(val.squeeze(), returns[i])

                policy_loss_sum += p_loss
                value_loss_sum += v_loss

            loss = (policy_loss_sum + value_loss_sum) / len(states)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ------------------------------------------------------------------------------
# 4. tuning & execution
# ------------------------------------------------------------------------------

def load_data_file():
    if not os.path.exists('impossible_100.json'):
        return {'meta': {'day_limit': 600, 'num_rooms': 6, 'num_doctors': 15},
                'patients': [{'id': i, 'duration': 30, 'compatible_rooms': [0, 1], 'compatible_doctors': [0, 1]} for i
                             in range(250)]}
    with open('impossible_100.json') as f:
        return json.load(f)


raw_data = load_data_file()


def objective(trial):
    # expanded search space for paper reproduction
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)  # dropout
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)  # entropy

    env = PatientSchedulingEnv(raw_data, stochastic=True)
    agent = PPOAgent(input_dim=4, hidden_dim=hidden_dim, lr=lr, num_layers=num_layers,
                     dropout=dropout, ent_coef=ent_coef)

    total_episodes = 25
    avg_reward = 0
    for _ in range(total_episodes):
        state, mask = env.reset()
        done = False
        memory = []
        ep_reward = 0
        while not done:
            action, log_prob, value = agent.select_action(state, mask)
            obs, reward, done, _ = env.step(action)
            memory.append((state, mask, action, log_prob, reward, value, done))
            state, mask = obs
            ep_reward += reward
        agent.update(memory)
        avg_reward += ep_reward

    val = avg_reward / total_episodes
    print(f"Trial {trial.number}: Reward={val:.1f} | lr={lr:.5f} dim={hidden_dim} drop={dropout:.2f}")
    return val


def save_schedule(env_instance, filename="schedule_gnn_paper_hard.csv"):
    # Group by (Day, Room)
    path_map = defaultdict(list)
    for item in env_instance.schedule_log:
        path_map[(item['day'], item['room'])].append(item)

    with open(filename, 'w', newline='') as f:
        f.write("Metric,Value\n")
        f.write(f"Total Assigned,{len(env_instance.schedule_log)}\n")
        f.write(",\n")
        f.write("--- OPTIMIZED SCHEDULE ---\n")
        writer = csv.writer(f)
        writer.writerow(['Room ID', 'Patient IDs'])

        # Explicitly iterate 0..4 (Days) and 0..5 (Rooms)
        for d in range(env_instance.num_days):
            for r in range(env_instance.num_rooms):
                patients = path_map[(d, r)]
                if patients:
                    patients.sort(key=lambda x: x['start'])
                    pids = [str(p['pid']) for p in patients]
                    pids_str = ";".join(pids)
                    writer.writerow([r, pids_str])

    print(f"Schedule saved to {filename}")


def run_final_model(best_params):
    print("\n" + "=" * 40)
    print(f"Training Final Model: {best_params}")
    print("=" * 40)

    env = PatientSchedulingEnv(raw_data, stochastic=True)
    val_env = PatientSchedulingEnv(raw_data, stochastic=False)

    agent = PPOAgent(input_dim=4,
                     hidden_dim=best_params['hidden_dim'],
                     lr=best_params['lr'],
                     num_layers=best_params['num_layers'],
                     dropout=best_params['dropout'],
                     ent_coef=best_params['ent_coef'])

    # "Early Stopping" Logic: Save Best Weights
    best_train_reward = -float('inf')
    best_weights = copy.deepcopy(agent.policy.state_dict())

    train_episodes = 100
    for ep in range(train_episodes):
        state, mask = env.reset()
        done = False
        memory = []
        ep_reward = 0
        while not done:
            action, log_prob, value = agent.select_action(state, mask)
            obs, r, done, _ = env.step(action)
            memory.append((state, mask, action, log_prob, r, value, done))
            state, mask = obs
            ep_reward += r
        agent.update(memory)

        # Save best model seen so far (Early Stopping proxy)
        if ep_reward > best_train_reward:
            best_train_reward = ep_reward
            best_weights = copy.deepcopy(agent.policy.state_dict())

        if ep % 10 == 0:
            print(f"  Episode {ep:3d}/{train_episodes} | Reward: {ep_reward:.1f} (Best: {best_train_reward:.1f})")

    # Load Best Weights for Inference
    print("\nRestoring Best Model Weights...")
    agent.policy.load_state_dict(best_weights)

    print("Running Final Inference (Deterministic)...")
    state, mask = val_env.reset()
    done = False
    count = 0
    while not done:
        # Greedy Action for Evaluation
        probs, _ = agent.policy(state, mask)
        action = torch.argmax(probs).item()

        obs, r, done, _ = val_env.step(action)
        state, mask = obs
        if r > 0: count += 1

    print("-" * 30)
    print(f"GNN-RL Result: {count} / 250 Patients Scheduled")
    print("-" * 30)

    save_schedule(val_env, "schedule_gnn_paper.csv")


if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(f"\nBest params: {study.best_params}\n")

    run_final_model(study.best_params)