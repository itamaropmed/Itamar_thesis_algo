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
# 0. GPU SETUP (Mac MPS / CUDA / CPU)
# ------------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Success: Using Apple M-Series GPU (MPS)")
else:
    device = torch.device("cpu")
    print(">>> Warning: Using CPU")

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------------------
# 1. ENVIRONMENT (RCPSP with Resource Disruptions - Cai et al. Style)
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
        self.stochastic = stochastic  # If True, resources might break

        self.p_map = {p['id']: p for p in self.patients}
        self.ids = [p['id'] for p in self.patients]

        self.state = None
        self.mask = None
        self.unscheduled = set()
        self.schedule_log = []

        # Resources [Day][Room][Time], [Day][Doc][Time]
        self.room_usage = None
        self.doc_usage = None

        # Disruption Status (Mocking broken resources)
        self.broken_rooms = set()
        self.broken_docs = set()

    def reset(self):
        self.unscheduled = set(self.ids)
        self.schedule_log = []
        self.room_usage = np.zeros((self.num_days, self.num_rooms, self.day_limit), dtype=bool)
        self.doc_usage = np.zeros((self.num_days, self.num_doctors, self.day_limit), dtype=bool)
        self.broken_rooms = set()
        self.broken_docs = set()
        return self._get_observation()

    def _get_observation(self):
        feats = []
        mask = []
        for pid in self.ids:
            p = self.p_map[pid]
            is_done = 1.0 if pid not in self.unscheduled else 0.0

            # Static Features
            norm_dur = p['duration'] / 600.0
            norm_rooms = len(p['compatible_rooms']) / self.num_rooms
            norm_docs = len(p['compatible_doctors']) / self.num_doctors

            # Dynamic Feature: Resource Availability Impact (Cai et al.)
            # "How many of my compatible resources are currently broken?"
            broken_r_count = sum(1 for r in p['compatible_rooms'] if r in self.broken_rooms)
            broken_d_count = sum(1 for d in p['compatible_doctors'] if d in self.broken_docs)

            res_stress = (broken_r_count + broken_d_count) / 10.0  # Normalized stress metric

            feats.append([norm_dur, norm_rooms, norm_docs, is_done, res_stress])
            mask.append(1.0 if pid in self.unscheduled else 0.0)

        return torch.tensor(feats, dtype=torch.float32).to(device), torch.tensor(mask, dtype=torch.float32).to(device)

    def step(self, action_idx):
        pid = self.ids[action_idx]
        if pid not in self.unscheduled:
            return self._get_observation(), -5.0, True, {}

        p = self.p_map[pid]
        actual_dur = p['duration']  # Duration is deterministic in Cai model (usually), resources are stochastic

        # --- SIMULATE RESOURCE DISRUPTION (Stochastic) ---
        if self.stochastic and random.random() < 0.05:  # 5% chance of breakdown per step
            # Break a random room or doctor for the rest of the day
            if random.random() < 0.5:
                self.broken_rooms.add(random.randint(0, self.num_rooms - 1))
            else:
                self.broken_docs.add(random.randint(0, self.num_doctors - 1))

        booked = False
        for d in range(self.num_days):
            if booked: break

            # Reset disruptions on new day (simplification)
            # In a full simulation, repair takes time. Here we assume daily reset.
            # But strictly for step logic, we just check availability.

            comp_rooms = [r for r in p['compatible_rooms'] if r not in self.broken_rooms]
            comp_docs = [doc for doc in p['compatible_doctors'] if doc not in self.broken_docs]

            if not comp_rooms or not comp_docs: continue  # Cannot schedule if all resources broken

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

        reward = 1.0 if booked else -1.0
        if not booked:
            self.unscheduled.remove(pid)

        done = (len(self.unscheduled) == 0)
        return self._get_observation(), reward, done, {}


# ------------------------------------------------------------------------------
# 2. model: GNN with Embeddings (Cai et al.)
# ------------------------------------------------------------------------------

class GNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.0):
        super(GNNPolicy, self).__init__()
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # GNN Layers (Modeling dependencies)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(p=dropout_rate)

        # Global Pooling embedding (Context)
        self.global_fc = nn.Linear(hidden_dim, hidden_dim)

        # Actor takes Node Embedding + Global Context
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenation
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
        # 1. Node Embeddings
        h = F.relu(self.embedding(x))
        for layer in self.layers:
            h = h + F.relu(layer(h))
            h = self.dropout(h)

        # 2. Global Graph Embedding (Pooling)
        # Average pooling of all node features to represent "Project State"
        global_h = torch.mean(h, dim=0, keepdim=True).expand(x.size(0), -1)
        global_h = F.relu(self.global_fc(global_h))

        # 3. Concatenate (Node + Global) - Cai et al. technique
        combined = torch.cat([h, global_h], dim=1)

        # 4. Actor
        scores = self.actor(combined).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        probs = F.softmax(scores, dim=0)

        # 5. Critic (uses global context directly)
        value = self.critic(torch.mean(h, dim=0))

        return probs, value


# ------------------------------------------------------------------------------
# 3. ppo agent
# ------------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, input_dim, hidden_dim, lr, num_layers, dropout, ent_coef, gamma=0.99, clip_eps=0.2):
        self.policy = GNNPolicy(input_dim, hidden_dim, num_layers, dropout).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
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

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):
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

                p_loss = -torch.min(surr1, surr2) - self.ent_coef * entropy
                v_loss = 0.5 * self.mse_loss(val.squeeze(), returns[i])

                policy_loss_sum += p_loss
                value_loss_sum += v_loss

            loss = (policy_loss_sum + value_loss_sum) / len(states)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ------------------------------------------------------------------------------
# 4. TUNING & EXECUTION
# ------------------------------------------------------------------------------

def load_data_file():
    if not os.path.exists('large_setup_250.json'):
        return {'meta': {'day_limit': 600, 'num_rooms': 6, 'num_doctors': 15},
                'patients': [{'id': i, 'duration': 30, 'compatible_rooms': [0, 1], 'compatible_doctors': [0, 1]} for i
                             in range(250)]}
    with open('large_setup_250.json') as f:
        return json.load(f)


raw_data = load_data_file()


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)

    env = PatientSchedulingEnv(raw_data, stochastic=True)  # Train with disruptions
    agent = PPOAgent(input_dim=5, hidden_dim=hidden_dim, lr=lr, num_layers=num_layers,
                     # input_dim=5 (added stress feat)
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
    print(f"Trial {trial.number}: Reward={val:.1f} | Params: {trial.params}")
    return val


def save_schedule(env_instance, filename="schedule_gnn_cai.csv"):
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
    print(f"Training Final Model (Cai et al.): {best_params}")
    print("=" * 40)

    env = PatientSchedulingEnv(raw_data, stochastic=True)
    val_env = PatientSchedulingEnv(raw_data, stochastic=False)  # Validation: No broken resources

    agent = PPOAgent(input_dim=5,
                     hidden_dim=best_params['hidden_dim'],
                     lr=best_params['lr'],
                     num_layers=best_params['num_layers'],
                     dropout=best_params['dropout'],
                     ent_coef=best_params['ent_coef'])

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

        if ep_reward > best_train_reward:
            best_train_reward = ep_reward
            best_weights = copy.deepcopy(agent.policy.state_dict())

        if ep % 10 == 0:
            print(f"  Episode {ep:3d}/{train_episodes} | Reward: {ep_reward:.1f} (Best: {best_train_reward:.1f})")

    print("\nRestoring Best Model Weights...")
    agent.policy.load_state_dict(best_weights)

    print("Running Final Inference (Deterministic)...")
    state, mask = val_env.reset()
    done = False
    count = 0
    while not done:
        probs, _ = agent.policy(state, mask)
        action = torch.argmax(probs).item()

        obs, r, done, _ = val_env.step(action)
        state, mask = obs
        if r > 0: count += 1

    print("-" * 30)
    print(f"Cai-GNN Result: {count} / 250 Patients Scheduled")
    print("-" * 30)

    save_schedule(val_env, "schedule_gnn_cai.csv")


if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(f"\nBest params: {study.best_params}\n")

    run_final_model(study.best_params)