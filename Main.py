import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import math
import warnings
import os
import copy
import shap
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_plot(fig, filename):
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=600, bbox_inches='tight')


def plot_classwise_shap(model, X_background, X_samples, feature_names, class_names, max_features=10):
    model_cpu = model.to('cpu')
    model_cpu.eval()

    background = X_background[:min(200, len(X_background))].detach().cpu()
    samples = X_samples[:min(300, len(X_samples))].detach().cpu()

    explainer = shap.GradientExplainer(model_cpu, background)
    shap_values = explainer.shap_values(samples)

    if isinstance(shap_values, list):
        shap_arrays = [np.squeeze(np.array(values)) for values in shap_values]
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_arrays = [np.squeeze(shap_values[:, :, i]) for i in range(shap_values.shape[2])]
        elif shap_values.ndim == 4:
            shap_arrays = [np.squeeze(shap_values[:, :, i, 0]) for i in range(shap_values.shape[2])]
        else:
            raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

    for class_idx, class_name in enumerate(class_names):
        class_shap = np.atleast_2d(shap_arrays[class_idx])
        mean_abs_shap = np.mean(np.abs(class_shap), axis=0)

        top_idx = np.argsort(mean_abs_shap)[-max_features:]
        top_features = [feature_names[i] for i in top_idx]
        top_values = mean_abs_shap[top_idx]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
        ax.barh(top_features, top_values, color='#5b8e7d', edgecolor='black')

        ax.set_title(f"SHAP Feature Importance - {class_name}",
                     fontsize=20, fontweight='bold', fontname='Times New Roman')
        ax.set_xlabel("Mean |SHAP Value|",
                      fontsize=18, fontweight='bold', fontname='Times New Roman')
        ax.set_ylabel("Features",
                      fontsize=18, fontweight='bold', fontname='Times New Roman')
        ax.tick_params(axis='both', labelsize=16)

        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontname('Times New Roman')

        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontname('Times New Roman')

        ax.grid(axis='x', linestyle='--', alpha=0.4)
        style_plot(ax)
        plt.tight_layout()
        save_plot(fig, f"{13 + class_idx:02d}_shap_{class_name.lower()}_class.png")
        plt.show()


def balance_training_data(X, y, threshold=0.8):
    class_counts = y.value_counts().sort_index()
    min_count = class_counts.min()
    max_count = class_counts.max()
    imbalance_ratio = min_count / max_count

    print("\nTraining Class Distribution Before Balancing:")
    print(class_counts)

    if imbalance_ratio >= threshold:
        print("Training data is already balanced enough. Skipping balancing.")
        return X, y

    print(f"Imbalance detected (ratio={imbalance_ratio:.3f}). Applying random oversampling on training data.")

    train_df = X.copy()
    train_df['classification'] = y.values

    majority_count = class_counts.max()
    balanced_parts = []

    for class_value in class_counts.index:
        class_df = train_df[train_df['classification'] == class_value]
        if len(class_df) < majority_count:
            class_df = class_df.sample(majority_count, replace=True, random_state=42)
        balanced_parts.append(class_df)

    balanced_df = pd.concat(balanced_parts, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    X_balanced = balanced_df.drop(columns=['classification'])
    y_balanced = balanced_df['classification']

    print("\nTraining Class Distribution After Balancing:")
    print(y_balanced.value_counts().sort_index())

    return X_balanced, y_balanced


def style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

df = pd.read_csv("Malware dataset.csv")

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)

print("\nClass Distribution:")
print(df['classification'].value_counts())

print("\nMissing Values:", df.isnull().sum().sum())

print("\nSample Data:")
print(df.head(3))

fig, ax = plt.subplots(dpi=600)

colors = ['#4C72B0', '#DD8452']  

df['classification'].value_counts().plot(
    kind='bar',
    ax=ax,
    color=colors,
    edgecolor='black'
)
for p in ax.patches:
    ax.annotate(str(p.get_height()),
                (p.get_x() + 0.2, p.get_height() + 1000),
                fontweight='bold',
                fontname='Times New Roman',
                fontsize=14)

ax.set_title("Class Distribution", fontsize=20,
             fontweight='bold', fontname='Times New Roman',pad=16)

ax.set_xlabel("Class", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.set_ylabel("Count", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.tick_params(axis='both', labelsize=16) 

for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

style_plot(ax)

plt.tight_layout()
save_plot(fig, "01_class_distribution.png")
plt.show()


df = df.drop(columns=['hash'], errors='ignore')
df = df.loc[:, df.nunique() > 1]

df['classification'] = df['classification'].map({'benign': 0, 'malware': 1})

X = df.drop('classification', axis=1)
y = df['classification']

feature_names = X.columns

X = pd.DataFrame(np.log1p(X), columns=feature_names)

X_plot = pd.DataFrame(
    StandardScaler().fit_transform(X),
    columns=feature_names
)

sample_features = X_plot.sample(min(35, len(X.columns)), axis=1, random_state=42)

axes = sample_features.hist(bins=10, edgecolor='black')
fig = axes.flatten()[0].figure

colors = plt.cm.tab20.colors

for i, ax in enumerate(axes.flatten()):
    if len(ax.patches) > 0:  
        for patch in ax.patches:
            patch.set_facecolor(colors[i % len(colors)])

    ax.set_title(ax.get_title(), fontweight='bold', fontname='Times New Roman')
    style_plot(ax)


plt.tight_layout()
save_plot(fig, "02_feature_histograms.png")
plt.show()

corr = X_plot.corr()

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

heatmap = sns.heatmap(
    corr,
    cmap='RdBu_r',         
    center=0,              
    vmin=-1, vmax=1,       
    linewidths=0.3,
    square=True,
    annot=True,
    fmt=".2f",
    annot_kws={
        "fontsize": 6.5,
        "fontweight": "bold",
        "fontname": "Times New Roman"
    },
    cbar_kws={"shrink": 1},
    ax=ax
)
ax.set_title("Correlation Heatmap",
             fontsize=18, fontweight='bold', fontname='Times New Roman')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90,
                   fontweight='bold', fontname='Times New Roman', fontsize=10)

ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   fontweight='bold', fontname='Times New Roman', fontsize=10)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


cbar = heatmap.collections[0].colorbar

cbar.ax.tick_params(labelsize=12)

for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

plt.tight_layout()
save_plot(fig, "03_correlation_heatmap.png")
plt.show()

top_corr = corr.abs().unstack().sort_values(ascending=False)
top_corr = top_corr[top_corr < 1].head(10)

print("\nTop Correlated Feature Pairs:\n")
print(top_corr)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

X_train, y_train = balance_training_data(X_train, y_train)

scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names,
    index=X_train.index
)
X_val = pd.DataFrame(
    scaler.transform(X_val),
    columns=feature_names,
    index=X_val.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names,
    index=X_test.index
)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val   = torch.tensor(X_val.values, dtype=torch.float32)
X_test  = torch.tensor(X_test.values, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.long)
y_val   = torch.tensor(y_val.values, dtype=torch.long)
y_test  = torch.tensor(y_test.values, dtype=torch.long)

class MOAPO:
    def __init__(self, n_features, pop_size=20, iterations=100):
        self.n_features = n_features
        self.pop_size = pop_size
        self.iterations = iterations
        self.archive = []
        self.best_scores = []
        self.avg_scores = []

    def initialize_population(self):
        return np.random.rand(self.pop_size, self.n_features)

    def binary_conversion(self, X):
        X_clipped = np.clip(X, -20, 20)  
        sigmoid = 1 / (1 + np.exp(-X_clipped))
        return (sigmoid > np.random.rand(*X.shape)).astype(int)

    def fitness(self, individual, X, y, X_val_fixed, y_val_fixed):
        selected = np.where(individual == 1)[0]
    
        if len(selected) == 0:
            return 0.0, 1.0
    
        X_sel = X[:, selected]
        X_val_sel = X_val_fixed[:, selected]
    
        try:
            clf = LogisticRegression(
                max_iter=1000,
                solver='liblinear'
            )
            clf.fit(X_sel, y.numpy())
    
            preds = clf.predict(X_val_sel)
            acc = (preds == y_val_fixed.numpy()).mean()
    
        except Exception as e:
            print(f"Feature subset evaluation failed: {e}")
            acc = 0.0
    
        feature_ratio = len(selected) / self.n_features
        return acc, feature_ratio

    # ---------------- DOMINANCE ---------------- #
    def dominates(self, a, b):
        return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])

    # ---------------- NON-DOMINATED SORT ---------------- #
    def non_dominated_sort(self, scores):
        n = len(scores)
        S = [[] for _ in range(n)]
        count = [0] * n
        fronts = [[]]
    
        for p in range(n):
            for q in range(n):
                if self.dominates(scores[p], scores[q]):
                    S[p].append(q)
                elif self.dominates(scores[q], scores[p]):
                    count[p] += 1
    
            if count[p] == 0:
                fronts[0].append(p)
    
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
    
            for p in fronts[i]:
                for q in S[p]:
                    count[q] -= 1
                    if count[q] == 0:
                        next_front.append(q)
    
            i += 1
            fronts.append(next_front)
    
        return fronts[:-1]

    # ---------------- CROWDING DISTANCE ---------------- #
    def crowding_distance(self, front, scores):
        if len(front) <= 2:
            return np.ones(len(front)) * float('inf')
    
        distance = np.zeros(len(front))
    
        for m in range(2):
            values = np.array([scores[i][m] for i in front])
            sorted_idx = np.argsort(values)
    
            distance[sorted_idx[0]] = float('inf')
            distance[sorted_idx[-1]] = float('inf')
    
            for i in range(1, len(front) - 1):
                prev_v = values[sorted_idx[i - 1]]
                next_v = values[sorted_idx[i + 1]]
                distance[sorted_idx[i]] += next_v - prev_v
    
        return distance

    # ---------------- ARCHIVE UPDATE ---------------- #
    def update_archive(self, population, scores):
        combined = list(zip(population, scores))
    
        if len(self.archive) > 0:
            combined = self.archive + combined
    
        all_scores = [s for _, s in combined]
        fronts = self.non_dominated_sort(all_scores)
    
        new_archive = []
    
        for front in fronts:
            if len(new_archive) + len(front) > self.pop_size:
                distances = self.crowding_distance(front, all_scores)
                sorted_idx = np.argsort(-np.array(distances))
    
                for i in sorted_idx:
                    if len(new_archive) < self.pop_size:
                        new_archive.append(combined[front[i]])
                break
            else:
                for idx in front:
                    new_archive.append(combined[idx])
    
        self.archive = new_archive
    # ---------------- LEADER ---------------- #
    def select_leader(self):
        if len(self.archive) == 0:
            return None
    
        scores = [s for _, s in self.archive]
        front = list(range(len(self.archive)))
    
        distances = self.crowding_distance(front, scores)
        distances = np.array(distances)
    
        finite = np.isfinite(distances)
    
        if np.any(finite):
            probs = distances.copy()
            probs[~finite] = 0
    
            if probs.sum() == 0:
                return self.archive[np.random.randint(len(self.archive))][0]
    
            probs = probs / probs.sum()
            return self.archive[np.random.choice(len(self.archive), p=probs)][0]
    
        return self.archive[np.random.randint(len(self.archive))][0]
    # ---------------- LEVY FLIGHT ---------------- #
    def levy_flight(self, dim):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))

        return step

    # ---------------- UPDATE POSITION ---------------- #
    def update_position(self, X, leader, iteration):
        new_X = []

        t = iteration / self.iterations
        A = 2 * (1 - t**2)  
        exploration_prob = 0.7 * (1 - t)
        defense_prob = 0.3 + 0.4 * t 

        for i in range(self.pop_size):
            r = np.random.rand()

            if r < exploration_prob:
                # -------- EXPLORATION (Levy + random walk) -------- #
                step = self.levy_flight(self.n_features)
                new_pos = X[i] + A * step

            else:
                # -------- EXPLOITATION (leader-based) -------- #
                rand_agent = X[np.random.randint(self.pop_size)]
                new_pos = X[i] + A * (leader - rand_agent)

            # -------- PORCUPINE DEFENSE -------- #
            if np.random.rand() < defense_prob:
                spike = np.random.randn(self.n_features) * 0.1
                new_pos += spike

            new_X.append(new_pos)

        X = np.array(new_X)

        # -------- MUTATION -------- #
        mutation_rate = 0.05 * (1 - t)
        mutation_mask = np.random.rand(*X.shape) < mutation_rate
        X = np.where(mutation_mask, np.random.rand(*X.shape), X)

        return X

    # ---------------- OPTIMIZATION ---------------- #
    def optimize(self, X, y, X_val, y_val):
        pop = self.initialize_population()
    
        print("\n========== MOAPO ==========")
        pbar = tqdm(range(self.iterations), desc="MOAPO Optimization", ncols=100)
    
        for it in pbar:
            binary_pop = self.binary_conversion(pop)
    
            idx = np.random.choice(len(X), size=min(100000, len(X)), replace=False)
            X_sub = X[idx]
            y_sub = y[idx]
    
            scores = [
                self.fitness(ind, X_sub, y_sub, X_val, y_val)
                for ind in binary_pop
            ]
    
            acc_values = [s[0] for s in scores]
    
            best_acc = max(acc_values)
            avg_acc = np.mean(acc_values)
    
            self.best_scores.append(best_acc)
            self.avg_scores.append(avg_acc)
    
            self.update_archive(binary_pop, scores)
    
            leader = self.select_leader()
            pop = self.update_position(pop, leader, it)
    
            pbar.set_postfix({
                "Best Acc": f"{best_acc:.4f}",
                "Avg Acc": f"{avg_acc:.4f}",
                "Archive": len(self.archive)
            })
    
            gc.collect()
    
        best_solution = max(self.archive, key=lambda x: x[1][0])
        selected_idx = np.where(best_solution[0] == 1)[0]
    
        print("\nBest Accuracy:", best_solution[1][0])
        print("Feature Ratio:", best_solution[1][1])
        print("Selected Features:", len(selected_idx))
    
        return selected_idx

all_features = feature_names.tolist() 

print("\n===== ALL FEATURES =====")
for i, feat in enumerate(all_features):
    print(f"{i+1}. {feat}")
moapo = MOAPO(n_features=X_train.shape[1])
selected_features = moapo.optimize(
    X_train.numpy(), y_train,
    X_val.numpy(), y_val
)
selected_feature_names = feature_names[selected_features].tolist()
print("\n===== SELECTED FEATURES =====")
for i, feat in enumerate(selected_feature_names):
    print(f"{i+1}. {feat}")

print(f"\nTotal features: {len(all_features)}")
print(f'Feature Selected names :{selected_feature_names}')

fig, ax = plt.subplots(dpi=600)

ax.plot(moapo.best_scores, linewidth=2.5, color='#1f77b4', label='Best Fitness')   # Blue
ax.plot(moapo.avg_scores, linewidth=2.5, linestyle='--', color='#ff7f0e', label='Average Fitness')  # Orange

ax.set_title("MOAPO Convergence Curve",
             fontsize=18, fontweight='bold', fontname='Times New Roman')

ax.set_xlabel("Iteration",
              fontsize=14, fontweight='bold', fontname='Times New Roman')

ax.set_ylabel("Fitness",
              fontsize=14, fontweight='bold', fontname='Times New Roman')

ax.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})

ax.tick_params(axis='both', labelsize=12)

for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

ax.grid(True, linestyle='--', alpha=0.4)

style_plot(ax)

plt.tight_layout()
save_plot(fig, "04_moapo_convergence_curve.png")
plt.show()

original = X.shape[1]
selected = len(selected_features)

fig, ax = plt.subplots(dpi=600)

labels = ["Original", "Selected"]
values = [original, selected]

ax.bar(labels, values,
       color=['#4CAF50', '#E53935'],
       edgecolor='black',
       linewidth=1.5)

for i, v in enumerate(values):
    ax.text(i, v + 1, str(v),
            ha='center',
            fontweight='bold',
            fontname='Times New Roman',fontsize=16)

ax.set_title("Feature Selection",
             fontsize=20, fontweight='bold', fontname='Times New Roman')

ax.set_ylabel("Number of Features",
              fontsize=18, fontweight='bold', fontname='Times New Roman')

ax.tick_params(axis='both', labelsize=16)

for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

ax.grid(axis='y', linestyle='--', alpha=0.4)

style_plot(ax)

plt.tight_layout()
save_plot(fig, "05_feature_selection.png")
plt.show()


X_train_sel = X_train.numpy()[:, selected_features]
X_val_sel   = X_val.numpy()[:, selected_features]
X_test_sel  = X_test.numpy()[:, selected_features]

X_train_sel = torch.tensor(X_train_sel, dtype=torch.float32)
X_val_sel   = torch.tensor(X_val_sel, dtype=torch.float32)
X_test_sel  = torch.tensor(X_test_sel, dtype=torch.float32)

# ---------------- MODEL ---------------- #
class HMSCA_PGN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.scale1 = nn.Linear(input_dim, hidden_dim * 2)
        self.scale2 = nn.Linear(input_dim, hidden_dim)

    
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # Sparse gating
        self.sparse_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        s1 = torch.relu(self.scale1(x))   
        s2 = torch.relu(self.scale2(x))  
    
        s1 = self.proj(s1)               
    
        seq = torch.stack([s1, s2], dim=1)   
    
        attn_out, _ = self.attn(seq, seq, seq)
    
        gate = self.sparse_gate(attn_out)
        sparse_out = attn_out * gate
    
        out, _ = self.lstm(sparse_out)
    
        return self.fc(out[:, -1, :])

# ---------------- AOSO ---------------- #
class AOSO:
    def __init__(self, param_space, pop_size=10, iterations=10):
        self.param_space = param_space
        self.pop_size = pop_size
        self.iterations = iterations
        self.best_scores = []
        self.avg_scores = []
        self.best_accs = []
        self.avg_accs = []
        self.best_losses = []

    def initialize(self):
        return np.array([
            [
                np.random.uniform(*self.param_space['lr']),
                np.random.uniform(*self.param_space['hidden'])
            ]
            for _ in range(self.pop_size)
        ])

    def clip(self, pop):
        pop[:, 0] = np.clip(pop[:, 0], *self.param_space['lr'])
        pop[:, 1] = np.clip(pop[:, 1], *self.param_space['hidden'])
        return pop

    def fitness(self, params, X_train, y_train, X_val, y_val):
        lr, hidden = params
        hidden = int(hidden)

        hidden = (hidden // 4) * 4
        if hidden < 4:
            hidden = 4

        model = HMSCA_PGN(X_train.shape[1], hidden_dim=hidden)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
    
        for _ in range(3):
            model.train()
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
    
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            loss_val = criterion(logits, y_val).item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_val).float().mean().item()

        score = acc - 0.05 * loss_val
    
        del model
        gc.collect()
    
        return score, acc, loss_val
    
    def optimize(self, X_train, y_train, X_val, y_val):
        pop = self.initialize()
        scores = np.zeros(self.pop_size)
        accs = np.zeros(self.pop_size)
        losses = np.zeros(self.pop_size)
    
        g_best = None
        g_best_score = -1
    
        pbar = tqdm(range(self.iterations), desc="AOSO Optimization", ncols=100)
    
        for it in pbar:
            for i in range(self.pop_size):
                score, acc, loss = self.fitness(pop[i], X_train, y_train, X_val, y_val)
                scores[i] = score
                accs[i] = acc
                losses[i] = loss
    
                if scores[i] > g_best_score:
                    g_best_score = scores[i]
                    g_best = pop[i].copy()
    
            iter_best_score = np.max(scores)
            iter_avg_score = np.mean(scores)
            iter_best_acc = np.max(accs)
            iter_avg_acc = np.mean(accs)
            iter_best_loss = np.min(losses)

            self.best_scores.append(iter_best_score)
            self.avg_scores.append(iter_avg_score)
            self.best_accs.append(iter_best_acc)
            self.avg_accs.append(iter_avg_acc)
            self.best_losses.append(iter_best_loss)
    
            pbar.set_postfix({
                "Best Score": f"{iter_best_score:.4f}",
                "Avg Score": f"{iter_avg_score:.4f}",
                "Best Acc": f"{iter_best_acc:.4f}",
                "Best Loss": f"{iter_best_loss:.4f}"
            })
    
            alpha = 1 - it / self.iterations
    
            for i in range(self.pop_size):
                r = np.random.rand(2)
                oscillation = np.sin(2 * np.pi * np.random.rand())
    
                pop[i] = pop[i] \
                         + alpha * r * (g_best - pop[i]) \
                         + 0.1 * oscillation * (np.random.rand(2) - pop[i])
    
            pop = self.clip(pop)
    
        hidden = int(g_best[1])
        hidden = (hidden // 4) * 4
        if hidden < 4:
            hidden = 4
    
        return {'lr': g_best[0], 'hidden': hidden}


aoso = AOSO({'lr': (1e-4, 5e-3), 'hidden': (32, 160)}, pop_size=12, iterations=15)
best_params = aoso.optimize(X_train_sel, y_train, X_val_sel, y_val)

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

ax.plot(aoso.best_scores, linewidth=2.5, color='#8338ec', label='Best Score')

ax.set_title("AOSO Optimization", fontsize=20,
             fontweight='bold', fontname='Times New Roman')

ax.set_xlabel("Iteration", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.set_ylabel("Optimization Score", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.tick_params(axis='both', labelsize=16)

for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})

plt.tight_layout()
save_plot(fig, "06_aoso_optimization.png")
plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_sel = X_train_sel.to(device)
X_val_sel   = X_val_sel.to(device)
X_test_sel  = X_test_sel.to(device)

y_train = y_train.to(device)
y_val   = y_val.to(device)
y_test  = y_test.to(device)

model = HMSCA_PGN(X_train_sel.shape[1], hidden_dim=best_params['hidden']).to(device)

optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)

class_counts = np.bincount(y_train.cpu().numpy())
weights = torch.tensor(
    [len(y_train) / c for c in class_counts],
    dtype=torch.float32
).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5
)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_acc = 0.0
best_epoch = 0
patience = 15
counter = 0
best_state_dict = None


# ---------------- TRAINING ---------------- #
pbar = tqdm(range(50), desc="Training", ncols=100)

for epoch in pbar:

    model.train()
    optimizer.zero_grad()

    out = model(X_train_sel)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    preds_train = torch.argmax(out, dim=1)
    acc_train = accuracy_score(
        y_train.cpu().numpy(),
        preds_train.cpu().numpy()
    )
    train_accs.append(acc_train)

    # ---- VALIDATION ----
    model.eval()
    with torch.no_grad():
        out_val = model(X_val_sel)
        loss_val = criterion(out_val, y_val)

        preds_val = torch.argmax(out_val, dim=1)
        acc_val = accuracy_score(
            y_val.cpu().numpy(),
            preds_val.cpu().numpy()
        )

    val_losses.append(loss_val.item())
    val_accs.append(acc_val)

    scheduler.step(acc_val)

    # ---- SAVE BEST ----
    if acc_val > best_acc:
        best_acc = acc_val
        best_epoch = epoch
        counter = 0
        best_state_dict = copy.deepcopy(model.state_dict())
    else:
        counter += 1

    # ---- EARLY STOPPING ----
    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

    pbar.set_postfix({
        "Train Acc": f"{acc_train:.4f}",
        "Val Acc": f"{acc_val:.4f}",
        "Best": f"{best_acc:.4f}"
    })

print(f"\nBest Model at Epoch {best_epoch} | Val Acc = {best_acc:.4f}")

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)


# ---------------- LOSS CURVE ---------------- #
fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

plt.plot(train_losses, label='Train Loss', color='#7014f2', linewidth=2.5)
plt.plot(val_losses, label='Test Loss', color='#00f59b', linestyle='--', linewidth=2.5)

plt.title("Loss Curve", fontsize=20, fontweight='bold', fontname='Times New Roman')
plt.xlabel("Epochs", fontsize=18, fontweight='bold', fontname='Times New Roman')
plt.ylabel("Loss", fontsize=18, fontweight='bold', fontname='Times New Roman')

plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold','size':16})
plt.grid(True, linestyle='--', alpha=0.4)

plt.xticks(fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.yticks(fontsize=16, fontweight='bold', fontname='Times New Roman')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
save_plot(fig, "07_loss_curve.png")
plt.show()

# ---------------- ACCURACY CURVE ---------------- #
fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

plt.plot(train_accs, label='Train Accuracy', color='#7014f2', linewidth=2.5)
plt.plot(val_accs, label='Test Accuracy', color='#00f59b', linestyle='--', linewidth=2.5)

plt.title("Accuracy Curve", fontsize=20, fontweight='bold', fontname='Times New Roman')
plt.xlabel("Epochs", fontsize=18, fontweight='bold', fontname='Times New Roman')
plt.ylabel("Accuracy", fontsize=18, fontweight='bold', fontname='Times New Roman')

plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold','size':16})
plt.grid(True, linestyle='--', alpha=0.4)

plt.xticks(fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.yticks(fontsize=16, fontweight='bold', fontname='Times New Roman')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
save_plot(fig, "08_accuracy_curve.png")
plt.show()

model.eval()
with torch.no_grad():
    outputs = model(X_test_sel)
    preds = torch.argmax(outputs, dim=1)
    probs_all = torch.softmax(outputs, dim=1).cpu().numpy()

y_test_np = y_test.cpu().numpy()
preds_np = preds.cpu().numpy()
y_test_bin = np.column_stack([
    (y_test_np == 0).astype(int),
    (y_test_np == 1).astype(int)
])

acc = accuracy_score(y_test_np, preds_np)
f1 = f1_score(y_test_np, preds_np)

print(f"Final Accuracy: {acc:.4f}")
print(f"Final F1 Score: {f1:.4f}")


# ---------------- CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_test_np, preds_np)

fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='RdPu',
            linewidths=1,
            linecolor='black',
            cbar=False, 
            annot_kws={"size": 16, "weight": "bold", "family": "Times New Roman"},
            ax=ax)

ax.set_title("Confusion Matrix", fontsize=20,
             fontweight='bold', fontname='Times New Roman')

ax.set_xlabel("Predicted", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.set_ylabel("Actual", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.tick_params(axis='both', labelsize=16)

for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Times New Roman')

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
save_plot(fig, "09_confusion_matrix.png")
plt.show()
# ---------------- ROC ---------------- #

class_names = ['Benign', 'Malware']
class_colors = ['#1f77b4', '#d62728']

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs_all[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2.5, color=class_colors[i],
            label=f"{class_name} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

ax.set_title("ROC Curve", fontsize=20,
             fontweight='bold', fontname='Times New Roman')

ax.set_xlabel("False Positive Rate", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.set_ylabel("True Positive Rate", fontsize=18,
              fontweight='bold', fontname='Times New Roman')
plt.xticks(fontweight='bold',fontsize=16,fontname='Times New Roman')
plt.yticks(fontweight='bold',fontsize=16,fontname='Times New Roman')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
save_plot(fig, "10_classwise_roc_curve.png")
plt.show()
# ---------------- PR CURVE ---------------- #

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

for i, class_name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], probs_all[:, i])
    ap = average_precision_score(y_test_bin[:, i], probs_all[:, i])
    ax.plot(recall, precision, linewidth=2.5, color=class_colors[i],
            label=f"{class_name} (AP = {ap:.4f})")

ax.set_title("Precision-Recall Curve", fontsize=20,
             fontweight='bold', fontname='Times New Roman')

ax.set_xlabel("Recall", fontsize=18,
              fontweight='bold', fontname='Times New Roman')

ax.set_ylabel("Precision", fontsize=18,
              fontweight='bold', fontname='Times New Roman')
plt.xticks(fontweight='bold',fontsize=16,fontname='Times New Roman')
plt.yticks(fontweight='bold',fontsize=16,fontname='Times New Roman')
ax.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
save_plot(fig, "11_classwise_precision_recall_curve.png")
plt.show()
# ---------------- FINAL METRICS ---------------- #
precision_val = precision_score(y_test_np, preds_np)
recall_val = recall_score(y_test_np, preds_np)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [acc, precision_val, recall_val, f1]

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

ax.bar(metrics, values,
       color=['#264653', '#e76f51', '#2a9d8f', '#e9c46a'],
       edgecolor='black')

for i, v in enumerate(values):
    ax.text(i, v + 0.001, f"{v:.3f}",
            ha='center',
            fontweight='bold',
            fontname='Times New Roman',fontsize=16)

ax.set_title("Final Performance Metrics", fontsize=20,
             fontweight='bold', fontname='Times New Roman')
plt.xticks(fontweight='bold',fontsize=16,fontname='Times New Roman')
plt.yticks(fontweight='bold',fontsize=16,fontname='Times New Roman')
plt.xlabel('Metrics',fontweight='bold',fontsize=18,fontname='Times New Roman')
plt.ylabel('Score',fontweight='bold',fontsize=18,fontname='Times New Roman')
ax.set_ylim(0.95, 1.005)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
save_plot(fig, "12_final_performance_metrics.png")
plt.show()

# ---------------- CLASS-WISE SHAP ---------------- #
plot_classwise_shap(
    model,
    X_train_sel,
    X_test_sel,
    selected_feature_names,
    class_names
)
