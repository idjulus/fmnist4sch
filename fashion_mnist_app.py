"""
FashionMNIST — интерактивное обучение нейросети
Запуск: streamlit run fashion_mnist_app.py

Зависимости:
    pip install streamlit torch torchvision plotly
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import time

st.set_page_config(
    page_title="FashionMNIST — Нейросеть",
    page_icon="🧠",
    layout="wide"
)

CLASSES = [
    "👕 Футболка", "👖 Брюки", "🧣 Свитер", "👗 Платье", "🧥 Пальто",
    "🥿 Сандалии", "👔 Рубашка", "👟 Кроссовки", "👜 Сумка", "👢 Ботинки"
]

# ─── Модель ────────────────────────────────────────────────────────────────────
class DynamicNet(nn.Module):
    """Полносвязная сеть с настраиваемым числом слоёв и нейронов."""
    def __init__(self, hidden_layers: int, neurons: int, dropout: float):
        super().__init__()
        sizes = [784] + [neurons] * hidden_layers + [10]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# ─── Загрузка данных (кешируется между перезапусками) ─────────────────────────
@st.cache_resource(show_spinner="Загружаем FashionMNIST…")
def load_datasets():
    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("./data", train=True,  download=True, transform=tf)
    test  = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    return train, test


# ─── Вспомогательные функции ───────────────────────────────────────────────────
def smooth(values: list, window: int = 10) -> list:
    out = []
    for i in range(len(values)):
        w = values[max(0, i - window): i + 1]
        out.append(sum(w) / len(w))
    return out


def arch_table(hidden_layers: int, neurons: int, dropout: float) -> str:
    rows = ["| Слой | Тип | Выход |", "|------|-----|-------|",
            "| 0 | Input → Flatten | 784 |"]
    for i in range(hidden_layers):
        rows.append(f"| {i+1} | Linear → ReLU → Dropout({dropout:.2f}) | {neurons} |")
    rows.append(f"| {hidden_layers+1} | Linear (выход) | 10 |")
    total = 784 * neurons + neurons
    for _ in range(hidden_layers - 1):
        total += neurons * neurons + neurons
    total += neurons * 10 + 10
    rows.append(f"\n**Параметров всего:** {total:,}")
    return "\n".join(rows)


# ─── Интерфейс ────────────────────────────────────────────────────────────────
st.title("🧠 Обучение нейросети — FashionMNIST")
st.caption("Настрой гиперпараметры и нажми **Обучить** — график loss обновляется в реальном времени.")

sidebar, main = st.columns([1, 3], gap="large")

# ── Панель гиперпараметров ─────────────────────────────────────────────────────
with sidebar:
    st.subheader("⚙️ Гиперпараметры")

    lr = st.select_slider(
        "Learning rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        value=0.001
    )
    epochs = st.slider("Эпохи", 1, 10, 1)
    batch_size = st.select_slider(
        "Размер батча",
        options=[32, 64, 128, 256, 512],
        value=32
    )
    hidden_layers = st.slider("Скрытых слоёв", 1, 4, 2)
    neurons = st.select_slider(
        "Нейронов в слое",
        options=[64, 128, 256, 512],
        value=64
    )
    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
    optimizer_name = st.selectbox("Оптимизатор", ["Adam", "SGD", "RMSprop"])
    subset_size = st.select_slider(
        "Размер обучающей выборки",
        options=[5_000, 10_000, 20_000, 60_000],
        value=5_000,
        help="5k ≈ 1 мин, 10k ≈ 2–3 мин, 60k ≈ 10–15 мин на CPU"
    )

    train_btn = st.button("🚀 Обучить", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📐 Архитектура")
    st.markdown(arch_table(hidden_layers, neurons, dropout))

# ── Основная область ──────────────────────────────────────────────────────────
with main:
    tab_train, tab_pred = st.tabs(["📈 Обучение", "🔍 Предсказания"])

    with tab_train:
        status    = st.empty()
        progress  = st.progress(0)
        chart     = st.empty()
        metrics   = st.empty()

    with tab_pred:
        pred_area = st.empty()
        if "model" not in st.session_state:
            pred_area.info("Сначала обучи модель на вкладке **Обучение**.")

# ─── Обучение ─────────────────────────────────────────────────────────────────
if train_btn:
    train_ds, test_ds = load_datasets()

    idx_tr = torch.randperm(len(train_ds))[:subset_size]
    idx_te = torch.randperm(len(test_ds))[:2000]
    train_loader = DataLoader(Subset(train_ds, idx_tr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(Subset(test_ds,  idx_te), batch_size=256)

    model = DynamicNet(hidden_layers, neurons, dropout)
    criterion = nn.CrossEntropyLoss()

    optimizers = {
        "Adam":    optim.Adam(model.parameters(),    lr=lr),
        "SGD":     optim.SGD(model.parameters(),     lr=lr, momentum=0.9),
        "RMSprop": optim.RMSprop(model.parameters(), lr=lr),
    }
    optimizer = optimizers[optimizer_name]

    batch_losses: list[float] = []
    epoch_stats: list[dict]   = []
    total_batches = epochs * len(train_loader)
    done = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

            epoch_loss   += loss.item()
            done         += 1
            batch_losses.append(loss.item())

            # Обновляем UI каждые 5 батчей
            if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total_batches - done)
                progress.progress(done / total_batches)
                status.info(
                    f"Эпоха **{epoch+1}/{epochs}** · батч {batch_idx+1}/{len(train_loader)} · "
                    f"loss: {loss.item():.4f} · прошло: {elapsed:.0f}с · осталось: ~{max(0, eta):.0f}с"
                )

                # График
                sm = smooth(batch_losses, window=8)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=batch_losses,
                    mode="lines",
                    line=dict(color="rgba(124,106,255,0.25)", width=1),
                    name="Loss (батч)"
                ))
                fig.add_trace(go.Scatter(
                    y=sm,
                    mode="lines",
                    line=dict(color="#7c6aff", width=2.5),
                    name="Loss (сглаженный)"
                ))
                # Вертикальные линии после каждой эпохи
                for e in range(epoch):
                    x_ep = (e + 1) * len(train_loader)
                    if x_ep < len(batch_losses):
                        fig.add_vline(
                            x=x_ep,
                            line_dash="dash",
                            line_color="rgba(255,106,155,0.6)",
                            annotation_text=f"Эпоха {e+1}",
                            annotation_font_size=11
                        )
                fig.update_layout(
                    xaxis_title="Батч",
                    yaxis_title="Cross-Entropy Loss",
                    height=360,
                    margin=dict(l=40, r=20, t=20, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                chart.plotly_chart(fig, use_container_width=True)

        # ── Валидация ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        total   = 0
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X)
                val_loss += criterion(out, y).item()
                correct  += (out.argmax(1) == y).sum().item()
                total    += y.size(0)

        epoch_stats.append({
            "epoch":      epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "val_loss":   val_loss   / len(test_loader),
            "accuracy":   correct / total * 100,
        })

        rows = ["| Эпоха | Train Loss | Val Loss | Accuracy |",
                "|-------|-----------|----------|----------|"]
        for s in epoch_stats:
            rows.append(
                f"| {s['epoch']} | {s['train_loss']:.4f} | {s['val_loss']:.4f} | {s['accuracy']:.1f}% |"
            )
        metrics.markdown("\n".join(rows))

    elapsed = time.time() - t0
    progress.progress(1.0)
    final_acc = epoch_stats[-1]["accuracy"]
    status.success(f"✅ Обучение завершено! Accuracy: **{final_acc:.1f}%** · Время: {elapsed:.0f}с")

    # Сохраняем в session_state для предсказаний
    st.session_state["model"]       = model
    st.session_state["test_loader"] = test_loader

    pred_area.empty()
    with tab_pred:
        pred_btn = st.button("🤔 Предсказать", type="secondary", width=200)
        st.session_state["pred_btn"] = pred_btn

# ─── Предсказания ──────────────────────────────────────────────────────────────
if "pred_btn" in st.session_state:
    if pred_btn:
        pred_area.empty()
        model = st.session_state["model"]
        model.eval()
        test_loader = st.session_state["test_loader"]
        X_s, y_s = next(iter(test_loader))
        id_samples = torch.randint(0,len(X_s),(16,))
        X_s, y_s = X_s[id_samples], y_s[id_samples]
    
        with torch.no_grad():
            preds = model(X_s).argmax(1)
    
        cols = 4
        rows_n = 4
        titles = [
            f"{'✅' if preds[i] == y_s[i] else '❌'} {CLASSES[preds[i].item()].split(' ', 1)[1]}"
            for i in range(16)
        ]
        fig2 = sp.make_subplots(rows=rows_n, cols=cols, subplot_titles=titles)
    
        for i in range(16):
            img = X_s[i].squeeze().numpy()
            r, c = divmod(i, cols)
            fig2.add_trace(
                go.Heatmap(z=np.flipud(img), colorscale="gray",
                           showscale=False, zmin=0, zmax=1),
                row=r + 1, col=c + 1
            )
            fig2.update_xaxes(showticklabels=False, row=r + 1, col=c + 1)
            fig2.update_yaxes(showticklabels=False, row=r + 1, col=c + 1, scaleanchor="x", constrain="domain")
            # fig2.update_layout(yaxis=dict(scaleanchor="x", constrain="domain"))
    
        correct_count = (preds == y_s).sum().item()
        fig2.update_layout(
            title_text=f"16 примеров из тестовой выборки — верно: {correct_count}/16",
            height=700,
            margin=dict(l=10, r=10, t=60, b=10)
        )
        with tab_pred:
            pred_area.plotly_chart(fig2, use_container_width=True)
