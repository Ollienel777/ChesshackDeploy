from pathlib import Path
import csv

import chess
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from nn_model import board_to_tensor, ValueNet


# Paths
SRC_DIR = Path(__file__).resolve().parent              # .../my-chesshacks-bot/src
ROOT_DIR = SRC_DIR.parent                              # .../my-chesshacks-bot
DATA_CSV = ROOT_DIR / "data" / "sf_positions.csv"
OUT_PATH = SRC_DIR / "valuenet.pt"                     # ./valuenet.pt


class PositionDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.samples = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["fen"], float(row["sf_score"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, result = self.samples[idx]
        board = chess.Board(fen)
        x = board_to_tensor(board)                     # (13, 8, 8)
        y = torch.tensor([result], dtype=torch.float32)
        return x, y


def train(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PositionDataset(DATA_CSV)
    if len(dataset) == 0:
        raise ValueError(f"No samples in {DATA_CSV} – did you generate sf_positions.csv?")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ValueNet().to(device)

    out_path = Path(__file__).with_name("valuenet.pt")

    if out_path.exists():
        try:
            print("[INFO] Continuing training from existing weights...")
            state = torch.load(out_path, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            print(f"[WARN] Failed to load existing weights ({e}). Training from scratch instead.")
    else:
        print("[INFO] No existing weights found. Training from scratch.")
    
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()

        for x, y in loader:
            x = x.to(device)
            y = y.to(device).squeeze(1)               # (batch,)

            opt.zero_grad()
            preds = model(x).squeeze(1)               # (batch,)

            error = preds - y                         # (batch,)
            base_loss = error ** 2                   # per-sample MSE

            # weights in [1, 3]: positions with |eval|≈1 (big advantage/disadvantage)
            # get up to 3x the weight
            weights = 1.0 + 2.0 * torch.abs(y)       # (batch,)
            loss = (weights * base_loss).mean()
            # ---------------------------------------------------------------------------

            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    train(epochs=10)
