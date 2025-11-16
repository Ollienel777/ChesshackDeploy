# my-chesshacks-bot/src/nn_model.py
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(13, 8, 8, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        idx = (0 if piece.color == chess.WHITE else 6) + (piece.piece_type - 1)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[idx, 7-rank, file] = 1.0
    planes[12,:,:] = 1.0 if board.turn == chess.WHITE else -1.0
    return planes


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
