import torch
from torch.utils.data import Dataset
import chess.pgn
import numpy as np
import random

class ChessPGNDataset(Dataset):
    def __init__(self, pgn_file_path, move_map,  max_games=1000):
        self.move_map = move_map
        self.positions = [] 
        with open(pgn_file_path) as opened_file:
            self._load_data(opened_file, max_games)

    def _load_data(self, file_handle, max_games=1000):
        for _ in range(max_games):
            game = chess.pgn.read_game(file_handle)
            if game is None: break
            sampled = self.sample_game_positions(game, num_samples=8)
            self.positions.extend(sampled)
    
    def sample_game_positions(self, game, num_samples=8):
        all_states = []
        board = game.board()
        for move in game.mainline_moves():
            all_states.append((board.fen(), move, board.turn))
            board.push(move)
        return random.sample(all_states, min(len(all_states), num_samples))

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen, move, turn = self.positions[idx]
        board = chess.Board(fen)
        
        if turn == chess.BLACK:
            board = board.mirror()
            move = chess.Move(chess.square_mirror(move.from_square), 
                              chess.square_mirror(move.to_square),
                              promotion=move.promotion)

        tensor = self.board_to_tensor(board)
        move_idx = self.move_map.get(move.uci(), 0)
        
        return torch.from_numpy(tensor).float(), torch.tensor(move_idx, dtype=torch.long)

    def board_to_tensor(self, board):
        matrix = np.zeros((12, 8, 8), dtype=np.int8)
        piece_to_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for square, piece in board.piece_map().items():
            plane = piece_to_idx[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            matrix[plane][chess.square_rank(square)][chess.square_file(square)] = 1
        return matrix

