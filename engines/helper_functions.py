import numpy as np
from chess import Board
import chess
import random
import chess.pgn
import torch

def board_to_matrix(board: Board):
    # 8x8 cause a chess board is 8x8
    # 12 = number of unique pieces.
    matrix = np.zeros((12, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards 
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    return matrix


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)

def create_move_map():
    all_moves = set()
    
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            all_moves.add(chess.Move(from_sq, to_sq).uci())
            
            if chess.square_rank(to_sq) in [0, 7]:
                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    all_moves.add(chess.Move(from_sq, to_sq, promotion=promo).uci())
    
    sorted_moves = sorted(list(all_moves))
    return {move: i for i, move in enumerate(sorted_moves)}

def sample_game_positions(game, num_samples=5):
    positions = []
    board = game.board()
    all_states = []
    for move in game.mainline_moves():
        all_states.append((board.fen(), move))
        board.push(move)
    
    if len(all_states) >= num_samples:
        positions = random.sample(all_states, num_samples)
    else:
        positions = all_states 
    return positions

def fen_to_12_plane_matrix(fen):
    board = chess.Board(fen)
    matrix = np.zeros((12, 8, 8), dtype=np.int8)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_to_idx[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            
            row = chess.square_rank(square)
            col = chess.square_file(square)
            
            matrix[plane][row][col] = 1
            
    return matrix

def turn_board_to_tensor(board:Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

def get_legal_moves_mask(board: Board, move_map: dict, device: torch.device = None):
    
    mask = torch.zeros(len(move_map), dtype=torch.bool)
    
    for move in board.legal_moves:
        move_uci = move.uci()
        if move_uci in move_map:
            move_idx = move_map[move_uci]
            mask[move_idx] = True
    
    if device is not None:
        mask = mask.to(device)
    
    return mask

def apply_legal_moves_mask(policy_logits: torch.Tensor, legal_moves_mask: torch.Tensor):
    
    masked_logits = policy_logits.clone()
    
    if masked_logits.dim() == 1:
        masked_logits[~legal_moves_mask] = float('-inf')
    elif masked_logits.dim() == 2:
        masked_logits[:, ~legal_moves_mask] = float('-inf')
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {masked_logits.dim()}D")
    
    return masked_logits

def get_best_legal_move(board: Board, policy_logits: torch.Tensor, move_map: dict, 
                        reverse_move_map: dict, device: torch.device = None, 
                        temperature: float = 1.0, colour = None):
    
    legal_mask = get_legal_moves_mask(board, move_map, device)
    
    masked_logits = apply_legal_moves_mask(policy_logits.squeeze(0), legal_mask)
    
    probabilities = torch.softmax(masked_logits / temperature, dim=0)
    
    move_idx = torch.multinomial(probabilities, 1).item()
    
    move_uci = reverse_move_map[move_idx]
    move = chess.Move.from_uci(move_uci)
    
    return move, probabilities[move_idx].item()

def compute_masked_policy_loss(pred_policy: torch.Tensor, target_move_indices: torch.Tensor, 
                               boards: list, move_map: dict, device: torch.device = None):
    
    batch_size = pred_policy.shape[0]
    total_loss = 0.0
    
    for i in range(batch_size):
        legal_mask = get_legal_moves_mask(boards[i], move_map, device)
        
        masked_logits = apply_legal_moves_mask(pred_policy[i:i+1], legal_mask)
        
        masked_logits = masked_logits.squeeze(0)
        
        target_idx = target_move_indices[i]
        
        log_probs = torch.log_softmax(masked_logits, dim=0)
        loss = -log_probs[target_idx]
        
        total_loss += loss
    
    return total_loss / batch_size