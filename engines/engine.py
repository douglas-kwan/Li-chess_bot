import sys
import os
import chess
import torch
import helper_functions as helper
from model import ChessNet

engine_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(engine_dir, 'engine.log')
import logging
logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s %(message)s')

logging.info("Engine starting up...")    

logging.info("Loading ML model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

MOVE_MAP = helper.create_move_map()
MOVE_MAP_LENGTH = len(MOVE_MAP)
REVERSE_MOVE_MAP = {v: k for k, v in MOVE_MAP.items()}
logging.info(f"Move map created with {MOVE_MAP_LENGTH} moves")

model = ChessNet(MOVE_MAP_LENGTH)
model_path = os.path.join(engine_dir, 'models', 'supervised_learning_chess_model_1.pth')
logging.info(f"Loading model from: {model_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
logging.info("Model loaded successfully")

def get_best_move(board, colour):
    """
    Get the best move from the neural network model.
    """
    try:
        working_board = chess.Board(board.fen())
        
        if colour == chess.BLACK:
            working_board = working_board.mirror()
        
        with torch.no_grad():
            board_tensor = helper.turn_board_to_tensor(working_board).to(device)
            policy, value = model(board_tensor)
        
        move, confidence = helper.get_best_legal_move(
            board=working_board,
            policy_logits=policy,
            move_map=MOVE_MAP,
            reverse_move_map=REVERSE_MOVE_MAP,
            device=device,
            temperature=1.0
        )
        
        if colour == chess.BLACK:
            move = chess.Move(chess.square_mirror(move.from_square), 
                            chess.square_mirror(move.to_square),
                            promotion=move.promotion)
        
        logging.info(f"get_best_move: colour={colour}, move={move}, confidence={confidence:.4f}")
        return move
    except Exception as e:
        logging.error(f"Error in get_best_move: {e}")
        logging.error(f"Exception type: {type(e).__name__}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def main():
    board = chess.Board()
    colour = None
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            command = line.strip()
            logging.debug(f"Received: {command}")
            tokens = command.split()
            if not tokens:
                continue

            if tokens[0] == "uci":
                print("id name MyNeuralBot_v1")
                print("id author YourName")
                print("uciok")
            
            elif tokens[0] == "isready":
                print("readyok")

            elif tokens[0] == "ucinewgame":
                board.reset()

            elif tokens[0] == "position":
                if "startpos" in tokens:
                    board = chess.Board()
                elif "fen" in tokens:
                    fen_str = " ".join(tokens[tokens.index("fen")+1 : tokens.index("fen")+7])
                    board = chess.Board(fen_str)
                
                if "moves" in tokens:
                    move_list = tokens[tokens.index("moves")+1:]
                    for m in move_list:
                        board.push_uci(m)

            elif tokens[0] == "go":
                if board.turn == chess.WHITE:
                    logging.info("I am playing as WHITE")
                    colour = chess.WHITE
                else:
                    logging.info("I am playing as BLACK")
                    colour = chess.BLACK
                
                move = get_best_move(board, colour)
                if move:
                    print(f"bestmove {move.uci()}")
                    logging.info(f"Sent: bestmove {move.uci()}")
                else:
                    logging.warning("Failed to get move, resigning")
                    print("bestmove 0000")

            elif tokens[0] == "quit":
                logging.info("Quitting engine.")
                break

            sys.stdout.flush() 

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            logging.error(f"Exception type: {type(e).__name__}")
            import traceback
            logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()