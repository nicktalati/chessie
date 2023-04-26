import chess
import chess.engine
import chess.svg
import pygame
import io
import random
import time
import sys
import tensorflow_probability as tfp
import tensorflow as tf
from csv_creator import material_count
from time_models import load_bayesian_nn_time_model
from complexity_model import get_complexity_scores

OFFSET_Y = 120


stockfish_path = "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
engine.configure({"Skill Level": 5})



WIDTH = HEIGHT = 500
PIECE_SIZE = WIDTH // 8 
LIGHT = (238, 238, 210)
DARK = (118, 150, 86)

def get_piece_name(piece):
    return f"{chess.COLOR_NAMES[piece.color]}_{chess.piece_name(piece.piece_type)}"

def load_images():
    pieces = {}
    for piece in chess.PIECE_TYPES:
        for color in chess.COLORS:
            piece_name = get_piece_name(chess.Piece(piece, color))
            image = pygame.image.load(f"assets/{piece_name}.png")
            image = pygame.transform.scale(image, (PIECE_SIZE, PIECE_SIZE))
            pieces[piece_name] = image
    return pieces

def draw_square_and_piece(screen, board, images, rank, file, selected_square=None):
    square = chess.square(file, rank)
    if square == selected_square:
        piece = None
    else:
        piece = board.piece_at(square)

    color = DARK if (file + rank) % 2 == 0 else LIGHT
    square_x = file * PIECE_SIZE
    square_y = (7 - rank) * PIECE_SIZE + OFFSET_Y / 2
    square_rect = pygame.Rect(square_x, square_y, PIECE_SIZE, PIECE_SIZE)
    pygame.draw.rect(screen, color, square_rect)
    if piece:
        piece_name = get_piece_name(piece)
        screen.blit(images[piece_name], square_rect)

def draw_board(screen, board, images, selected_square=None):
    for rank in range(8):
        for file in range(8):
            draw_square_and_piece(screen, board, images, rank, file, selected_square)

def random_move(board):
    moves = list(board.legal_moves)
    return random.choice(moves)

def get_user_promotion_piece(screen, images, color):
    PROMOTION_WIDTH = WIDTH // 2
    PROMOTION_HEIGHT = HEIGHT // 8
    PROMOTION_X = WIDTH // 4
    PROMOTION_Y = HEIGHT // 2 - PROMOTION_HEIGHT // 2

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
                    if PROMOTION_X <= x < PROMOTION_X + PROMOTION_WIDTH and PROMOTION_Y <= y < PROMOTION_Y + PROMOTION_HEIGHT:
                        idx = (x - PROMOTION_X) // (PROMOTION_WIDTH // 4)
                        promotion_pieces = ['q', 'r', 'b', 'n']
                        return promotion_pieces[idx]

        # Draw promotion options
        for i, piece_type in enumerate(['q', 'r', 'b', 'n']):
            piece = chess.Piece.from_symbol(piece_type)
            piece.color = color
            piece_name = get_piece_name(piece)
            location = pygame.Rect(PROMOTION_X + i * (PROMOTION_WIDTH // 4),
                                   PROMOTION_Y, PIECE_SIZE, PIECE_SIZE)
            screen.blit(images[piece_name], location)

        pygame.display.flip()

def render_game_clock(screen, font, white_time, black_time):
    clock_background_color = (0, 0, 0)
    clock_background_rect_white = pygame.Rect(0, HEIGHT + OFFSET_Y / 2 + 10, WIDTH, OFFSET_Y / 2)
    clock_background_rect_black = pygame.Rect(0, 0, WIDTH, OFFSET_Y / 2)
    pygame.draw.rect(screen, clock_background_color, clock_background_rect_white)
    pygame.draw.rect(screen, clock_background_color, clock_background_rect_black)

    white_text = font.render(f"White: {int(white_time // 1000 // 60)}:{int((white_time // 1000) % 60):02d}", True, (255, 255, 255))
    black_text = font.render(f"Black: {int(black_time // 1000 // 60)}:{int((black_time // 1000) % 60):02d}", True, (255, 255, 255))

    screen.blit(white_text, (WIDTH // 2 - white_text.get_width() // 2, HEIGHT + OFFSET_Y / 2 + 20))
    screen.blit(black_text, (WIDTH // 2 - black_text.get_width() // 2, 20))

def main():
    AI_MOVE_EVENT = pygame.USEREVENT + 1
    complexity_model = tf.keras.models.load_model('complexity_model/complexity_model.h5')
    time_model = load_bayesian_nn_time_model('bayesian_nn_time_model.h5')

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Chess")

    screen = pygame.display.set_mode((WIDTH, HEIGHT + OFFSET_Y))

    initial_time = 3 * 60 * 1000  # 3 minutes per player in milliseconds
    white_time = initial_time
    black_time = initial_time
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    start_time = pygame.time.get_ticks()  # Store the start time
    last_update_time = start_time

    images = load_images()
    board = chess.Board()
    dragging = False
    selected_piece = None
    selected_square = None
    delta_x, delta_y = 0, 0

    running = True
    ai_move_requested = False
    while running:
        clock.tick(60)  # Limit the frame rate to 60 FPS

        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - last_update_time  # Calculate elapsed time since the last update

        if board.turn == chess.WHITE:
            white_time -= elapsed_time
        else:
            black_time -= elapsed_time

        last_update_time = current_time  # Update the last_update_time for the next loop

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == AI_MOVE_EVENT:
                if ai_move_requested:
                    ai_move = engine.play(board, chess.engine.Limit(time=0.1)).move
                    print(f"AI move: {ai_move}")
                    board.push(ai_move)
                    if board.is_game_over():
                        print("Game Over")
                        print(board.result())
                    ai_move_requested = False
                    pygame.time.set_timer(AI_MOVE_EVENT, 0)  # Stop the AI_MOVE_EVENT timer

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
                    y -= OFFSET_Y // 2
                    file, rank = x // PIECE_SIZE, 7 - y // PIECE_SIZE
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.color == board.turn:
                        dragging = True
                        selected_piece = piece
                        selected_square = square
                        delta_x, delta_y = x % PIECE_SIZE, y % PIECE_SIZE

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging:
                    x, y = event.pos
                    y -= OFFSET_Y // 2
                    file, rank = x // PIECE_SIZE, 7 - y // PIECE_SIZE
                    target_square = chess.square(file, rank)

                    move = None
                    promotion_moves = [chess.Move(selected_square, target_square, promotion=piece) for piece in chess.PIECE_TYPES[1:]]
                    pseudo_legal_promotion_moves = [m for m in promotion_moves if board.is_pseudo_legal(m)]

                    if len(pseudo_legal_promotion_moves) > 0:
                        promotion_piece = get_user_promotion_piece(screen, images, board.turn)
                        move = [m for m in pseudo_legal_promotion_moves if m.promotion == chess.Piece.from_symbol(promotion_piece).piece_type][0]
                    else:
                        move = chess.Move(selected_square, target_square)

                    if move in board.legal_moves:
                        board.push(move)
                        if board.is_game_over():
                            print("Game Over")
                            print(board.result())
                        else:
                            ai_move_requested = True
                            fen = board.fen()
                            white_elo = (1500 - 1506.20) / 52.25
                            black_elo = (1500 - 1503.28) / 53.13
                            white_time_left = (white_time / 1000 - 119.43) / 53.13
                            black_time_left = (black_time / 1000 - 119.24) / 53.64
                            white_material, black_material = material_count(fen)
                            white_material = (white_material - 27.06) / 11.22
                            black_material = (black_material - 27.15) / 11.12
                            expected_loss = get_complexity_scores(fen, 1500, complexity_model)['cp_loss']
                            blunder_chance = get_complexity_scores(fen, 1500, complexity_model)['blunder_chance']
                            expected_loss = (expected_loss - 30.78) / 19.51
                            blunder_chance = (blunder_chance - 0.0450) / 0.088
                            time = time_model.predict([[white_elo,
                                                        black_elo,
                                                        white_time_left,
                                                        black_time_left,
                                                        white_material,
                                                        black_material,
                                                        expected_loss,
                                                        blunder_chance]])
                            time = time[0][0]
                            print(time)
                            pygame.time.set_timer(AI_MOVE_EVENT, int(1000 * time))  # Schedule the AI_MOVE_EVENT in 1000 ms (1 second)

                    dragging = False
                    selected_piece = None
                    selected_square = None
                    delta_x, delta_y = 0, 0

        draw_board(screen, board, images, selected_square)

        if dragging:
            x, y = pygame.mouse.get_pos()
            piece_name = f"{chess.COLOR_NAMES[selected_piece.color]}_{chess.piece_name(selected_piece.piece_type)}"
            screen.blit(images[piece_name], pygame.Rect(x - delta_x, y - delta_y, PIECE_SIZE, PIECE_SIZE))

        # Render the game clock
        render_game_clock(screen, font, white_time, black_time)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
