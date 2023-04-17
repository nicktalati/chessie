import chess
import chess.svg
import pygame
import io
import random
import time
import sys


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

def draw_square_and_piece(screen, board, images, rank, file):
    piece = board.piece_at(chess.square(file, rank))
    color = DARK if (file + rank) % 2 == 0 else LIGHT
    square_x = file * PIECE_SIZE
    square_y = (7 - rank) * PIECE_SIZE
    square = pygame.Rect(square_x, square_y, PIECE_SIZE, PIECE_SIZE)
    pygame.draw.rect(screen, color, square)
    if piece:
        piece_name = get_piece_name(piece)
        screen.blit(images[piece_name], square)

def draw_board(screen, board, images):
    for rank in range(8):
        for file in range(8):
            draw_square_and_piece(screen, board, images, rank, file)

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
    white_text = font.render(f"White: {int(white_time // 60)}:{int(white_time % 60):02d}", True, (0, 0, 0))
    black_text = font.render(f"Black: {int(black_time // 60)}:{int(black_time % 60):02d}", True, (0, 0, 0))

    screen.blit(white_text, (WIDTH // 2 - white_text.get_width() // 2, HEIGHT - 30))
    screen.blit(black_text, (WIDTH // 2 - black_text.get_width() // 2, 10))

def main():
    AI_MOVE_EVENT = pygame.USEREVENT + 1

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Chess")

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    initial_time = 5 * 60  # 5 minutes per player
    white_time = initial_time
    black_time = initial_time
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    images = load_images()
    board = chess.Board()
    dragging = False
    selected_piece = None
    selected_square = None
    delta_x, delta_y = 0, 0

    running = True
    while running:
        clock.tick(60)  # Limit the frame rate to 60 FPS

        if board.turn == chess.WHITE:
            white_time -= clock.get_time() / 1000
        else:
            black_time -= clock.get_time() / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == AI_MOVE_EVENT:
                ai_move = random_move(board)
                print(f"AI move: {ai_move}")
                board.push(ai_move)
                if board.is_game_over():
                    print("Game Over")
                    print(board.result())
                pygame.time.set_timer(AI_MOVE_EVENT, 0)  # Stop the AI_MOVE_EVENT timer

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos
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
                            pygame.time.set_timer(AI_MOVE_EVENT, 1000)  # Schedule the AI_MOVE_EVENT in 1000 ms (1 second)

                    dragging = False
                    selected_piece = None
                    selected_square = None
                    delta_x, delta_y = 0, 0

        draw_board(screen, board, images)

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
