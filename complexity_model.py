import tensorflow as tf

elo_scale = 3000
piece_letters = "pnbrqk"
all_pieces = "pPnNbBrRqQkK"
castling_values = "kKqQ"

def invert_letters(fen):
    """
    Invert the case of piece letters in a FEN string.

    Args:
    fen (str): The FEN string.

    Returns:
    str: The FEN string with inverted piece letters.
    """
    temp_letter = "T"
    fen_invert = fen
    for letter in piece_letters:
        upper = letter.upper()
        fen_invert = fen_invert.replace(letter, temp_letter)
        fen_invert = fen_invert.replace(upper, letter)
        fen_invert = fen_invert.replace(temp_letter, upper)
    return fen_invert

def invert_fen(fen):
    """
    Invert a FEN string so that white is always the side to move.

    Args:
    fen (str): The FEN string.

    Returns:
    str: The inverted FEN string with white to move.
    """
    parts = fen.split(" ")
    position_part = parts[0]
    color_part = parts[1]

    if color_part == "w":
        return fen

    castle_part = parts[2]

    fen_invert_letters = invert_letters(position_part)
    fen_split = fen_invert_letters.split("/")

    position_part_invert = "/".join(reversed(fen_split))
    castle_part_invert = invert_letters(castle_part)

    return position_part_invert + " w " + castle_part_invert + " " + " ".join(parts[3:])

def fill_fen(fen):
    """
    Fill a FEN string by replacing digit placeholders with 'E'.

    Args:
    fen (str): The FEN string.

    Returns:
    str: The filled FEN string.
    """
    fen_fill = fen
    for i in range(1, 9):
        fen_fill = fen_fill.replace(str(i), "E" * i)
    return fen_fill.replace("/", "")

def get_features(fen):
    """
    Get the position features from a FEN string.

    Args:
    fen (str): The FEN string.

    Returns:
    list: A list of position features.
    """
    fen_filled = fill_fen(fen)
    features = []
    for val in fen_filled:
        for piece in all_pieces:
            features.append(float(val == piece))
    return features

def get_castle_features(castle_string):
    """
    Get the castling features from a FEN string.

    Args:
    castle_string (str): The castling part of the FEN string.

    Returns:
    list: A list of castling features.
    """
    features = []
    for s in castling_values:
        features.append(float(s in castle_string))
    return features

def get_all_features(fen, elo):
    """
    Get all features for a given FEN string and Elo rating.

    Args:
    fen (str): The FEN string.
    elo (float): The Elo rating.

    Returns:
    list: A list of all features.
    """
    if fen is None:
        return []

    fen_white = invert_fen(fen)
    parts = fen_white.split(" ")

    position_part = parts[0]
    castle_part = parts[2]

    features_position = get_features(position_part)
    features_castle = get_castle_features(castle_part)
    features_other = [elo / elo_scale]

    return features_position + features_castle + features_other

def get_complexity_scores(fen, elo, model):
    """
    Get the complexity scores (CP loss and blunder chance) for a given FEN and Elo rating.

    Args:
    fen (str): The FEN string.
    elo (float): The Elo rating.
    model (tf.keras.Model): The trained TensorFlow model.

    Returns:
    dict: A dictionary containing the CP loss and blunder chance scores.
    """
    all_features = get_all_features(fen, elo)
    features_tensor = tf.constant([all_features], dtype=tf.float32)
    pred = model(features_tensor)
    cp_loss = pred.numpy()[0][0] * 36.77
    blunder_chance = pred.numpy()[0][1] * 0.13
    return {"cp_loss": cp_loss, "blunder_chance": blunder_chance}


if __name__ == "__main__":
    trained_model = tf.keras.models.load_model("complexity_model/complexity_model.h5")

    fen = "r1b1r3/pp1n1pk1/2pR1np1/4p2q/2B1P2P/2N1Qp2/PPP4R/2K3N1 w - - 1 18"
    elo = 1000

    complexity_scores = get_complexity_scores(fen, elo, trained_model)

    cp_loss = complexity_scores["cp_loss"]
    blunder_chance = complexity_scores["blunder_chance"]

    print(f"Expected loss: {cp_loss:.0f} CP")
    print(f"Blunder chance: {blunder_chance * 100:.0f}%")