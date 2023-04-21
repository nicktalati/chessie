import csv
import re
import tensorflow as tf
from complexity_model import get_complexity_scores

def process_csv(input_file, output_file, model):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['white_elo', 'black_elo', 'white_time_left', 'black_time_left', 'expected_loss', 'blunder_chance', 'move_time']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        game = -1

        for row in reader:
            if int(row['Game']) != game:
                game = int(row['Game'])
                ply = -1
                white_elo = float(row['White Elo'])
                black_elo = float(row['Black Elo'])
                fens = ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1']
                pgn = row['PGN']
                move_times = re.findall(r'%clk\s(\d{1,2}:\d{2}:\d{2}(\.\d{1,2})?)', pgn)
                move_times = [time[0] for time in move_times]
                move_times = [sum(float(t.split(':')[i]) * 60**(2-i) for i in range(3)) for t in move_times]
                move_times = [180.0, 180.0] + move_times
                print(move_times)

            ply += 1
            fen = fens[ply]
            fens.append(row['FEN'])
            wb = fen.split(' ')[1]
        
            if wb == 'w':
                white_time_left = move_times[ply]
                black_time_left = move_times[ply + 1]

            else:
                black_time_left = move_times[ply]
                white_time_left = move_times[ply + 1]

            complexity_scores = get_complexity_scores(fens[ply], (white_elo + black_elo) / 2, model)
            expected_loss = complexity_scores['cp_loss']
            blunder_chance = complexity_scores['blunder_chance']

            move_time = move_times[ply] - move_times[ply + 2]


            output_row = {
                'white_elo': white_elo,
                'black_elo': black_elo,
                'white_time_left': round(white_time_left, 2),
                'black_time_left': round(black_time_left, 2),
                'expected_loss': round(expected_loss, 2),
                'blunder_chance': round(blunder_chance, 5),
                'move_time': round(move_time, 2)
            }

            writer.writerow(output_row)

input_csv = 'new_games_cond.csv'
output_csv = 'training_data.csv'
# input_csv = 'input.csv'
# output_csv = 'output.csv'
model = tf.keras.models.load_model("complexity_model.h5")
process_csv(input_csv, output_csv, model)
