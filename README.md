# Chessie

This project aims to create a chess bot that not only plays moves at a human level, but also mimics the time usage of a human player. This makes the bot more suitable for practicing time management skills during a chess game. The bot is designed to have an Elo rating of around 1500 and it adjusts its move time based on various factors such as players' Elo ratings, time remaining on the clock, material count, and move complexities.

## Features

- Chess bot with an Elo rating of around 1500
- Plays three-minute blitz games
- Mimics human time usage based on various factors
- Utilizes a custom model for move complexity and time management
- Provides a more human-like experience when practicing against the bot

## Installation

1. Clone the repository:

```
git clone https://github.com/nicktalati/chessie
```

2. Navigate to the project folder:

```
cd chessie
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```
or
```
conda env create --file=requirements.yml
```

4. Download and install the [Stockfish chess engine](https://stockfishchess.org/download/) for your platform. Make sure to add the path to the `stockfish_path` variable in the `main.py` file:

```python
stockfish_path = "/path/to/stockfish"
```
e.g., 
```python
stockfish_path = "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish"
```

5. Run the `play.py` script to start playing against the bot:

```python
python play.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
