
import socket
import random
import sys
import time

HOST = '127.0.0.1'
PORT = 1234

def find_empty_positions(board):
    return [(i, j) for i in range(5) for j in range(5) if board[i][j] == 0]

def set_move(board, move, player):
    i = move // 10 - 1
    j = move % 10 - 1
    if 0 <= i < 5 and 0 <= j < 5 and board[i][j] == 0:
        board[i][j] = player
        return True
    return False

def random_move(board, player):
    empty = find_empty_positions(board)
    if not empty:
        return 0
    i, j = random.choice(empty)
    board[i][j] = player
    return (i + 1) * 10 + (j + 1)

def play_game(conn):
    board = [[0] * 5 for _ in range(5)]
    last_move = 0
    end_game = False

    # Initial greeting
    conn.send(b"READY")

    # Receive bot intro message: "player_number BotName"
    intro = conn.recv(1024).decode().strip()
    print(f"[bot intro] {intro}")
    try:
        player_str = intro.split()[0]
        bot_player = int(player_str)
        if bot_player not in [1, 2]:
            raise ValueError
    except:
        print("Invalid player number from bot. Closing connection.")
        conn.close()
        return

    server_player = 3 - bot_player

    print(f"Game start: Bot is player {bot_player}, Server (random) is player {server_player}")
    first_player = "Bot" if bot_player == 1 else "Server (random)"
    print(f"Player who goes first: {first_player}")

    while not end_game:
        if bot_player == 1:
            # Bot goes first
            # Tell bot: status=0 your move, last_move=server's last move (0 for first turn)
            msg = 0 * 100 + last_move
            conn.send(str(msg).encode())

            # Receive bot move
            move_data = conn.recv(1024).decode()
            if not move_data:
                print("Connection lost")
                break
            move = int(move_data)
            print(f"[bot move] {move}")
            if not set_move(board, move, bot_player):
                print("Invalid bot move. Ending game.")
                conn.send(str(5 * 100).encode())  # Your error
                break
            last_move = move

            # Server random move
            time.sleep(0.2)
            move = random_move(board, server_player)
            print(f"[server move] {move}")

            if not find_empty_positions(board):
                print("Board full. Draw.")
                conn.send(str(3 * 100 + last_move).encode())
                end_game = True
                continue

            # Tell bot: status=0 your move, last_move=server move
            msg = 0 * 100 + move
            conn.send(str(msg).encode())

        else:
            # Server goes first
            # Server random move
            move = random_move(board, server_player)
            print(f"[server move] {move}")

            # Tell bot: status=0 your move, last_move=server move
            msg = 0 * 100 + move
            conn.send(str(msg).encode())

            # Receive bot move
            move_data = conn.recv(1024).decode()
            if not move_data:
                print("Connection lost")
                break
            move = int(move_data)
            print(f"[bot move] {move}")
            if not set_move(board, move, bot_player):
                print("Invalid bot move. Ending game.")
                conn.send(str(5 * 100).encode())  # Your error
                break
            last_move = move

            if not find_empty_positions(board):
                print("Board full. Draw.")
                conn.send(str(3 * 100 + last_move).encode())
                end_game = True
                continue

    print("Game ended, waiting for new connection...\n")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[*] Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"[+] Connected by {addr}")
                try:
                    play_game(conn)
                except Exception as e:
                    print(f"Error during game: {e}")
                print("Connection closed.")

if __name__ == "__main__":
    main()
