#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>

#define SIZE 5

int board[SIZE][SIZE];
int max_depth;

void setBoard() {
    memset(board, 0, sizeof(board));
}

void setMove(int move, int player) {
    int row = (move / 10) - 1;
    int col = (move % 10) - 1;
    if (row >= 0 && row < SIZE && col >= 0 && col < SIZE)
        board[row][col] = player;
}

bool isMovesLeft() {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            if (board[i][j] == 0) return true;
    return false;
}

// Check if placing a piece at (row, col) creates a winning line
bool checkWinningMove(int row, int col, int player) {
    board[row][col] = player; // Temporarily place the piece
    
    // Check all directions for 4 in a row
    int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};
    
    for (int d = 0; d < 4; d++) {
        int dx = directions[d][0];
        int dy = directions[d][1];
        int count = 1; // Count the piece we just placed
        
        // Check positive direction
        for (int i = 1; i < 4; i++) {
            int nr = row + i * dx;
            int nc = col + i * dy;
            if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && board[nr][nc] == player) {
                count++;
            } else break;
        }
        
        // Check negative direction
        for (int i = 1; i < 4; i++) {
            int nr = row - i * dx;
            int nc = col - i * dy;
            if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && board[nr][nc] == player) {
                count++;
            } else break;
        }
        
        if (count >= 4) {
            board[row][col] = 0; // Remove temporary piece
            return true;
        }
    }
    
    board[row][col] = 0; // Remove temporary piece
    return false;
}

// Check if opponent has 3 in a row that needs blocking
bool checkBlockingMove(int row, int col, int player) {
    int opponent = 3 - player;
    return checkWinningMove(row, col, opponent);
}

// Check if placing at (i,j) creates good spacing (X _ _ X pattern)
int getSpacingBonus(int i, int j, int player) {
    int bonus = 0;
    int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}}; // horizontal, vertical, diag1, diag2
    
    for (int d = 0; d < 4; d++) {
        int dx = directions[d][0];
        int dy = directions[d][1];
        
        // Check X _ _ X pattern (3 spaces apart)
        int x1 = i + 3*dx, y1 = j + 3*dy;
        int x2 = i - 3*dx, y2 = j - 3*dy;
        
        if (x1 >= 0 && x1 < SIZE && y1 >= 0 && y1 < SIZE && board[x1][y1] == player) {
            // Check if middle spaces are empty
            if (board[i + dx][j + dy] == 0 && board[i + 2*dx][j + 2*dy] == 0) {
                bonus += 8000; // HUGE bonus for X _ _ X
            }
        }
        
        if (x2 >= 0 && x2 < SIZE && y2 >= 0 && y2 < SIZE && board[x2][y2] == player) {
            // Check if middle spaces are empty
            if (board[i - dx][j - dy] == 0 && board[i - 2*dx][j - 2*dy] == 0) {
                bonus += 8000; // HUGE bonus for X _ _ X
            }
        }
    }
    
    return bonus;
}

// HEAVILY penalize adjacent pieces (stupid clustering)
int getAdjacentPenalty(int i, int j, int player) {
    int penalty = 0;
    int adjacent[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
    
    for (int a = 0; a < 8; a++) {
        int ni = i + adjacent[a][0];
        int nj = j + adjacent[a][1];
        
        if (ni >= 0 && ni < SIZE && nj >= 0 && nj < SIZE && board[ni][nj] == player) {
            penalty -= 5000; // MASSIVE penalty for clustering
        }
    }
    
    return penalty;
}

int getPositionBonus(int i, int j, int player) {
    int move = (i + 1) * 10 + (j + 1);
    int bonus = 0;
    
    // HEAVILY penalize center tile
    if (move == 33) return -50000;
    
    // Prioritize corners MUCH more
    if (move == 11 || move == 15 || move == 51 || move == 55) bonus += 5000;
    
    // Favor central tiles (but not dead center)
    else if (move == 22 || move == 24 || move == 42 || move == 44) bonus += 3000;
    
    // Secondary good positions
    else if (move == 23 || move == 32 || move == 34 || move == 43) bonus += 1500;
    
    // Edge positions
    else if (i == 0 || i == SIZE-1 || j == 0 || j == SIZE-1) bonus += 800;
    
    // Add spacing bonus (X _ _ X pattern)
    bonus += getSpacingBonus(i, j, player);
    
    // Subtract clustering penalty
    bonus += getAdjacentPenalty(i, j, player);
    
    return bonus;
}

int evaluateHeuristic(int player) {
    int score = 0;
    int opponent = 3 - player;

    // Check all possible 4-in-a-row positions
    // Horizontal
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j <= SIZE - 4; j++) {
            int ai = 0, op = 0, empty = 0;
            for (int k = 0; k < 4; k++) {
                if (board[i][j + k] == player) ai++;
                else if (board[i][j + k] == opponent) op++;
                else empty++;
            }
            
            if (ai == 4) score += 100000; // Win
            if (op == 4) score -= 100000; // Loss
            if (ai == 3 && op == 0) score += 10000; // 3 in a row, can win
            if (op == 3 && ai == 0) score -= 15000; // Opponent 3 in a row, must block
            if (ai == 2 && op == 0) score += 1000;
            if (op == 2 && ai == 0) score -= 1500;
        }
    }
    
    // Vertical
    for (int i = 0; i <= SIZE - 4; i++) {
        for (int j = 0; j < SIZE; j++) {
            int ai = 0, op = 0, empty = 0;
            for (int k = 0; k < 4; k++) {
                if (board[i + k][j] == player) ai++;
                else if (board[i + k][j] == opponent) op++;
                else empty++;
            }
            
            if (ai == 4) score += 100000;
            if (op == 4) score -= 100000;
            if (ai == 3 && op == 0) score += 10000;
            if (op == 3 && ai == 0) score -= 15000;
            if (ai == 2 && op == 0) score += 1000;
            if (op == 2 && ai == 0) score -= 1500;
        }
    }
    
    // Diagonal (top-left to bottom-right)
    for (int i = 0; i <= SIZE - 4; i++) {
        for (int j = 0; j <= SIZE - 4; j++) {
            int ai = 0, op = 0, empty = 0;
            for (int k = 0; k < 4; k++) {
                if (board[i + k][j + k] == player) ai++;
                else if (board[i + k][j + k] == opponent) op++;
                else empty++;
            }
            
            if (ai == 4) score += 100000;
            if (op == 4) score -= 100000;
            if (ai == 3 && op == 0) score += 10000;
            if (op == 3 && ai == 0) score -= 15000;
            if (ai == 2 && op == 0) score += 1000;
            if (op == 2 && ai == 0) score -= 1500;
        }
    }
    
    // Diagonal (top-right to bottom-left)
    for (int i = 0; i <= SIZE - 4; i++) {
        for (int j = 3; j < SIZE; j++) {
            int ai = 0, op = 0, empty = 0;
            for (int k = 0; k < 4; k++) {
                if (board[i + k][j - k] == player) ai++;
                else if (board[i + k][j - k] == opponent) op++;
                else empty++;
            }
            
            if (ai == 4) score += 100000;
            if (op == 4) score -= 100000;
            if (ai == 3 && op == 0) score += 10000;
            if (op == 3 && ai == 0) score -= 15000;
            if (ai == 2 && op == 0) score += 1000;
            if (op == 2 && ai == 0) score -= 1500;
        }
    }

    // Position bonuses
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] == player) {
                score += getPositionBonus(i, j, player);
            } else if (board[i][j] == opponent) {
                score -= getPositionBonus(i, j, opponent) / 3;
            }
        }
    }

    return score;
}

int alphaBeta(int depth, int alpha, int beta, bool maximizingPlayer, int player) {
    if (depth == 0 || !isMovesLeft())
        return evaluateHeuristic(player);

    int opponent = 3 - player;

    if (maximizingPlayer) {
        int maxEval = -1000000;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                if (board[i][j] == 0) {
                    board[i][j] = player;
                    int eval = alphaBeta(depth - 1, alpha, beta, false, player);

                    board[i][j] = 0;
                    if (eval > maxEval) maxEval = eval;
                    if (eval > alpha) alpha = eval;
                    if (beta <= alpha) return maxEval;
                }
            }
        }
        return maxEval;
    } else {
        int minEval = 1000000;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                if (board[i][j] == 0) {
                    board[i][j] = opponent;
                    int eval = alphaBeta(depth - 1, alpha, beta, true, player);
                    board[i][j] = 0;
                    if (eval < minEval) minEval = eval;
                    if (eval < beta) beta = eval;
                    if (beta <= alpha) return minEval;
                }
            }
        }
        return minEval;
    }
}

// Modified bestMove function that always respects max_depth
int bestMove(int player, int max_depth) {
    int bestVal = -1000000;
    int move = -1;

    // Use minimax with the specified depth for ALL moves
    // No more shortcuts - let the evaluation function handle priorities
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] == 0) {
                board[i][j] = player;
                int moveVal = alphaBeta(max_depth, -1000000, 1000000, false, player);
                board[i][j] = 0;
                
                if (moveVal > bestVal) {
                    bestVal = moveVal;
                    move = (i + 1) * 10 + (j + 1);
                }
            }
        }
    }
    
    return move;
}

int main(int argc, char *argv[]) {
    int server_socket;
    struct sockaddr_in server_addr;
    char server_message[16], player_message[16];
    bool end_game;
    int player, msg, move;

    gsl_rng *generator = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(generator, time(NULL));

    if (argc != 6) {
        printf("Wrong number of arguments\n");
        return -1;
    }

    max_depth = atoi(argv[5]);

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        printf("Unable to create socket\n");
        return -1;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(atoi(argv[2]));
    server_addr.sin_addr.s_addr = inet_addr(argv[1]);

    if (connect(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        printf("Unable to connect\n");
        return -1;
    }

    if (recv(server_socket, server_message, sizeof(server_message), 0) < 0) {
        printf("Error while receiving server's message\n");
        return -1;
    }

    memset(player_message, '\0', sizeof(player_message));
    snprintf(player_message, sizeof(player_message), "%s %s", argv[3], argv[4]);
    if (send(server_socket, player_message, strlen(player_message), 0) < 0) {
        printf("Unable to send message\n");
        return -1;
    }

    setBoard();
    end_game = false;
    sscanf(argv[3], "%d", &player);

    while (!end_game) {
        memset(server_message, '\0', sizeof(server_message));
        if (recv(server_socket, server_message, sizeof(server_message), 0) < 0) {
            printf("Error while receiving server's message\n");
            return -1;
        }
        sscanf(server_message, "%d", &msg);
        move = msg % 100;
        msg = msg / 100;
        if (move != 0) {
            setMove(move, 3 - player);
        }
        if (msg == 0 || msg == 6) {
            move = bestMove(player, max_depth);
            setMove(move, player);
            memset(player_message, '\0', sizeof(player_message));
            snprintf(player_message, sizeof(player_message), "%d", move);
            if (send(server_socket, player_message, strlen(player_message), 0) < 0) {
                printf("Unable to send message\n");
                return -1;
            }
        } else {
            end_game = true;
            switch (msg) {
                case 1: printf("You won.\n"); break;
                case 2: printf("You lost.\n"); break;
                case 3: printf("Draw.\n"); break;
                case 4: printf("You won. Opponent error.\n"); break;
                case 5: printf("You lost. Your error.\n"); break;
            }
        }
    }

    close(server_socket);
    gsl_rng_free(generator);
    return 0;
}
