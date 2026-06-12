def get_winner(board: list[str]) -> str | None:
    """Returns 'X', 'O', 'draw', or None if game is still ongoing."""
    WINNING_COMBOS = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # cols
        [0,4,8], [2,4,6],           # diags
    ]
    for combo in WINNING_COMBOS:
        vals = [board[i] for i in combo]
        if vals[0] != "" and vals[0] == vals[1] == vals[2]:
            return vals[0]  # 'X' or 'O'
    if all(cell != "" for cell in board):
        return "draw"
    return None # game still ongoing

def minimax(board: list[str], is_maximizing: bool, maximizing_symbol: str, depth: int = 0) -> int:
    """
    Recursively scores a board position from the maximizing player's perspective.
    Returns +1 (maximizer wins), -1 (minimizer wins), 0 (draw).
    """
    minimizing_symbol = "O" if maximizing_symbol == "X" else "X"
    winner = get_winner(board)

    if winner == maximizing_symbol:
        return 10 - depth # win sooner - higher score
    if winner == minimizing_symbol:
        return depth - 10 # lose later - higher score (increase opportunities for opponent to make a mistake)
    if winner == "draw":
        return 0
    
    empty_cells = [i for i, cell in enumerate(board) if cell == ""]

    if is_maximizing:
        best = -2
        for cell in empty_cells:
            board[cell] = maximizing_symbol
            score = minimax(board, False, maximizing_symbol, depth+1)
            board[cell] = ""
            best = max(best, score)
        return best
    else :
        best = 2
        for cell in empty_cells:
            board[cell] = minimizing_symbol
            score = minimax(board, True, maximizing_symbol, depth+1)
            board[cell] = ""
            best = min(best, score)
        return best
    
def best_move(board: list[str], ai_symbol: str) -> int:
    """
    Returns the index (0-8) of the best move for ai_symbol.
    This is the only function the rest of the app needs to call.
    """
    empty_cells = [i for i, cell in enumerate(board) if cell == ""]
    best_score = -2
    best_cell = empty_cells[0] # fallback (always overwritten)

    for cell in empty_cells:
        board[cell] = ai_symbol
        score = minimax(board, False, ai_symbol)
        board[cell] = ""
        if score > best_score:
            best_score = score
            best_cell = cell

    return best_cell

class MinimaxAgent:
    """Wraps the minimax function as a stateless agent."""
    def act(self, board: list[str], symbol: str) -> int:
        return best_move(board, symbol)