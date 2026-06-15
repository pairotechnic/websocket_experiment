"""
Self-play Q-learning trainer.
Run from the /tictactoe directory.
    python -m ai.train
"""
from pathlib import Path
from ai.q_agent import QAgent
from game import Game
import uuid
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import math


SAVE_PATH = Path(__file__).parent / "models" / f"q_table_self_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl"

# Print a progress line every N episodes out of total episodes
EPISODES = 200_000
LOG_EVERY = 10_000

LOG_PATH = Path(__file__).parent / "models" / "_training_log.xlsx"

def log_run(agent: QAgent, episodes: int, wins: dict, q_table_size: int):
    cols = [
        "model filename", "alpha", "gamma", "epsilon start", "epsilon min",
        "epsilon decay", "epsilon min episode", "episodes", "x wins", "o wins", "draws", "q-table size"
    ]

    # Load or create workbook
    if LOG_PATH.exists():
        wb = openpyxl.load_workbook(LOG_PATH)
        ws = wb.active
    else :
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Training Log"

        # Header row
        header_fill = PatternFill("solid", start_color="1F4E79")
        for c, col in enumerate(cols, start=1):
            cell = ws.cell(row=1, column=c, value=col)
            cell.font = Font(bold=True, color="FFFFFF", name="Arial")
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        col_widths = [36, 8, 8, 14, 12, 14, 20, 10, 10, 10, 10, 14]
        for i, w in enumerate(col_widths, start=1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    # Compute epsilon_min_episode: episode at which epsilon first hits epsilon_min
    # epsilon after ep episodes = max(epsilon_min, epsilon_start * decay^ep )
    # solve : epsilon_start * decay^ep = epsilon_min
    # ep = log(epsilon_min / epsilon_start) / log(decay)
    if agent.epsilon_decay < 1.0:
        ep_min = math.ceil(
            math.log(agent.epsilon_min / 1.0) / math.log(agent.epsilon_decay)
        )
        ep_min = min(ep_min, episodes)
    else :
        ep_min = episodes

    total = sum(wins.values())
    row = [
        SAVE_PATH.name,
        agent.alpha,
        agent.gamma,
        1.0,    # epsilon always starts at 1, for pure random exploration in the beginning
        agent.epsilon_min,
        agent.epsilon_decay,
        ep_min,
        episodes,
        wins["X"],
        wins["O"],
        wins["Draw"],
        q_table_size
    ]

    # Alternate row shading
    next_row = ws.max_row + 1
    fill = PatternFill("solid", start_color="D6E4F0") if next_row % 2 == 0 else None
    for c, val in enumerate(row, start=1):
        cell = ws.cell(row=next_row, column=c, value=val)
        cell.font = Font(name="Arial")
        cell.alignment = Alignment(horizontal="center")
        if fill:
            cell.fill = fill

    wb.save(LOG_PATH)
    print(f"Logged run -> {LOG_PATH}")


def run_episode(agent_x: QAgent, agent_o: QAgent) -> str | None:
    """
    Play one full game between two Q-agents.
    Returns "X", "O", or None (draw).
    """
    game = Game(game_id=str(uuid.uuid4())[:8])

    agents = [agent_x, agent_o] # index 0 = X, index 1 = O
    symbols = ["X", "O"]

    while not game.game_over:
        idx = game.current_turn # whose turn it is (0 or 1)
        agent = agents[idx]

        action = agent.act(game.board)
        game.make_move(idx, action) # ANSWER : Isn't action being passed a Q-value of action here? Shouldn't we be passing the cell/action instead in game.make_move?
        
        if game.game_over:
            # Terminal rewards
            if game.winner == symbols[idx]: # this agent won
                agents[idx].learn(game.board, reward=1.0, done=True)
                agents[1-idx].learn(game.board, reward=-1.0, done=True)
            else :
                agents[0].learn(game.board, reward = 0.5, done=True)
                agents[1].learn(game.board, reward=0.5, done=True)
        else :
            # Intermediate move - no reward yet
            agents[idx].learn(game.board, reward=0.0, done=False)
    
    return game.winner

def train():
    agent_x = QAgent(symbol="X")
    agent_o = QAgent(symbol="O")

    wins = {"X": 0, "O": 0, "Draw": 0}
    total_wins = {"X": 0, "O": 0, "Draw": 0}

    for ep in range(1, EPISODES + 1):
        winner = run_episode(agent_x, agent_o)
        key = winner if winner else "Draw"
        wins[key] += 1
        total_wins[key] += 1

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        if ep % LOG_EVERY == 0:
            total = sum(wins.values())
            print(
                f"Episode {ep:>7} | "
                f"ε={agent_x.epsilon:.4f} | "
                f"X wins: {wins['X']/total:.1%}  "
                f"O wins: {wins['O']/total:.1%}  "
                f"Draws: {wins['Draw']/total:.1%} | "
                f"Q-table size: {len(agent_x.q_table)}"
            )
            wins = {"X": 0, "O": 0, "Draw": 0}

    # Save agent_x's table - at inference time we always load one agent
    # and let it play as whichever symbol the human isn't using
    agent_x.save(SAVE_PATH)

    # Accumulate final wins across all episodes for the log
    # (move wins dict outside the loop and don't reset after last log)
    log_run(agent_x, EPISODES, total_wins, len(agent_x.q_table))
    print("Training complete.")

if __name__ == "__main__":
    train()

####################################################

"""
Analysis : 
    
    Given : 
        EPISODES = 200_000
        LOG_EVERY = 10_000
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995
    Results : 
        Episode  200000 | ε=0.0100 | X wins: 96.8%  O wins: 1.2%  Draws: 2.0% | Q-table size: 2802
    Explanation : 
        The Q-table size (~2800) is suspiciously small. Tic-tac-toe has 5,478 legal board states. 
        We'd expect the Q-table to have entries approaching that number after 200k episodes. 
        2802 means the agents are repeatedly visiting the same narrow set of states — 
        they've converged on a small set of favourite openings and never explore beyond them.
    Reason: 
        ε decayed too fast
        With epsilon_decay=0.9995, ε hits its minimum of 0.01 very early:
        0.9995^episode = 0.01  →  episode ≈ 919
        So after just ~919 episodes, both agents are almost always exploiting. 
        They lock into whatever strategies they learned in the first 919 games and never escape. 
        The remaining 199,000 episodes just reinforce the same narrow paths.
    The X wins ~97% problem : 
        This is a self-reinforcing collapse. Early on, X stumbles onto a decent opening. 
        O never learns to counter it because O also stopped exploring at episode ~919. 
        So X keeps winning, X's strategy gets heavily reinforced, O's counter-play never develops. 
        They've reached a local equilibrium, not a good one.
"""