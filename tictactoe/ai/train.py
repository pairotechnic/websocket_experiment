"""
Self-play Q-learning trainer.
Run from the /tictactoe directory.
    python -m ai.train
"""
from pathlib import Path
from ai.q_agent import QAgent
from game import Game
import uuid

SAVE_PATH = Path(__file__).parent / "q_table_self.pkl"

# Print a progress line every N episodes out of total episodes
EPISODES = 200_000
LOG_EVERY = 10_000

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

    for ep in range(1, EPISODES + 1):
        winner = run_episode(agent_x, agent_o)

        if winner == "X":
            wins["X"] += 1
        elif winner == "O":
            wins["O"] += 1
        else :
            wins["Draw"] += 1

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