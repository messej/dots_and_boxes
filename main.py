from players import *
from visualizer import Visualizer
import arcade
import json
import os
from game import Arena
from game import Paper
from players.nn_ai import Network
import torch

from timeit import default_timer as Timer
gen = 70
ai_name = 'nnai'
ai_shape = 3, 3
file_name = f'{ai_name}{ai_shape}_gen{gen}.pt'
file_path = os.path.join("game",  file_name)
gen2 = 60
file_name = f'{ai_name}{ai_shape}_gen{gen2}.pt'
file_path2 = os.path.join("game",  file_name)

if __name__ == '__main__':
    rows = ai_shape[0]
    cols = ai_shape[1]
    size = ai_shape
    model = Network(*size)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    player = NNAI(f"NN{gen}", *size, model, exploration_turns=1, explore_chance=0)
    model = Network(*size)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    player2 = NNAI(f"NN{gen2}", *size, model, exploration_turns=1, explore_chance=0)
    players = [NNAI('NN0', n=rows, m=cols), NNAI('NN1', n=rows, m=cols)]
    players = [player2, player]
    # players = players[::-1]

    # players = [Brute('Brute0'), Brute('Brute1')]

    arena = Arena(players, "testing", (rows, cols))
    r = arena.create_round()

    # super rigorous way of testing preformance
    t0 = Timer()
    arena.play_round(r)
    t1 = Timer()
    print(t1 - t0)

    # for i in range(100):
    #     players = [GenericAI('player1'), GenericAI('player2')]
    #     paper = Paper(players[0], players[1], rows, cols)
    #     while not paper.winner():
    #         paper.update()
    #         # print(paper.grid)
    #     print(paper.winner())

    # may remove
    file_path = os.path.join(Arena.BASEDIR, arena.arena_name)
    file_path = os.path.join(file_path, "round1match1.json")
    with open(file_path, "r") as read_file:
        match = json.load(read_file)
    sample_game = Visualizer("round1match1", match, frame_rate=8)
    # print(len(sample_game.match["moves"]))
    arcade.run()


