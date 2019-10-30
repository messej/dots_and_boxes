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
gen = 0
ai_name = 'nnai'
file_name = '{0}_gen{1}.pt'.format(ai_name, gen)
file_path = os.path.join("game",  file_name)

if __name__ == '__main__':
    rows = 5
    cols = 5
    size = 5, 5
    model = Network(*size)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    player = NNAI("NN", model, *size)
    players = [NNAI('NN0', n=rows, m=cols), NNAI('NN1', n=rows, m=cols)]
    players = [GenericAI('1_generic'), NNAI('-1_nn', n=rows, m=cols, exploration_turns=3)]
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


