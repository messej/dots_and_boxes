from game import Paper
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import multiprocessing
import itertools
import os

from players.nn_ai import NNAI, Network



class DataGenArena(Dataset):
    BASEDIR = 'player_files'

    def __init__(self, player, arena_name, size=(7, 7), matches=100, data=None):
        self.player = player
        self.arena_name = arena_name
        self.rows = size[0]
        self.cols = size[1]
        self.grid_size = (2*size[0]+1) * (2*size[1]+1)
        self.round_num = 0
        self.max_game_len = 2*self.rows*self.cols + self.rows + self.cols
        self.shuffle = True  # might not be the best way or necc
        if data is None:
            self.data, self.labels = self.play_matches(matches)
        else:
            self.data, self.labels = data

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return len(self.labels)

    def play_match(self, *_):
        game = Paper(self.player, self.player, self.rows, self.cols)
        game_states = []
        while game.winner() is None:
            game_state = game.grid
            game_state90 = np.rot90(game_state)
            game_state180 = np.rot90(game_state90)
            game_state270 = np.rot90(game_state180)
            game_state_flip = np.flip(game_state, 0)
            game_state_flip90 = np.flip(game_state90, 0)
            game_state_flip180 = np.flip(game_state180, 0)
            game_state_flip270 = np.flip(game_state270, 0)
            game_states.extend((game_state, game_state90, game_state180, game_state270,
                                game_state_flip, game_state_flip90, game_state_flip180, game_state_flip270))
            game.update()

        if self.shuffle:
            random.shuffle(game_states)
        pad_len = 8*self.max_game_len - len(game_states)
        pad = random.sample(game_states, pad_len)
        game_states.extend(pad)
        game_states = np.stack(game_states)
        print(game_states.shape)
        game_results = [game.winner()]*len(game_states)
        return game_states, game_results

    def play_matches(self, matches=100):
        pool = multiprocessing.Pool()
        result = pool.map(self.play_match, range(matches))
        pool.close()
        pool.join()

        game_states, game_results = zip(*result)
        game_states, game_results = np.concatenate(game_states), list(itertools.chain(*game_results))

        game_states = np.reshape(game_states, (len(game_states), self.grid_size))
        return torch.from_numpy(game_states), torch.Tensor(game_results)


"""
if __name__ == '__main__':
    size = 3, 3
    nm = (2 * size[0] + 1) * (2 * size[1] + 1)
    player = NNAI("NN", None, size[0], size[1], 3)
    dataset = DataGenArena(player, "test_data_gen", size, 100)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=100,
                              shuffle=True,
                              num_workers=2)
    for epoch in range(2):
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # Run your training process
            print(epoch, i, "inputs", inputs.data.shape, "labels", labels.data[:3])"""

if __name__ == '__main__':
    print("this is temp, will need to make trainer")
    size = 5, 5
    model = Network(*size)
    player = NNAI("NN", model, *size)
    dataset = DataGenArena(player, "test_data_gen", size, matches=10)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=10,
                              shuffle=True,
                              num_workers=2)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(2):
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass: Compute predicted y by passing x to the model
            # print(inputs.float().size)
            # print(inputs.float().size())

            y_pred = model(inputs.float())

            # Compute and print loss
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.data.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """
    gen = 0
    ai_name = 'nnai'
    file_name = '{0}_gen{1}.pt'.format(ai_name, gen)
    file_path = os.path.join('player_files', file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print(file_path)
    torch.save(model.state_dict(), file_path)"""
