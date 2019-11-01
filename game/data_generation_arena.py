from game import Paper
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import multiprocessing
import itertools
import os

try:
    import psutil
except ImportError:
    IMPORT_FAILED = True
else:
    IMPORT_FAILED = False


def limit_cpu():
    """is called at every process start"""
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


class DataGenArena(Dataset):
    BASEDIR = 'player_files'

    def __init__(self, player, matches=100, arena_name='data_gen', data=None):
        self.player = player
        self.arena_name = arena_name
        self.rows = player.shape[0]
        self.cols = player.shape[1]
        self.grid_shape = (2*player.shape[0]+1) * (2*player.shape[1]+1)
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
        if IMPORT_FAILED:
            pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
        else:
            pool = multiprocessing.Pool(initializer=limit_cpu, processes=(multiprocessing.cpu_count() - 1))
        result = pool.map(self.play_match, range(matches))
        pool.close()
        pool.join()

        game_states, game_results = zip(*result)
        game_states, game_results = np.concatenate(game_states), list(itertools.chain(*game_results))

        # TODO: see if I can implement this with view just for fun
        game_states = np.reshape(game_states, (len(game_states), self.grid_shape))
        return torch.from_numpy(game_states), torch.Tensor(game_results)


if __name__ == '__main__':
    from players.nn_ai import NNAI
    size = 7, 7
    nm = (2 * size[0] + 1) * (2 * size[1] + 1)
    player = NNAI("NN", size[0], size[1], exploration_turns=3)
    dataset = DataGenArena(player, 100)
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
            print(epoch, i, "inputs", inputs.data.shape, "labels", labels.data[:3])

"""
if __name__ == '__main__':
    print("this is temp, will need to make trainer")
    size = 5, 5
    model = Network(*size)
    # model.load_state_dict(torch.load(file_name))
    # model.eval()
    player = NNAI("NN", model, *size)
    dataset = DataGenArena(player, matches=500)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=20,
                              shuffle=True,
                              num_workers=2)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(15):
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

    gen = 1
    ai_name = 'nnai'
    file_name = '{0}_gen{1}.pt'.format(ai_name, gen)
    file_path = os.path.join('player_files', file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = file_name
    print(file_path)
    with open(file_path, 'wb+') as model_file:
        torch.save(model.state_dict(), model_file)"""
