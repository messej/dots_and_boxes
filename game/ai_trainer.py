import torch
from game import DataGenArena
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os


# criterion = torch.nn.SmoothL1Loss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


class AITrainer:
    # TODO: concurrent usage of self.model and player.network makes me nervous
    # TODO: add name? hmmm
    # also unsure about loss and optimizer params
    # TODO: remove gen parameter (?)
    def __init__(self, player, model_name=None, model_dir=None, matches=100, batch_size=8,
                 loss=torch.nn.SmoothL1Loss, optimizer=torch.optim.SGD, lr=0.001, gen=0):
        self.player = player
        self.matches = matches
        self.batch_size = batch_size
        self.epochs = 2

        self.model_dir = model_dir
        self.gen = gen
        # TODO: fix use of preexisting model
        self.model_path = model_name
        if self.model_path is not None:
            self.player.load_model(self.model_path)
            # TODO: figure out how to get gen from somewhere
            self.gen = gen
        else:
            self.gen = 0
            self.model_path = self.full_path()
            self.player.save_model(self.model_path)

        # self.model = player.network
        self.loss = loss()
        self.optimizer = optimizer(self.player.network.parameters(), lr=lr)
        self.generations()

    def generation(self):
        path = self.full_path()
        self.player.load_model(path)
        model = self.player.network
        data = DataGenArena(self.player, matches=self.matches)
        train_loader = DataLoader(dataset=data,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=2)

        for epoch in range(self.epochs):
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(inputs.float())

                # Compute and print loss
                loss = self.loss(y_pred, labels)
                print(epoch, i, loss.data.item())

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.gen += 1

        path = self.full_path()
        model.save_model(path)

    def generations(self):
        while True:
            self.generation()

    def full_path(self):
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        # TODO: implement player __repr__?
        file_name = f'{self.player.name}{self.player.shape}_gen{self.gen}.pt'
        if self.model_dir is not None:
            path = os.path.join(self.model_dir, file_name)
        else:
            path = file_name
        return path


if __name__ == '__main__':
    from players.nn_ai import NNAI
    ai = NNAI("nnai", 7, 7, exploration_turns=5, explore_chance=0.15)
    trainer = AITrainer(ai)
