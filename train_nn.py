########################################################################################################################
#                                                                                                                      #
# The main body of this code is taken from:                                                                            #
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/01-basics/feedforward_neural_network                #
#                                                                                                                      #
# Adaptations by Maartje ter Hoeve.                                                                                    #
# Comments about adaptations specifically to run this code with Sacred start with 'SACRED'                             #
#                                                                                                                      #
# Please have a look at the Sacred documentations for full details about Sacred itself: https://sacred.readthedocs.io/ #
#                                                                                                                      #
########################################################################################################################

import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets.MovingMNIST import MovingMNIST

from sacred import Experiment
from sacred.observers import MongoObserver

from model_nn import EncDec

# Get experiment name from command line
EXPERIMENT_NAME = sys.argv[1]
del sys.argv[1]

YOUR_CPU = None  # None is the default setting and will result in using localhost, change if you want something else
DATABASE_NAME = 'my_database'

ex = Experiment(EXPERIMENT_NAME)

ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:

    def __init__(self):
        # SACRED: we don't need any parameters here, they're in the config and the functions get a @ex.capture handle
        # later
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.loss_fn = nn.MSELoss()
        self.train_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.test_loader = self.get_dataloaders()

    # SACRED: The parameters input_size, hidden_size and num_classes come from our Sacred config file. Sacred finds
    # these because of the @ex.capture handle. Note that we did not have to add these parameters when we called this
    # method in the init.
    @ex.capture
    def make_model(self, input_size, hidden_size, num_classes):
        # model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        model = EncDec(input_size, hidden_size, kernel_size=(3, 3)).to(device)
        return model

    # SACRED: The parameter learning_rate comes from our Sacred config file. Sacred finds this because of the
    # @ex.capture handle. Note that we did not have to add this parameter when we called this method in the init.
    @ex.capture
    def make_optimizer(self, learning_rate):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    # SACRED: Here we do not use any parameters from the config file and hence we do not need the @ex.capture handle.
    def get_datasets(self):
        train_dataset = MovingMNIST('data/MovingMNIST/mnist_test_seq.npy', train=True, download=True,)
        test_dataset = MovingMNIST('data/MovingMNIST/mnist_test_seq.npy', train=False)

        def to_float(x: torch.Tensor): return x.to(dtype=torch.float32)
        def norm(x: torch.Tensor): return (x - train_dataset.mean) / train_dataset.std
        def t(x): return norm(to_float(x))

        train_dataset.transform = t
        test_dataset.transform = t

        return train_dataset, test_dataset

    # SACRED: The parameter batch_size comes from our Sacred config file. Sacred finds this because of the
    # @ex.capture handle. Note that we did not have to add this parameter when we called this method in the init.
    @ex.capture
    def get_dataloaders(self, batch_size):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        return train_loader, test_loader

    # SACRED: The parameter num_epochs comes from our Sacred config file. Sacred finds this because of the
    # @ex.capture handle. Note that we did not have to add this parameter when we called this method.
    # _run is a special object you can pass to your function and it allows you to keep track of parameters (like we do).
    @ex.capture
    def train(self, num_epochs, _run):
        self.model.train()
        total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device, dtype=torch.float32)

                # Forward pass
                outputs = self.model(images)

                loss = self.loss_fn(outputs, labels)
                _run.log_scalar('loss', float(loss.data))  # SACRED: Keep track of the loss

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    def test(self):
        with torch.no_grad():
            self.model.eval()
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(device)
                labels = labels.to(device, dtype=torch.float32)
                outputs = self.model(images)
                total += labels.size(0)
                mse = torch.mean((outputs - labels) ** 2, (1, 2, 3))
                correct += (mse < 0.1).sum().item()

            accuracy = 100 * correct / total
            print('Accuracy of the network on the {} test images: {} %'.format(total, accuracy))
            return accuracy  # SACRED: We return this so that we can add it to our MongoDB

    # SACRED: The parameter model_file comes from our Sacred config file. Sacred finds this because of the
    # @ex.capture handle. Note that we did not have to add these parameters when we called this method.
    @ex.capture
    def run(self, _run):
        # SACRED: we don't need any parameters for train and test, as they're in the config and the functions get a
        # @ex.capture handle later
        self.train()

        accuracy = self.test()

        model_file = f'{_run.experiment_info["name"]}_{_run._id}.pt'
        torch.save(self.model, model_file)
        print('Model saved in {}'.format(model_file))

        return accuracy


@ex.config
def get_config():
    """
    Where you would normally do something like:
    parser = argparse.ArgumentParser(...)
    parser.add_argument(...)
    ...
    Now you need to store all your parameters in a function called get_config().
    Put the @ex.config handle above it to ensure that Sacred knows this is the config function it needs to look at.
    """
    input_size = 1
    hidden_size = 32
    num_classes = 1
    num_epochs = 50  # SACRED: Have a look at train_nn.job for an example of how we can change parameter settings
    batch_size = 32
    learning_rate = 0.0001


@ex.main
def main(_run):
    """
    Sacred needs this main function, to start the experiment.
    If you want to import this experiment in another file (and use its configurations there, you can do that as follows:
    import train_nn
    ex = train_nn.ex
    Then you can use the 'ex' the same way we also do in this code.
    """

    trainer = Trainer()
    accuracy = trainer.run()

    return {'accuracy': accuracy}  # SACRED: Everything you return here is stored as a result,
    # and will be shown as such on Sacredboard


if __name__ == '__main__':
    ex.run_commandline()  # SACRED: this allows you to run Sacred not only from your terminal,
    # (but for example in PyCharm)