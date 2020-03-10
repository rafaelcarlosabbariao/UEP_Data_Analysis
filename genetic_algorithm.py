# Genetic Algorithm on Neural Network

# GENERATE
from sklearn.model_selection import train_test_split
import csv
from functools import reduce
from operator import add
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
# from keras.layers import GaussianNoise
# from matplotlib import pyplot as plt

def train_networks(networks):
    for network in networks:
        network.train_network()

def get_average_accuracy(networks):
    total_loss = 0
    for network in networks:
        total_loss += network.loss

    return total_loss / len(networks)

def get_best_accuracy(networks):
    sorted_networks = sorted(networks, key=lambda x: x.loss)
    return sorted_networks[0]

# set defaults
batch_size = 10
epochs = 20
input_shape = (2,)

def compile_model(network, input_shape):
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']

    model = Sequential()

    # Add each layer
    for i in range(nb_layers):

        # need input shape
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
            # model.add(GaussianNoise(1, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))
            # model.add(GaussianNoise(1, input_shape=input_shape))

        model.add(Dropout(0.5))

    # output layer
    model.add(Dense(1, activation='elu'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.save("genalgonNNwith25noise.h5")

    return model

# get the data
x = [[], []]
y = []

with open('D:/Python/Machine Learning/Research/data25noise.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]}  {row[1]} {row[2]}')
            x[0].append(float(row[0]))
            x[1].append(float(row[1]))
            y.append(float(row[2]))
            line_count += 1

x = np.transpose(x)
y = np.transpose(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def train_and_score(network):
    model = compile_model(network, input_shape)

    tensorboard = TensorBoard(log_dir="D:/Python/Machine Learning/Research/logs/{}",
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None,
                              update_freq='epoch')

    filepath = "genalgonNNwith25noise.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    callbacks = [tensorboard, checkpoint]

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        callbacks=callbacks,
                        # validation_data=(x_test, y_test),
                        validation_split=0.2)

    score = model.evaluate(x_test, y_test, verbose=0)  # if verbose=1, loss shown is val loss

    print(history.history['loss']) # for each epoch
    print(history.history['val_loss'])
    print(network)
    print("Test loss: %.3f" % score)

    return score

class Network():
    def __init__(self, nn_param_choices=None):
        nn_param_choices = {'nb_neurons': [1, 3, 5, 7, 9, 12, 15, 18], 'nb_layers': [1, 2, 3, 4],
                            'activation': ['relu', 'elu'],
                            'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']}
        self.loss = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # list: represents MLP network

    def create_random_network(self):
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set_network(self, network):
        self.network = network

    def train_network(self):
        '''now trains the network and records the accuracy'''
        if self.loss == 0.:
            self.loss = train_and_score(self.network)

    def print_network(self):
        print("Network loss: %.3f" % (self.loss))
        print(self.network)

# OPTIMIZER
# class that holds a genetic algorithm for evolving a network

retain = 0.4
random_select = 0.1
mutate_chance = 0.005

class Optimizer():
    def __init__(self, nn_param_choices, retain, random_select, mutate_chance):

        # these are default
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        pop = []
        for _ in range(0, count):
            network = Network(self.nn_param_choices)
            network.create_random_network()
            pop.append(network)
        return pop

    def fitness(self, network):
        return network.loss

    def grade(self, pop):
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float(len(pop))

    def breed(self, mother, father):
        # decide to create two children
        children = []
        for _ in range(2):

            # breed the widths
            child = {}
            for param in self.nn_param_choices:
                child[param] = random.choice([mother.network[param],
                                              father.network[param]])

            # create network object
            network = Network(self.nn_param_choices)
            network.create_set_network(child)

            # randomly mutate some of the children
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        random_layer = random.choice(list(self.nn_param_choices.keys()))

        # update one layer with a random neuron count
        network.network[random_layer] = random.choice(self.nn_param_choices[random_layer])

        return network

    def evolve(self, pop):
        '''The main algorithm function. Evolve a population of networks.'''
        graded = [(self.fitness(network), network) for network in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]
        retain_length = int(len(graded)*self.retain)
        parents = graded[:retain_length]

        # randomly add other individuals to promote genetic diversity
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if self.mutate_chance > random.random():
                individual = self.mutate(individual)

        # crossover parents to create children
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        while len(children) < desired_length:
            male = random.randint(0, parents_length-1)
            female = random.randint(0,  parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                babies = self.breed(male, female)
                for baby in babies:
                    # don't grow larger than desired length
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents

generations = 10
population = 20

def generate(generations, networks, nn_param_choices):
    choice = None

    optimizer = Optimizer(nn_param_choices, retain, random_select, mutate_chance)
    networks = optimizer.create_population(networks)

    # Evolve the generation
    for i in range(generations):
        print("***Doing generation %d of %d***" % (i + 1, generations))

        # train and get accuracy for networks
        train_networks(networks)

        # Get avg accuracy for this generation
        average_loss = get_average_accuracy(networks)

        best_loss = get_best_accuracy(networks)

        # Print out avg accuracy of each generation
        print('-'*80)
        print("Generation average: %.3f" % (average_loss))
        print("Best loss: %.3f" % (best_loss.loss))
        print(best_loss.network)
        print('-'*80)

        # Evolve except for the last iteration
        if i != generations - 1:
            networks = optimizer.evolve(networks)

        if i == generations - 1:
            print(networks[0])

    # Sort population
    networks = sorted(networks, key=lambda x: x.loss, reverse=True)

    # Print the better performing ones
    print_networks(networks)
    print(networks[0])

def print_networks(networks):
    for network in networks:
        print('-' * 80)
        network.print_network()

def main():
    generations = 10
    population = 20
    nn_param_choices = {'nb_neurons': [1, 3, 5, 7, 9, 12], 'nb_layers': [1, 2],
                        'activation': ['relu', 'elu'],
                        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']}
    print("***Evolving %d generations with population %d***" % (generations, population))

    generate(generations, population, nn_param_choices)

if __name__ == '__main__':
    main()











