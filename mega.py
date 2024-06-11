import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.models import Sequential
import random
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
import model.resnet as resnets
import model.vgg as vgg
from sklearn.model_selection import train_test_split

def xception_model():
    base_model = Xception(weights=None, include_top=False, input_tensor=Input(shape=(32, 32, 3)))
    base_model.trainable = True  # 기본 모델의 가중치를 학습에 포함

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def densenet121():
    base_model = DenseNet121(include_top=False, weights=None, input_tensor=Input(shape=(32, 32, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def densenet169():
    base_model = DenseNet169(include_top=False, weights=None, input_tensor=Input(shape=(32, 32, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def xception():
    model = xception_model()
    return model

def vgg16():
    model = vgg.VGG16(input_shape=(32, 32, 3), num_classes = 10)
    return model

def vgg19():
    model = vgg.VGG19(input_shape=(32, 32, 3), num_classes = 10)
    return model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = '1'
    tf.random.set_seed(seed)

seed_everything(46)
# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
original_x_train = x_train 
original_y_train = y_train
x_test = x_test.astype('float32') / 255

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=46)

# Define models
model1 = resnets.ResNet56(input_shape=(32, 32, 3), num_classes=10)
model2 = resnets.ResNet56(input_shape=(32, 32, 3), num_classes=10)

# Compile the models
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the models
history1 = model1.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_val, y_val))
history2 = model2.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_val, y_val))

print()
print()
print()
print()
test_loss_fusion, test_acc_fusion = model1.evaluate(x_test, y_test, verbose=2)
print(f"Model 1 Test Accuracy: {test_acc_fusion}")
test_loss_fusion, test_acc_fusion = model2.evaluate(x_test, y_test, verbose=2)
print(f"Model 2 Test Accuracy: {test_acc_fusion}")

# Function to create initial population
def create_population(size, weights1, weights2):
    population = []
    for _ in range(size):
        alpha = random.random()
        individual = [(1 - alpha) * w1 + alpha * w2 for w1, w2 in zip(weights1, weights2)]
        population.append(individual)
    return population

# Function to evaluate fitness of an individual
def evaluate_fitness(model, weights, x_test, y_test):
    model.set_weights(weights)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc

# Tournament selection
def tournament_selection(population, fitnesses, k):
    selected = []
    for _ in range(k):
        tournament = random.sample(list(zip(population, fitnesses)), 3)
        tournament.sort(key=lambda x: x[1], reverse=True)
        selected.append(tournament[0][0])
    return selected

# Roulette wheel selection
def roulette_wheel_selection(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    selected = random.choices(population, weights=selection_probs, k=num_parents)
    return selected

# Function for crossover
def crossover(parents):
    alpha = random.random()
    child = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parents[0], parents[1])]
    return child

# Function for mutation
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            noise = np.random.normal(0, 0.1, individual[i].shape)
            individual[i] += noise
    return individual

# Genetic Algorithm parameters
population_size = 20  # Increase population size
num_generations = 20  # Increase number of generations
num_parents = 4  # Increase number of parents for better genetic diversity
mutation_rate = 0.02  # Slightly higher mutation rate

# Initialize model for fusion
model_fusion = resnets.ResNet56(input_shape=(32, 32, 3), num_classes=10)
model_fusion.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create initial population
population = create_population(population_size, model1.get_weights(), model2.get_weights())

# Genetic Algorithm
best_individual = None
best_fitness = 0

for generation in range(num_generations):
    fitnesses = [evaluate_fitness(model_fusion, individual, x_val, y_val) for individual in population]
    
    # Update best individual
    max_fitness = max(fitnesses)
    if max_fitness > best_fitness:
        best_fitness = max_fitness
        best_individual = population[np.argmax(fitnesses)]
    
    next_population = [best_individual]  # Elitism: Keep the best individual
    for _ in range(population_size - 1):
        # You can choose either tournament_selection or roulette_wheel_selection here
        parents = tournament_selection(population, fitnesses, num_parents)
        # parents = roulette_wheel_selection(population, fitnesses, num_parents)
        child = crossover(parents)
        child = mutation(child, mutation_rate)
        next_population.append(child)
    
    population = next_population
    print(f'Generation {generation+1} - Best Fitness: {best_fitness}')

# Set best weights to the fusion model
model_fusion.set_weights(best_individual)

test_loss_fusion, test_acc_fusion = model_fusion.evaluate(x_test, y_test, verbose=2)
print(f"Model Fusion Test Accuracy: {test_acc_fusion}")

