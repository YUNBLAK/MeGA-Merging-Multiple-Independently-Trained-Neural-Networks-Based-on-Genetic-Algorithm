import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from model.resnet import ResNet56
from model.resnet import ResNet110
from model.resnet import ResNet152
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = '1'
    tf.random.set_seed(seed)

def save_model(model, path):
    model.save(path)
    print(f"Model saved at {path}")

def load_model(path):
    model = tf.keras.models.load_model(path, custom_objects={'ResNet152': ResNet152})
    print(f"Model loaded from {path}")
    return model

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


# Function to perform genetic algorithm on pairs of models
def genetic_algorithm(model1, model2, x_test, y_test, num_generations, population_size, num_parents, mutation_rate, save_path):
    if os.path.exists(save_path):
        return load_model(save_path)
    
    model_fusion = ResNet152(input_shape=(32, 32, 3), num_classes=10)
    model_fusion.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    population = create_population(population_size, model1.get_weights(), model2.get_weights())

    best_individual = None
    best_fitness = 0

    for generation in range(num_generations):
        fitnesses = [evaluate_fitness(model_fusion, individual, x_test, y_test) for individual in population]
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[np.argmax(fitnesses)]
        next_population = [best_individual]
        for _ in range(population_size - 1):
            parents = tournament_selection(population, fitnesses, num_parents)
            child = crossover(parents)
            child = mutation(child, mutation_rate)
            next_population.append(child)
        population = next_population
        print(f'Generation {generation+1} - Best Fitness: {best_fitness}')

    model_fusion.set_weights(best_individual)
    save_model(model_fusion, save_path)
    return model_fusion

# Perform hierarchical merging
def hierarchical_merging(model_paths, x_test, y_test):
    iteration = 0
    while len(model_paths) > 1:
        new_model_paths = []
        for i in range(0, len(model_paths), 2):
            if i + 1 < len(model_paths):
                iteration += 1
                merged_model_path = f"merged_model_{iteration}.h5"
                if os.path.exists(merged_model_path):
                    print(f"Merged model already exists: {merged_model_path}")
                    new_model_paths.append(merged_model_path)
                    continue
                print(f"Merging models {model_paths[i]} and {model_paths[i+1]}")
                model1 = load_model(model_paths[i])
                model2 = load_model(model_paths[i+1])
                genetic_algorithm(model1, model2, x_test, y_test, num_generations, population_size, num_parents, mutation_rate, merged_model_path)
                new_model_paths.append(merged_model_path)
        model_paths = new_model_paths
    return model_paths[0]

seed_everything(46)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Define and compile 8 ResNet56 models
model_paths = [f"resnet152_model_{i}.h5" for i in range(8)]
models = [ResNet152(input_shape=(32, 32, 3), num_classes=10) for _ in range(8)]

for i, model in enumerate(models):
    if not os.path.exists(model_paths[i]):
        model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Training model {i+1}")
        model.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_test, y_test))
        save_model(model, model_paths[i])
    else:
        models[i] = load_model(model_paths[i])

# Genetic Algorithm parameters
population_size = 20
num_generations = 20
num_parents = 4
mutation_rate = 0.02

# Hierarchical merging of all models
final_model_path = hierarchical_merging(model_paths, x_test, y_test)
final_model = load_model(final_model_path)

# Fine-tune the final fusion model
final_model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the final fusion model
test_loss_fusion, test_acc_fusion = final_model.evaluate(x_test, y_test, verbose=2)
print(f"Final Model Fusion Test Accuracy after Fine-tuning: {test_acc_fusion}")

# Save the final fusion model
save_model(final_model, "final_fusion_model.h5")
