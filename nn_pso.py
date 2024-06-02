import random
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from keras.losses import BinaryCrossentropy # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
from tools import MetricsHistory

NEURONS_RANGE = (5, 100)

TF_ENABLE_ONEDNN_OPTS=0

def create_model(neurons_per_layer):
    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu'))
    for neurons in neurons_per_layer:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def random_hyperparameters(n_layers):
    hyperparameters = []
    for _ in range(n_layers - 1):
        neurons = random.randint(*NEURONS_RANGE)
        hyperparameters.append(neurons)
    hyperparameters.append(random.randint(*NEURONS_RANGE))
    return hyperparameters


def random_velocity():
    v_neurons = random.uniform(-1, 1)
    return v_neurons


def update_velocity(particle, global_best_position, w=0.5, c1=1, c2=2):
    new_velocity = []
    for i, v_neurons in enumerate(particle.velocity):
        r1 = random.random()
        r2 = random.random()
        best_neurons = particle.best_position[i]
        current_neurons = particle.position[i]
        if global_best_position is not None and i < len(global_best_position):
            global_best_neurons = global_best_position[i]
        else:
            global_best_neurons = current_neurons
        cognitive_velocity_neurons = c1 * r1 * (best_neurons - current_neurons)
        social_velocity_neurons = c2 * r2 * (global_best_neurons - current_neurons)
        v_neurons_updated = w * v_neurons + cognitive_velocity_neurons + social_velocity_neurons
        new_velocity.append(v_neurons_updated)
    particle.velocity = new_velocity

def update_position(particle):
    new_position = []
    for i, v_neurons in enumerate(particle.velocity):
        current_neurons = particle.position[i]
        new_neurons = max(min(current_neurons + v_neurons, NEURONS_RANGE[1]), NEURONS_RANGE[0])
        new_position.append(int(round(new_neurons)))
    particle.position = new_position


class Particle:
    def __init__(self, n_layers):
        self.position = random_hyperparameters(n_layers)
        self.velocity = [random_velocity() for _ in range(n_layers)]
        self.best_position = self.position
        self.best_error = float('inf')


def pso(n_particles, n_iterations, X_train, y_train, X_val, y_val, X_test, y_test):
    global_best_position = None
    global_best_error = float('inf')
    no_improve_count = 0
    particles = [Particle(random.randint(2, 5)) for _ in range(n_particles)]
    best_metrics_history = None
    all_accuracies = []
    all_val_losses = []
    
    for iteration in range(n_iterations):
        improved = False
        for particle in particles:
            model = create_model(particle.position)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
            metrics_history = MetricsHistory()
            history_best = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_val, y_val), callbacks=[es, metrics_history], shuffle=True)
            predictions = model.predict(X_test)
            bce = BinaryCrossentropy(from_logits=False)
            error = bce(y_test, predictions).numpy()
            if error < particle.best_error:
                particle.best_error = error
                particle.best_position = particle.position
            if error < global_best_error:
                global_best_error = error
                global_best_position = particle.best_position.copy()
                best_metrics_history = metrics_history
                model.save('output_files/pso_model.h5')
                plt.figure(figsize=(12, 6))
                plt.plot(history_best.history['loss'])
                plt.plot(history_best.history['val_loss'])
                plt.ylabel('Strata')
                plt.xlabel('Epoka')
                plt.legend(['Training set', 'Validation set'], loc='upper right')
                plt.savefig('output_files/pso_best_model_history.png')
                plt.close()
                improved = True
                no_improve_count = 0
            
            all_accuracies.append(metrics_history.accuracy)
            all_val_losses.append(metrics_history.val_loss)

        if not improved:
            no_improve_count += 1

        if no_improve_count >= 5:
            print(f"Brak poprawy globalnej straty przez {iteration+1}. Zatrzymywanie działania algorytmu PSO...")
            return global_best_position, best_metrics_history, all_accuracies, all_val_losses

        for particle in particles:
            update_velocity(particle, global_best_position)
            update_position(particle)
        print(f"Iteration {iteration+1}, Best Error: {global_best_error}")
    return global_best_position, best_metrics_history, all_accuracies, all_val_losses



