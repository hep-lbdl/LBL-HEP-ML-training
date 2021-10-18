import os
import sherpa
import pprint
import energyflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Get the current sherpa trial from the runner
client = sherpa.Client()
trial = client.get_trial()



# Show all hyperparameters for trial
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(trial.parameters)



# Get the available GPU from SHERPA
gpu = os.environ.get("SHERPA_RESOURCE", '')
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


# Set tensorflow memory options
tf.get_logger().setLevel(3)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



# Load the data and preprocess it
X, Y = energyflow.qg_nsubs.load(num_data=-1)

if trial.parameters["preprocessing"] == "log":
    X = np.log(X)

elif trial.parameters["preprocessing"] == "standardize":
    X = (X - X.mean(axis=0)) / X.std(axis=0)

elif trial.parameters["preprocessing"] == "min_max":
    X = (X - X.min(axis=0)) / X.max(axis=0)


# Train, Validation, Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=1/9, random_state=1) # 0.11 x 0.9 = 0.1

print("Number of Training Samples:", X_train.shape[0])
print("Number of Validation Samples:", X_val.shape[0])
print("Number of Testing Samples:", X_test.shape[0])


# Create the model
# Create the input layer to the network
x = model_input = tf.keras.layers.Input(shape=(45,))

# Create a series of fully connected layers
for i in range(trial.parameters["number_of_layers"]):

    if trial.parameters["batch_normalization"]:
            x = tf.keras.layers.BatchNormalization()(x)
        
    # Create a dense layer 
    x = tf.keras.layers.Dense(
        units=trial.parameters["number_of_nodes"],
        activation=trial.parameters["activation"]
    )(x)

    # Dropout layers with probablity 
    x = tf.keras.layers.Dropout(trial.parameters["dropout"])(x)

# Create the final layer in the network
model_output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# Build the model graph
model = tf.keras.Model(
    inputs=model_input, 
    outputs=model_output
)


# Compile the model with loss and optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=trial.parameters["learning_rate"]),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.TruePositives(name="true_positives"),
        tf.keras.metrics.FalsePositives(name="false_positives"),
    ]
)


# Create model callbacks for during training
callbacks = [
    client.keras_send_metrics(
        trial,
        objective_name="val_loss",
        context_names=["accuracy", "val_accuracy", "auc", "val_auc",
                       "true_positives", "val_true_positives",
                       "false_positives", "val_false_positives"]
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=trial.parameters['learning_rate_decay'],
        patience=25,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "ParallelResults/Models/%05d" % trial.id,
        save_best_only=True,
        monitor="val_loss"
    )
]


# Train the model
model.fit(
    x=X_train, 
    y=Y_train,
    epochs=100,
    verbose=2,
    batch_size=1024,
    validation_data=(X_val, Y_val),
    callbacks=callbacks
)
