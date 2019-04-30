import tensorflow as tf

if __name__ == "__main__":
    # Load and prepare data into a test and validation set
    mnist = tf.keras.datasets.mnist
    (training_inputs, training_labels), (test_inputs, test_labels) = mnist.load_data()
    training_inputs, test_inputs = training_inputs / 255.0, test_inputs / 255.0

    # Create the neural network, model based on example found on the internet which reached about 98%.
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains the model
    # Running more than one epochs does not have a major impact on accuracy.
    #   At one run it already has 96.7%, with two or maybe three runs this is bumped up to about 97.6%
    model.fit(training_inputs, training_labels, epochs=3)

    # Validate the model and print loss/accuracy rates
    model.evaluate(test_inputs, test_labels)
