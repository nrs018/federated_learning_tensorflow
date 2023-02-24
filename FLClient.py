import collections
from sys import argv

import attr
import functools
import numpy as np
import tensorflow as tf

import tensorflow_federated as tff
import os
import FLConfig
from datetime import datetime
import os
tf.get_logger().setLevel('ERROR')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(None)



def main(argv):
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    def preprocess(dataset):
        def batch_format_fn(element):
            """Flatten a batch of EMNIST data and return a (features, label) tuple."""
            return (tf.reshape(element['pixels'], [-1, 784]),
                    tf.reshape(element['label'], [-1, 1]))

        # return dataset.batch(BATCH_SIZE).map(batch_format_fn)
        return dataset.repeat(FLConfig.NUM_EPOCHS).shuffle(FLConfig.SHUFFLE_BUFFER).batch(FLConfig.BATCH_SIZE).map(
            batch_format_fn).prefetch(
            FLConfig.PREFETCH_BUFFER)

    # client_ids = np.random.choice(emnist_train.client_ids, size=NUM_CLIENTS, replace=False)
    client_ids = []
    client_ids.append(argv[1])
    # print(client_ids)
    federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x)) for x in client_ids]

    def create_keras_model():

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
        ])
        return model
        #return tf.keras.models.Sequential([
        #    tf.keras.layers.Input(shape=(784,)),
        #    tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        #    tf.keras.layers.Softmax(),
        #])

    def model_fn():
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def server_model_fn():
        keras_model = create_keras_model()
        keras_model.load_weights('/home/narisu/src/TFF/Model/globalModel.h5')
        # print(keras_model.get_weights())
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def initialize_fn():
        model = model_fn()

    @tf.function
    def client_update(model, dataset, server_weights, client_optimizer):
        """Performs training (using the server model weights) on the client's dataset."""
        # Initialize the client model with the current server weights.
        client_weights = model.trainable_variables
        # Assign the server weights to the client model.
        tf.nest.map_structure(lambda x, y: x.assign(y),
                              client_weights, server_weights)

        # Use the client_optimizer to update the local model.
        for batch in dataset:
            with tf.GradientTape() as tape:
                # Compute a forward pass on the batch of data
                outputs = model.forward_pass(batch)

            # Compute the corresponding gradient
            grads = tape.gradient(outputs.loss, client_weights)
            grads_and_vars = zip(grads, client_weights)

            # Apply the gradient using a client optimizer.
            client_optimizer.apply_gradients(grads_and_vars)

        return client_weights

    @tf.function
    def server_update(model, mean_client_weights):
        """Updates the server model weights as the average of the client model weights."""
        model_weights = model.trainable_variables
        # Assign the mean client weights to the server model.
        tf.nest.map_structure(lambda x, y: x.assign(y),
                              model_weights, mean_client_weights)
        return model_weights

    federated_float_on_clients = tff.FederatedType(tf.float32, tff.CLIENTS)

    @tff.tf_computation
    def server_init():
        model = server_model_fn()

        return model.trainable_variables

    @tff.federated_computation
    def initialize_fn():
        return tff.federated_value(server_init(), tff.SERVER)

    whimsy_model = model_fn()
    tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)
    model_weights_type = server_init.type_signature.result

    @tff.tf_computation(tf_dataset_type, model_weights_type)
    def client_update_fn(tf_dataset, server_weights):
        model = model_fn()
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        return client_update(model, tf_dataset, server_weights, client_optimizer)

    @tff.tf_computation(model_weights_type)
    def server_update_fn(mean_client_weights):
        model = model_fn()
        return server_update(model, mean_client_weights)

    federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
    federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

    @tff.federated_computation(federated_server_type, federated_dataset_type)
    def next_fn(server_weights, federated_dataset):
        # Broadcast the server weights to the clients.
        server_weights_at_client = tff.federated_broadcast(server_weights)

        # Each client computes their updated weights.
        # client_weights的类型为tensorflow_federated.python.core.impl.federated_context.value_impl.Value
        # Value是comp：building_blocks.ComputationBuildingBlock 的实例，其中包含计算此值的逻辑。
        client_weights = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))

        # The server averages these updates.
        mean_client_weights = tff.federated_mean(client_weights)

        # The server updates its model.
        server_weights = tff.federated_map(server_update_fn, mean_client_weights)

        return server_weights

    central_emnist_test = emnist_test.create_tf_dataset_from_all_clients().take(1000)
    central_emnist_test = preprocess(central_emnist_test)

    def evaluate(server_state):
        keras_model = create_keras_model()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        keras_model.set_weights(server_state)
        # 保存client模型参数
        # 保存形式为client_id + '.h5'
        keras_model.save(str('/home/narisu/src/TFF/Model/' + client_ids[0]) + '.h5')
        print(client_ids[0], ' saved successfully!')
        keras_model.set_weights(server_state)
        # keras_model.evaluate(central_emnist_test)

    federated_algorithm = tff.templates.IterativeProcess(initialize_fn=initialize_fn, next_fn=next_fn)
    server_state = federated_algorithm.initialize()

    def x_server_model_fn():
        keras_model = create_keras_model()
        # print('this server init')
        keras_model.load_weights('/home/narisu/src/TFF/Model/globalModel.h5')
        #keras_model.summary()
        return keras_model.get_weights()

    server_state = x_server_model_fn()
    start_time = datetime.utcnow()
    for round in range(1):
        # print(server_state)
        server_state = federated_algorithm.next(server_state, federated_train_data)
        # print(server_state)
        evaluate(server_state)
    end_time = datetime.utcnow()
    # print(' bu bao han tensorflow qidong shijian ', end_time - start_time)


if __name__ == "__main__":
    start_time = datetime.utcnow()
    # execute only if run as a script
    main(argv)
    end_time = datetime.utcnow()
    # print('xxxxxxx  bao han tensorflow qidong shijian ', end_time - start_time)
