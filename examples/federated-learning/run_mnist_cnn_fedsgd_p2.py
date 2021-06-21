"""An example of the secure aggregation protocol for federated learning."""

import logging
import sys

import tensorflow as tf
import numpy as np

import tf_encrypted as tfe
import tensorflow.keras as keras
from tf_encrypted.keras import backend as KE
from convert import decode

if len(sys.argv) > 1:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.Pond())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class ModelOwner:
    """Contains code meant to be executed by some `ModelOwner` Player.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """

    epochs = 21
    LEARNING_RATE = 0.1
    party = 2
    batchsize = 128
    ITERATIONS = 60000//party // batchsize *epochs
    

    def __init__(self, player_name):
        self.player_name = player_name

        with tf.device(tfe.get_config().get_player(player_name).device_name):
            self._initialize_weights()

    def _initialize_weights(self):

        num_classes = 10
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=(28, 28, 1), activation='relu', padding='same'))
        self.model.add(keras.layers.MaxPool2D(pool_size=2))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(256, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        
        self.opt = tf.train.AdamOptimizer()
        self.model.compile(optimizer=self.opt, loss=tf.keras.losses.sparse_categorical_crossentropy)
        
        self.params = self.model.trainable_variables

    def _build_model(self, x, y):
        """Build the model function for federated learning.

    Includes loss calculation and backprop.
    """
    
        
        model=self.model
        
        predictions = model(x)
        params = model.trainable_variables
            
        

        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y)
        )

        opt = self.opt
        
        grads_and_vars = opt.compute_gradients(loss,params)
        grads = list(zip(*grads_and_vars))[0]
        return predictions, loss, grads

    def build_update_step(self, x, y):
        """Build a graph representing a single update step.

    This method will be called once by all data owners
    to create a local gradient computation on their machine.
    """
        _, _, grads = self._build_model(x, y)
        return grads

    def _build_validation_step(self, x, y):
        predictions, loss, _ = self._build_model(x, y)
        most_likely = tf.argmax(predictions, axis=1)
        equality = tf.math.equal(most_likely, y)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
        return most_likely, loss, accuracy

    def _build_data_pipeline(self):
        """Build data pipeline for validation by model owner."""

        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, [28,28,1])
            return image, label


        with np.load("./MNIST_DATA/2Parties/data_party0.npz") as data:
            train_examples = data['x_train']
            train_labels = data['y_train'].astype(np.int64)
            test_examples = data['x_test']
            test_labels = data['y_test'].astype(np.int64)

        dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

        dataset = dataset.map(normalize)
        dataset = dataset.batch(10000//2)
        dataset = dataset.take(1)  # keep validating on the same items
        dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    @tfe.local_computation
    def update_model(self, *grads):
        """Perform a single update step.

    This will be performed on the ModelOwner device
    after securely aggregating gradients.

    Args:
      *grads: `tf.Variables` representing the federally computed gradients.
    """
        params = self.params
        grads = [tf.cast(grad, tf.float32) for grad in grads]
        opt = self.opt
        
        update_op=opt.apply_gradients(zip(grads, params))

            
        

        with tf.name_scope("validate"):
            x, y = self._build_data_pipeline()
            y_hat, loss, accuracy = self._build_validation_step(x, y)

            with tf.control_dependencies([update_op]):
                print_loss = tf.print("loss", loss)
                print_accuracy = tf.print("accuracy", accuracy)
                return tf.group(print_loss, print_accuracy)
            
            
    @tfe.local_computation
    def update_model_batch(self, *grads):
        """Perform a single update step.

    This will be performed on the ModelOwner device
    after securely aggregating gradients.

    Args:
      *grads: `tf.Variables` representing the federally computed gradients.
    """
        params = self.params
        grads = [tf.cast(grad, tf.float32) for grad in grads]
        opt = self.opt
        
        update_op=opt.apply_gradients(zip(grads, params))
        with tf.name_scope("validate"):
            with tf.control_dependencies([update_op]):
                return tf.group()


class DataOwner:
    """Contains methods meant to be executed by a data owner.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

    BATCH_SIZE = 128

    def __init__(self, player_name, local_data_file, build_update_step):
        self.player_name = player_name
        self.local_data_file = local_data_file
        self._build_update_step = build_update_step

    def _build_data_pipeline(self):
        """Build local data pipeline for federated DataOwners."""

        def normalize(image, label):
            image = tf.cast(image, tf.float32) 
            image = tf.reshape(image, [28,28,1])
            return image, label


        with np.load(self.local_data_file) as data:
            train_examples = data['x_train']
            train_labels = data['y_train'].astype(np.int64)
            test_examples = data['x_test']
            test_labels = data['y_test'].astype(np.int64)

        dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        dataset = dataset.map(normalize)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    @tfe.local_computation
    def compute_gradient(self):
        """Compute gradient given current model parameters and local data."""
        with tf.name_scope("data_loading"):
            x, y = self._build_data_pipeline()

        with tf.name_scope("gradient_computation"):
            grads = self._build_update_step(x, y)

        return grads

import time
if __name__ == "__main__":
    
    starttime=time.time()

    logging.basicConfig(level=logging.DEBUG)

    model_owner = ModelOwner("model-owner")
    data_owners = [
        DataOwner(
            "data-owner-0", "./MNIST_DATA/2Parties/data_party0.npz", model_owner.build_update_step
        ),
        DataOwner(
            "data-owner-1", "./MNIST_DATA/2Parties/data_party1.npz", model_owner.build_update_step
        ),

    ]

    model_grads = zip(*(data_owner.compute_gradient() for data_owner in data_owners))

    with tf.name_scope("secure_aggregation"):
        aggregated_model_grads = [
            tfe.add_n(grads) / len(grads) for grads in model_grads
        ]

    iteration_op = model_owner.update_model(*aggregated_model_grads)
    iteration_op_batch = model_owner.update_model_batch(*aggregated_model_grads)

    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")

        for i in range(model_owner.ITERATIONS):
            iter_each_epoch=60000//128//2
            epoch=i//iter_each_epoch
            if i % iter_each_epoch == 0:
                print("Epoch {} time: {} s".format(epoch, time.time()-starttime))
                sess.run(iteration_op, tag="iteration")
            else:
                sess.run(iteration_op_batch)
                
        print('End time: {}'.format(time.time()-starttime))
