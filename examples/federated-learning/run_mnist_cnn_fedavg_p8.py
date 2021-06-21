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
    # tfe.set_protocol(tfe.protocol.SecureNN())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class ModelOwner:
    """Contains code meant to be executed by some `ModelOwner` Player.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """

    epochs = 21
    LEARNING_RATE = 0.1
    party = 8
    batchsize = 128
    ITERATIONS = 60000//party // batchsize *epochs
    
    #ITERATIONS = 200

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
        

        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        self.model.compile(optimizer=self.opt, loss=tf.keras.losses.sparse_categorical_crossentropy)
        
        self.params = self.model.trainable_variables
        self.weights = self.model.weights

    def _build_model(self, x, y):
        """Build the model function for federated learning.

    Includes loss calculation and backprop.
    """
    
        
        model=self.model
        
        predictions = model(x)
        params = model.trainable_variables
        weights = model.weights
            
        

        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y)
        )
        
        opt = self.opt
        
        grads_and_vars = opt.compute_gradients(loss,params)

        grads = list(zip(*grads_and_vars))[0]
        
        
        return predictions, loss, grads, params, weights

    def build_update_step(self, x, y):
        """Build a graph representing a single update step.

    This method will be called once by all data owners
    to create a local gradient computation on their machine.
    """
        _, _, grads, params, weights = self._build_model(x, y)
        return grads, params, weights

    def _build_validation_step(self, x, y):
        predictions, loss, _ , _, _= self._build_model(x, y)
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
    def reset_dataonwer_model(self, ops):
        return tf.group(ops)
        
    @tfe.local_computation
    def update_model(self, *weights):
        """Perform a single update step.

    This will be performed on the ModelOwner device
    after securely aggregating gradients.

    Args:
      *grads: `tf.Variables` representing the federally computed gradients.
    """
        opt = self.opt
        
            
        with tf.name_scope("update"):
            update_op = tf.group(
                *[
                    w.assign(tf.cast(weights[i], tf.float32))
                    for i, w in enumerate(self.model.weights)
                ]
            )
        

        with tf.name_scope("validate"):
            x, y = self._build_data_pipeline()
            y_hat, loss, accuracy = self._build_validation_step(x, y)
            
            with tf.control_dependencies([update_op]):
                print_loss = tf.print("loss", loss)
                print_accuracy = tf.print("accuracy", accuracy)
                print_fun = tf.print("update_model")
                return tf.group(print_loss, print_accuracy, print_fun)
            
    @tfe.local_computation
    def update_model_batch(self, *grads):
        """Perform a single update step.

    This will be performed on the ModelOwner device
    after securely aggregating gradients.

    Args:
      *grads: `tf.Variables` representing the federally computed gradients.
    """
        params = self.params
        weights = self.weights
        grads = [tf.cast(grad, tf.float32) for grad in grads]
        opt = self.opt
        
        update_op=opt.apply_gradients(zip(grads, params))
        '''with tf.name_scope("update"):
            update_op = tf.group(
                *[
                    param.assign(param - grad * self.LEARNING_RATE)
                    for param, grad in zip(params, grads)
                ]
            )'''
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

    def __init__(self, player_name, local_data_file, build_update_step, reset_dataonwer_model):
        self.player_name = player_name
        self.local_data_file = local_data_file
        self._build_update_step = build_update_step
        self._reset_dataonwer_model = reset_dataonwer_model
        
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
        
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        self.model.compile(optimizer=self.opt, loss=tf.keras.losses.sparse_categorical_crossentropy)
        
        self.params = self.model.trainable_variables
        self.weights = self.model.weights
        self.grads_list=[]
        self.grads_value=[]
        
        ITERATIONS=60000//8//128
        self.itr_count=tf.Variable(0, tf.int32)
        self.itr_max=tf.constant(ITERATIONS)
        self.itr_max2=tf.constant(ITERATIONS+1)
        
        with tf.name_scope("parameters"):
            for i, w in enumerate(self.params):
                w_shape=w.shape
                itr_=tf.reshape(self.itr_max2, (1,))
                g_shape=tf.concat([itr_,w_shape],axis=0)
                
                self.g = tf.Variable(tf.zeros(w.shape, tf.float32))
                self.grads_list.append(self.g)
                self.gg = tf.Variable(tf.zeros(w.shape, tf.float32))
                self.grads_value.append(self.gg)
        
        
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

    def build_update_step_local(self, x, y):

        
        model=self.model
        
        predictions = model(x)
        params = model.trainable_variables
        weights = model.weights
            
        

        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y)
        )
        opt = self.opt
        

        grads_and_vars = opt.compute_gradients(loss,params)

        grads = list(zip(*grads_and_vars))[0]
        
        return predictions, loss, grads, params, weights
    
    @tfe.local_computation
    def compute_gradient(self):
        """Compute gradient given current model parameters and local data."""
        
        
            
        with tf.name_scope("compute_gradient"):
            grads = [g.read_value() for g in self.grads_list]
            
        return grads
    
    @tfe.local_computation
    def compute_weights(self):
        """Compute gradient given current model parameters and local data."""
        
        
            
        with tf.name_scope("compute_weights"):
            weights = [w.read_value() for w in self.model.weights]
            
        return weights
    
    
    @tfe.local_computation
    def reset_gradient(self):
        """Compute gradient given current model parameters and local data."""
        
        
            
        with tf.name_scope("compute_gradient"):
            x, y = self._build_data_pipeline()
            m_grads, m_params, m_weights = self._build_update_step(x, y)
            
            
            setweights_op = tf.group(
                *[
                    w.assign(m_weights[i])
                    for i, w in enumerate(self.model.weights)
                ]
            )
            reset_op = tf.group(
                *[
                    g.assign(g*0)
                    for g in self.grads_list
                ]
            )
            print_result = tf.print("reset_gradient")
            tf_group = tf.group(setweights_op,reset_op, print_result)
            
        

            return tf_group
    
    @tfe.local_computation
    def reset_model_weight(self):
        """Compute gradient given current model parameters and local data."""
        with tf.name_scope("data_loading"):
            x, y = self._build_data_pipeline()

        with tf.name_scope("gradient_computation"):
            grads, params, weights = self._build_update_step(x, y)
            
        print(params)
        self.params=params
        self.weights=weights
        self.set_model_weights()
        self.grads=grads

        return grads
    
    
    @tfe.local_computation
    def compute_gradient_local(self):
        """Build the model function for federated learning.

    Includes loss calculation and backprop.
    """
    
        
        with tf.name_scope("data_loading"):
            x, y = self._build_data_pipeline()
        
        model=self.model
        
        predictions = model(x)
        params = model.trainable_variables
        weights = model.weights
            
        

        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y)
        )
        opt = self.opt
        
        grads_and_vars = opt.compute_gradients(loss,params)
        grads = list(zip(*grads_and_vars))[0]
        self.grads_value=grads
        
        
        for i, g in enumerate(grads):
            self.grads_list[i]=tf.add(self.grads_list[i], g)
        
        
        return grads
    
    @tfe.local_computation
    def compute_gradient_local333(self):
        """Build the model function for federated learning.

    Includes loss calculation and backprop.
    """
    
        
        with tf.name_scope("data_loading"):
            x, y = self._build_data_pipeline()
            
        with tf.name_scope("build_update_step_local"):
            predictions, loss, grads, params, weights = self.build_update_step_local(x, y)
            
        
        with tf.name_scope("assign_gradient"):
            assign_op = tf.group(
                *[
                    #g.assign(g+grads[i])
                    tf.compat.v1.assign(g, g+grads[i])
                    for i, g in enumerate(self.grads_list)
                ]
            )
            assign_op2 = tf.group(
                *[
                    tf.compat.v1.assign(gg, grads[i])
                    for i, gg in enumerate(self.grads_value)
                ]
            )
            
        
        with tf.name_scope("assign_gradient2"):

            with tf.control_dependencies([assign_op,assign_op2]):
                print_loss = tf.print("loss", self.grads_list[0])
                print_fun = tf.print("compute_gradient_local333",self.itr_count)
                return tf.group(print_loss,print_fun)
        
        
    



                
    
    @tfe.local_computation
    def update_model_batch(self):
        """Perform a single update step.

    This will be performed on the ModelOwner device
    after securely aggregating gradients.

    Args:
      *grads: `tf.Variables` representing the federally computed gradients.
    """
        params = self.params
        weights = self.weights
        grads= self.grads_value
        grads = [tf.cast(grad, tf.float32) for grad in grads]
        opt = self.opt
        
        update_op=opt.apply_gradients(zip(grads, params))
        with tf.name_scope("validate"):
            with tf.control_dependencies([update_op]):
                
                print_loss = tf.print('update_model_batch',self.itr_count)
                itr_plus=tf.assign_add(self.itr_count,1)
                return tf.group(print_loss,itr_plus)
    
    @tfe.local_computation
    def set_model_weights(self):
        
        
        with tf.name_scope("data_loading"):
            x, y = self._build_data_pipeline()

        with tf.name_scope("gradient_computation"):
            grads, params, weights = self._build_update_step(x, y)
            
        
        with tf.name_scope("set_model_weights"):
            reset_op = tf.group(
                *[
                    w.assign(weights[i])
                    for i, w in enumerate(self.model.weights)
                ]
            )
        
        with tf.name_scope("set_model_weights2"):

            with tf.control_dependencies([reset_op]):
                return tf.group()
        
        

import time
if __name__ == "__main__":
    
    starttime=time.time()

    logging.basicConfig(level=logging.DEBUG)

    model_owner = ModelOwner("model-owner")
    data_owners = [
        DataOwner(
            "data-owner-0", "./MNIST_DATA/8Parties/data_party0.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
        DataOwner(
            "data-owner-1", "./MNIST_DATA/8Parties/data_party1.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
        DataOwner(
            "data-owner-2", "./MNIST_DATA/8Parties/data_party2.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
        DataOwner(
            "data-owner-3", "./MNIST_DATA/8Parties/data_party3.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
        DataOwner(
            "data-owner-4", "./MNIST_DATA/8Parties/data_party4.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
        DataOwner(
            "data-owner-5", "./MNIST_DATA/8Parties/data_party5.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
        DataOwner(
            "data-owner-6", "./MNIST_DATA/8Parties/data_party6.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
        DataOwner(
            "data-owner-7", "./MNIST_DATA/8Parties/data_party7.npz", model_owner.build_update_step, model_owner.reset_dataonwer_model
        ),
    ]

    model_weights = zip(*(data_owner.compute_weights() for data_owner in data_owners))

        
    with tf.name_scope("secure_aggregation"):
        aggregated_model_weights = [
            tfe.add_n(weights) / len(weights) for weights in model_weights
        ]

    
    
    
    compute_gradient_ops =model_owner.reset_dataonwer_model([data_owner.compute_gradient_local333() for data_owner in data_owners])
    
    with tf.name_scope("iteration_op1"):
        with tf.control_dependencies([compute_gradient_ops]):
            iteration_op1 = model_owner.reset_dataonwer_model([data_owner.update_model_batch() for data_owner in data_owners])

    with tf.name_scope("iteration_op2"):
        with tf.control_dependencies([iteration_op1]):
            iteration_op2 = model_owner.update_model(*aggregated_model_weights)
            
    with tf.name_scope("iteration_op3"):
        with tf.control_dependencies([iteration_op2]):
            iteration_op3 =model_owner.reset_dataonwer_model([data_owner.reset_gradient() for data_owner in data_owners])
            
    

    fusion_freq=5
    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")
        
        sess.run(model_owner.reset_dataonwer_model([data_owner.reset_gradient() for data_owner in data_owners]))

        for i in range(model_owner.ITERATIONS*fusion_freq):
            # print('i',i)
            iter_each_epoch=60000//128//8*fusion_freq
            epoch=i//iter_each_epoch
                
            if i % iter_each_epoch == 0 and i>0:
                print("Epoch {} time: {} s".format(epoch, time.time()-starttime))
                sess.run(iteration_op3, tag="iteration2")
                
            else:
                sess.run(iteration_op1, tag="iteration1")
                
        print('End time: {}'.format(time.time()-starttime))
