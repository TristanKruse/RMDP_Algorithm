import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s [Line: %(lineno)d]')


class algorithm:
    def __init__(self):
       pass
    
    def sampling_level1(self):
        pass


    def update_networks_level(self):
        # return loss 
        pass 

    def shuffle_and_split_data(self, num_batches, *arrays):
        # Ensure all arrays have the same length
        assert all(len(array) == len(arrays[0]) for array in arrays), "All input arrays must have the same length"

        # Get the indices and shuffle them
        indices = tf.range(start=0, limit=tf.shape(arrays[0])[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        # Shuffle each array according to the shuffled indices
        shuffled_arrays = [tf.gather(array, shuffled_indices) for array in arrays]

        # Split each array into the specified number of batches
        batch_size = len(shuffled_arrays[0]) // num_batches
        split_arrays = [
            [array[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
            for array in shuffled_arrays
        ]

        # Combine the batches
        return list(zip(*split_arrays))


    def update_networks(self):

        return   # Optionally return the total average loss for monitoring purposes


    def evaluate_policy(self):
        return