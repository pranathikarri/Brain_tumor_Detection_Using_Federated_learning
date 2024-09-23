import pickle
import numpy as np
import os


def aggregate_weights(path):
    weights_directory = path
    weight_files = [f for f in os.listdir(weights_directory) if f.endswith('.h5')]  # Use `.pkl` for deserialization
    if not weight_files:
        print("No weight files found.")
        return None  # Return None if no weights are found
    
    all_weights = []
    for file in weight_files:
        file_path = os.path.join(weights_directory, file)
        # Load the serialized weights from the file
        with open(file_path, 'rb') as f:
            weights = pickle.load(f)  # Deserialize the weights
            all_weights.append(weights)
    
    # Ensure that all_weights is a list of weight arrays
    average_weights = [np.mean(layer, axis=0) for layer in zip(*all_weights)]  
     # Save the aggregated weights separately
    with open('./aggregated_weights.h5', 'wb') as f:
        pickle.dump(average_weights, f)
    print("Aggregated weights saved as aggregated_weights.h5")
    return average_weights