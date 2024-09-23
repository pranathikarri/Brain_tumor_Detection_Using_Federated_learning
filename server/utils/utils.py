import pickle


def save_model_weights(client_id, weights):
    # Serialize the weights list using pickle
    filename = f"./received_weights/Client_{client_id}.weights.h5"  # Use `.pkl` for serialization
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)  # Serialize the weights before saving
    print(f"Saved weights as {filename}")




def save_trained_model(global_model,average_weights):
    if average_weights is None:
        print("No weights to aggregate. Global model not updated.")
        return
    
    # Set the weights to the global model
    global_model.set_weights(average_weights)
    print("Averaged weights assigned to the global model.")
    
    # Save the global model
    with open('./trained_global_model.pkl', 'wb') as f:
        pickle.dump(global_model, f)  # Save the entire model object
    print("Global model saved as trained_global_model.pkl")
