import asyncio
import pickle
from user_input.input_model import Input_Model
import websockets
from cryptography.fernet import Fernet
import os

from aggregation.aggregation import aggregate_weights
from security.cryptography import key_generate, encrypt_obj, decrypt_obj
from utils.utils import save_model_weights, save_trained_model


HOST = '192.168.43.248'
PORT = 8000

NUM_CLIENTS = 5
client_counter = 0

FILE_PATH = "./user_input/data_processing_and_training.py"

global_model = Input_Model()
print("GLobal Model")
print(global_model.summary())



# Ensure the 'received_files' directory exists
if not os.path.exists("./received_weights"):
    os.makedirs("./received_weights")

#Key generation
cipher_suite = key_generate()



async def send_model(websocket, file):
    try:
        model = Input_Model()
        serialized_model = pickle.dumps(model)
        encrypted_model = encrypt_obj(cipher_suite,serialized_model)

        await websocket.send(str(len(encrypted_model)).encode())
        
        # Send in chunks
        chunk_size = 1024  # Adjust this size as needed
        for i in range(0, len(encrypted_model), chunk_size):
            await websocket.send(encrypted_model[i:i + chunk_size])

        print("Model sent successfully.")

        # Sending the file
        try:
            with open(file, 'rb') as f:
                file_data = f.read()
            encrypted_file = encrypt_obj(cipher_suite,file_data)

            await websocket.send(str(len(encrypted_file)).encode())

            # Send in chunks
            for i in range(0, len(encrypted_file), chunk_size):
                await websocket.send(encrypted_file[i:i + chunk_size])

            print("File sent successfully.")
        except Exception as e:
            print(f"Error during file sending: {e}")

    except Exception as e:
        print(f"Error during data sending: {e}")

    

async def receive_data_from_client(websocket):
    global client_counter
    try:
        # Receive client ID
        client_id = await websocket.recv()
        print(f"Received client ID: {client_id}")

        # Receive the size of encrypted weights
        encrypted_weights_size = int(await websocket.recv())
        print(f"Expected encrypted weights size: {encrypted_weights_size} bytes")

        # Receive the encrypted weights data in chunks
        encrypted_weights = bytearray()  # Ensure we're using a bytearray for combining chunks
        while len(encrypted_weights) < encrypted_weights_size:
            chunk = await websocket.recv()
            if isinstance(chunk, bytes):
                encrypted_weights.extend(chunk)  # Extend bytearray with the received chunk
            else:
                print(f"Warning: Received chunk is not in bytes format. Type: {type(chunk)}")

        # Ensure the received data length matches the expected length
        if len(encrypted_weights) != encrypted_weights_size:
            print(f"Error: Received data size {len(encrypted_weights)} does not match expected size {encrypted_weights_size}")
            return

        # Decrypt the received weights data
        decrypted_weights = decrypt_obj(cipher_suite,bytes(encrypted_weights))  # Convert to bytes before decryption
        # Deserialize the decrypted weights
        client_model_weights = pickle.loads(decrypted_weights)
        print(f"Received and decrypted model weights from client {client_id}")
        
        # Save the weights to an .h5 file
        try:
          save_model_weights(client_id, client_model_weights)
          client_counter+=1
        except Exception as e:
            print(f"Error in saving model ; {e}")
    except Exception as e:
        print(f"Error during data reception: {e}")

async def handle_client(websocket, path):
    try:
        await send_model(websocket, FILE_PATH)
        await receive_data_from_client(websocket)

        if client_counter == NUM_CLIENTS:
         aw = aggregate_weights("./received_weights")
         save_trained_model(global_model,aw)
         print("All clients' data received and aggregated. Server will stop now.")
         await stop_server()
    except Exception as e:
        print(f"Error in handling client: {e}")
    
async def stop_server():
    print("Stopping server...")
    # Cancel all running tasks except for this one
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    # Allow tasks to cancel gracefully
    await asyncio.gather(*tasks, return_exceptions=True)
    print("Server stopped gracefully.")

async def start_server():
    try:
        async with websockets.serve(handle_client, HOST, PORT):
            print(f"Server listening on ws://{HOST}:{PORT}")
            await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        print("Server task cancelled. Shutting down.")
    finally:
        print("Server shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down.")
