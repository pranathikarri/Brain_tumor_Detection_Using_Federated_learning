# Brain Tumor Detection Using Federated Learning
This project explores the use of federated learning to detect brain tumors from MRI scans while preserving data privacy.
## Project Structure
* `clients/`: Contains the code and logic for the individual client participants (e.g., simulated hospitals) in the federated learning setup.
* `input_images_to_test/`: This directory is likely for storing sample images that can be used to test the trained model.
* `server/`: This directory holds the code for the federated learning server, responsible for aggregating model updates from clients.
* `Brain_Tumor.py`: This Python file probably contains the core deep learning model definition and the main training or inference logic for the brain tumor detection.

## Project Visual
![Screenshot of project structure](Screenshot%202024-09-22%20155106.png)

## How to Use

To run this project:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Pranav-95/Federated_learning_Brain_tumor_Detection.git](https://github.com/Pranav-95/Federated_learning_Brain_tumor_Detection.git)
    cd Federated_learning_Brain_tumor_Detection
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv fl_env
    source fl_env/bin/activate  # Use `.\fl_env\Scripts\activate` on Windows
    ```
3.  **Install dependencies:**
    *(You will need to check your project for a `requirements.txt` file or list libraries here)*
    ```bash
    pip install tensorflow numpy pandas scikit-learn flwr # Example libraries
    ```
4.  **Download the dataset:**
    * Obtain the "Brain Tumor Classification (MRI)" dataset from Kaggle.
    * Place the extracted dataset files into a `data/` folder within this project directory. (You might need to create this folder if it doesn't exist: `mkdir data`).
5.  **Run the Federated Learning simulation:**
    * Open two or more separate terminal windows.
    * In each terminal, activate your virtual environment (`source fl_env/bin/activate`).
    * In one terminal, start the **server**:
        ```bash
        python server.py
        ```
    * In the other terminals, start the **clients** (replace `0` with `1`, `2`, `3`, `4`, `5` for each client):
        ```bash
        python clients/client.py --cid 0 # Assuming 'clients' folder and 'client.py' and '--cid' argument
        ```
        *(**Note:** The exact command for running clients might vary, check your `client.py` file or any other instructions in your project.)*

## Results

[Describe your model's accuracy, precision, recall, or other performance metrics here. You could add more result screenshots if you have them.]
