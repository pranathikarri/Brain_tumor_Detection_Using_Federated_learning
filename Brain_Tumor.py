from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import pickle

with open('trained_global_model.pkl', 'rb') as f:
    model = pickle.load(f)


def Brain_Tumor(a):
   

    img = image.load_img(a, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 64, 64, 3)
    img_array = img_array / 255.0  # Rescale to [0, 1]

    
    predictions = model.predict(img_array)


    predicted_class = np.argmax(predictions)
    print(f'Predicted Class: {predicted_class}')

Brain_Tumor("D:\BE_Project\FL_WEBSOCKET\input_images_to_test\Tr-glTr_0002.jpg")
