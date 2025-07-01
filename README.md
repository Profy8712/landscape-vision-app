# landscape-vision-app

A simple web app for classifying landscape images into four categories: desert, green area, cloudy, and water.  
Upload an image and get an instant prediction using a pre-trained neural network.

## Usage
- Clone the repo
- Install requirements: `pip install flask tensorflow`
- Place your model file as `landscape_classifier.h5` in the root directory
- Run the app: `python app.py`
- Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser
