var path_model = 'https://www.smkn1kuwus.sch.id/model_machine_learning/tfjs_model/model.json';
let model;

// Load the model
async function loadModel() {
    try {
        model = await tf.loadLayersModel(path_model);
        console.log('Model loaded');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

loadModel();

window.loadImage = function(event) {
    const image = document.getElementById('uploadedImage');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.style.display = 'block'; // Make sure the image is visible
    image.onload = () => predict(image);
}

async function predict(imageElement) {
    const tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([64, 64]) // Resize image to 64x64
        .toFloat()
        .expandDims();

    const predictions = await model.predict(tensor).data();
    displayPrediction(predictions);
}

function displayPrediction(predictions) {
    const classes = ['cat', 'dog', 'elephant', 'horse', 'lion'];
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    document.getElementById('predictionResult').innerText = `Prediction: ${classes[maxIndex]}`;
}
