package com.edho.animalclassification

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class ImageClassifier(context: Context) {

    private var interpreter: Interpreter

    init {
        val model = loadModelFile(context)
        interpreter = Interpreter(model)
    }

    private fun loadModelFile(context: Context): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("cnn_image_classification.tflite")
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classifyImage(image: Bitmap): String {
        // Resize the image to the expected input size of the model (e.g., 224x224)
        val resizedBitmap = Bitmap.createScaledBitmap(image, 256, 256, true)

        // Convert the image to TensorImage
        val tensorImage = TensorImage.fromBitmap(resizedBitmap)

        // Load the image into a TensorBuffer
        val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
        inputBuffer.loadBuffer(tensorImage.buffer)

        // Prepare the output buffer
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, NUM_CLASSES), DataType.FLOAT32)

        // Run the model inference
        interpreter.run(inputBuffer.buffer, outputBuffer.buffer.rewind())

        // Find the class with the highest probability
        val confidence = outputBuffer.floatArray
        val maxIndex = confidence.indices.maxByOrNull { confidence[it] } ?: -1
        return labels[maxIndex]
    }

    companion object {
        const val NUM_CLASSES = 1000 // Adjust according to your model

        val labels = arrayOf("cat", "dog", "elephant", "horse", "lion") // Add labels for your classes
    }
}
