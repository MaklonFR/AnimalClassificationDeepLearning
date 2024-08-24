package com.edho.animalclassification
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var imageView: ImageView
    private lateinit var btnChooseImage: Button
    private lateinit var tvResult: TextView
    private lateinit var tflite: Interpreter

    private val imageSize = 64  // Assuming the model expects 64x64 images

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        btnChooseImage = findViewById(R.id.btnChooseImage)
        tvResult = findViewById(R.id.tvResult)

        // Load TFLite model
        val model = FileUtil.loadMappedFile(this, "cnn_image_classification.tflite")
        tflite = Interpreter(model)

        btnChooseImage.setOnClickListener {
            chooseImage()
        }

    }
    private fun chooseImage() {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, 1)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 1 && resultCode == Activity.RESULT_OK) {
            val uri = data?.data ?: return
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            imageView.setImageBitmap(bitmap)

            classifyImage(bitmap)
        }
    }

    private fun classifyImage(bitmap: Bitmap) {
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3) // 4 bytes per float32
        byteBuffer.order(ByteOrder.nativeOrder())
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        val intValues = IntArray(imageSize * imageSize)
        resizedBitmap.getPixels(intValues, 0, imageSize, 0, 0, imageSize, imageSize)
        var pixel = 0
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f) // Red
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)  // Green
                byteBuffer.putFloat((value and 0xFF) / 255.0f)          // Blue
            }
        }
        byteBuffer.rewind()

        // Prepare input buffer
        val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
        inputBuffer.loadBuffer(byteBuffer)

        // Prepare output buffer
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 5), DataType.FLOAT32) // Adjust the output shape as needed)

        // Run inference
        tflite.run(inputBuffer.buffer, outputBuffer.buffer)

        // Get the labels
        val labels = FileUtil.loadLabels(this, "labels.txt")

        // Map output to labels
        val tensorLabel = TensorLabel(labels, outputBuffer)
        val floatMap = tensorLabel.mapWithFloatValue

        // Get the top prediction
        val maxEntry = floatMap.maxByOrNull { it.value }
        tvResult.text = "Result Prediction " + maxEntry?.key + " : " + maxEntry?.value.toString()
    }

}