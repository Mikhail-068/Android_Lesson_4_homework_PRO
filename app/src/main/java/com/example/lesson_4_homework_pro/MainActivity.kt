package com.example.lesson_4_homework_pro

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.lesson_4_homework_pro.ml.ModelBus
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

private const val IMAGE_H = 128
private const val IMAGE_W = 64

class MainActivity : AppCompatActivity() {
    val PICK_IMAGE_REQUEST = 1 // константа для кода запроса изображения
    lateinit var selectBtn: Button
    lateinit var acceptBtn: Button
    lateinit var img: ImageView
    lateinit var predText: TextView
    var selectedBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectBtn = findViewById(R.id.select_1)    // кнопка выбора картинки
        acceptBtn = findViewById(R.id.predict_1)   // кнопка запука НС
        img = findViewById(R.id.image_1)           // поле, показывающее картинку
        predText = findViewById(R.id.text_1)       // текстовое поле, хранящее результат

        // настраиваем обработчик изображений для НС
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(IMAGE_H, IMAGE_W, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0.0f, 1.0f))
            .build()

        //------------------------------------------------------------------
        // обработчик кнопки выбора картинки из памяти устройства
        selectBtn.setOnClickListener {
            // создаем новое намерение для выбора изображения из галереи
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            // запускаем активность для получения результата
            startActivityForResult(intent, PICK_IMAGE_REQUEST)
        }


        //------------------------------------------------------------------
        // обработчик кнопки запуска алгоритма с НС
        acceptBtn.setOnClickListener {

            // заглушка, которая не позволяет запустить НС, если картинка не выбрана
            if (selectedBitmap == null) {
                Toast.makeText(
                    this@MainActivity,
                    "Изображение не выбрано!",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            // создаем объект для нашего изображения
            var image = TensorImage(DataType.FLOAT32)
            // добавляем изображание
            image.load(selectedBitmap)
            // предобработка изображения
            image = imageProcessor.process(image)

            // создаем экземляр модели
            val model = ModelBus.newInstance(this)

            // Создает входные данные для справки.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, IMAGE_H, IMAGE_W, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(image.buffer) // просто передать изобр нельзя!!!

            // Запускает вывод модели и получает результат
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.getFloatValue(0)

            val labels = application.resources.assets.open("Label.txt").bufferedReader().readLines()

            // Выводим результат на экран
            predText.text = labels[outputFeature0.toInt()]

            // закрываем модель
            model.close()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        // проверяем, что результат соответствует нашему коду запроса и не пустой
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            // получаем URI выбранного изображения
            val imageUri = data.data;
            // устанавливаем его в наш ImageView
            selectedBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)
            img.setImageBitmap(selectedBitmap)
        }
    }
}
