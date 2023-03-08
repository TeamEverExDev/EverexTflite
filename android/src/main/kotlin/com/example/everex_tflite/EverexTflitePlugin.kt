package com.example.everex_tflite

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Environment
import android.os.Environment.DIRECTORY_DCIM
import android.util.Log
import android.view.Surface
import androidx.annotation.NonNull
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
//import kotlinx.coroutines.*
/** EverexTflitePlugin */

data class MyResult(val value1: Int, val value2: Int, val value3: Int, val value4: Int)
class EverexTflitePlugin : FlutterPlugin, MethodCallHandler {

    private lateinit var channel: MethodChannel
    var isInitialized = false
    private lateinit var context: Context
    private lateinit var flutterAsset: FlutterPlugin.FlutterAssets
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0
    private var modelInputSize: Int = 0
    var outputWidth: Int = 0
    var outputHeight: Int = 0
    private lateinit var heatmapOutput: Array<Array<Array<FloatArray>>>
    private lateinit var prevHeatmap: Array<Array<FloatArray>>
    var numJoints: Int = 17
    private var positions = FloatArray(numJoints * 2){0f}
    private var positions2 = FloatArray(numJoints * 2){0f}
    private var inputImageBuffer: TensorImage? = null
    private var imageprocessorRot0: ImageProcessor? = null
    private var imageprocessorRot90: ImageProcessor? = null
    private var imageprocessorRot180: ImageProcessor? = null
    private var imageprocessorRot270: ImageProcessor? = null

    private val poseEstimationUtil: PoseEstimationUtil = PoseEstimationUtil()

    var createImage1: Boolean = true
    var createImage2: Boolean = true
    var createImage3: Boolean = true
    var x: Int = 0
    var y: Int = 0
    var width: Int = 240
    var height: Int = 320
    var x_0: Int = 0
    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "everex_tflite")
        channel.setMethodCallHandler(this)
        context = flutterPluginBinding.applicationContext
        flutterAsset = flutterPluginBinding.flutterAssets
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "loadModel" -> {
                //loadModel
                var fileName = call.arguments as String
                val loadModelMethod = LoadModelMethod()
                val assetManager: AssetManager = context.assets
                Log.d("asset", flutterAsset.getAssetFilePathBySubpath(fileName))
                val model = loadModelMethod.loadModelFile(
                    assetManager,
                    flutterAsset.getAssetFilePathBySubpath(fileName)
                )
                //init interpreter
                val options = Interpreter.Options()
//                gpuDelegate = GpuDelegate()
//                options.addDelegate(gpuDelegate)
                var interpreter = Interpreter(model, options)
                var inputShape = interpreter!!.getInputTensor(0).shape()
                var outputShape = interpreter!!.getOutputTensor(0).shape()
                val inputDataType = interpreter!!.getInputTensor(0).dataType()
                inputImageHeight = inputShape[1]
                inputImageWidth = inputShape[2]


                modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * CHANNEL_SIZE

                outputHeight = outputShape[1]
                outputWidth = outputShape[2]
                numJoints = outputShape[3]
                prevHeatmap = Array(outputHeight) { Array(outputWidth) { FloatArray(numJoints) } }
                heatmapOutput =
                    Array(1) { Array(outputHeight) { Array(outputWidth) { FloatArray(numJoints) } } }

                this.interpreter = interpreter

                imageprocessorRot0 = buildImageProcessor(0)
                imageprocessorRot90 = buildImageProcessor(-1)
                imageprocessorRot180 = buildImageProcessor(-2)
                imageprocessorRot270 = buildImageProcessor(-3)

                inputImageBuffer = TensorImage(inputDataType)

                isInitialized = true
            }
            "runModel" -> {
                check(isInitialized) { "TF Lite Interpreter is not initialized yet." }
                var arg: HashMap<*, *> = call.arguments as HashMap<*, *>
                var byteArray: List<ByteArray> = arg.get("bytesList") as List<ByteArray>
                var strides: IntArray = arg.get("strides") as IntArray
                var imageHeight: Int = arg.get("imageHeight") as Int
                var imageWidth: Int = arg.get("imageWidth") as Int
                var rotations: Int = arg.get("rotation") as Int
                var imageRotationDegree: Int = arg.get("imageRotationDegree") as Int
                var cameraLensDirection: String = arg.get("cameraLensDirection") as String
                var deviceOrientation: String = arg.get("deviceOrientation") as String


                var data = YuvConverter.NV21toJPEG(
                    YuvConverter.YUVtoNV21(
                        byteArray,
                        strides,
                        imageWidth,
                        imageHeight
                    ), 320, 240, 100
                )


                var decodeBitmap: Bitmap = BitmapFactory.decodeByteArray(data, 0, data.size)

                var rotation = 1
                if (cameraLensDirection == "front" && deviceOrientation == "portraitUp") {
                    decodeBitmap = matrixBitmap(decodeBitmap, -1f, 1f, 90f)
                } else if (cameraLensDirection == "front" && deviceOrientation == "landscapeLeft") {
                    decodeBitmap = matrixBitmap(decodeBitmap, -1f, 1f, 0f)
                } else if (cameraLensDirection == "front" && deviceOrientation == "landscapeRight") {
                    decodeBitmap = matrixBitmap(decodeBitmap, -1f, 1f, 180f)
                } else if (cameraLensDirection == "back" && deviceOrientation == "portraitUp") {
                    decodeBitmap = matrixBitmap(decodeBitmap, 1f, 1f, 90f)
                } else if (cameraLensDirection == "back" && deviceOrientation == "landscapeLeft") {
                    decodeBitmap = matrixBitmap(decodeBitmap, 1f, 1f, 0f)
                } else if (cameraLensDirection == "back" && deviceOrientation == "landscapeRight") {
                    decodeBitmap = matrixBitmap(decodeBitmap, 1f, 1f, 180f)
                }

                if (deviceOrientation != "portraitUp") {
                    decodeBitmap = Bitmap.createBitmap(decodeBitmap, x, y, width, height)
                }
//                Log.e("width",width.toString())
//                Log.e("height",height.toString())
                decodeBitmap = Bitmap.createBitmap(decodeBitmap, x, y, width, height)

                inputImageBuffer!!.load(decodeBitmap)

                if (!createImage1) {
                    createImage1 = true;
                    bitmapToFile(
                        decodeBitmap,
                        "b_${cameraLensDirection}_${deviceOrientation}",
                        context
                    )
                }

                rotation = 0
                inputImageBuffer = when (rotation) {
                    Surface.ROTATION_0 -> imageprocessorRot0!!.process(inputImageBuffer)
                    Surface.ROTATION_90 -> imageprocessorRot90!!.process(inputImageBuffer)
                    Surface.ROTATION_180 -> imageprocessorRot180!!.process(inputImageBuffer)
                    Surface.ROTATION_270 -> imageprocessorRot270!!.process(inputImageBuffer)
                    else -> {
                        error("incorrect rotation value")
                    }
                }

                val byteBuffer = inputImageBuffer!!.buffer

                if (!createImage2) {
                    createImage2 = true;
                    bitmapToFile(
                        inputImageBuffer!!.bitmap,
                        "a_${cameraLensDirection}_${deviceOrientation}",
                        context
                    )
                }

                interpreter?.run(byteBuffer, heatmapOutput)

                positions2 =
                    poseEstimationUtil.getJointPositions(
                        heatmapOutput,
                        outputHeight,
                        outputWidth,
                        numJoints
                    )

                if (deviceOrientation != "portraitUp") {
                    width = 240
                    xvaluescale(positions)
                    x = findCenterValues(positions)
                }
                valuescale(positions2)
//                width = 240
//                height = 320
//                xvaluescale(positions)

//                x = findCenterValues(positions, x)
                val(value1,value2,value3,value4) = findCenterValues2(positions2)
                x = (value1 +x)/2
                y = (value2 +y)/2
                width = (value3+width)/2
                height = (value4+height)/2
//                x = value1
//                Log.e("x",x.toString())
//                y = value2
//                Log.e("y",y.toString())
//                width = value3
//                Log.e("width",width.toString())
//                height = value4
//                Log.e("width",height.toString())
//                Log.e("width",width.toString())
//                Log.e("height",height.toString())

//                Log.e("TAG", "positions = ${positions.contentToString()}")
//                Log.e("TAG", "positions2 = ${positions2.contentToString()}")
                result.success(true)
                for (i in positions2.indices) {
                    positions[i] = (positions[i] + positions2[i]) / 2
                }
                Log.e("height",positions[0].toString())
            }
            "outPut" -> {
                result.success(positions)
            }
            "checkInitialize" -> {
                result.success(isInitialized)
            }
            "close" -> {
                isInitialized = false
                interpreter!!.close()
                Timber.d("Closing TFLite interpreter...")
                if (gpuDelegate != null) {
                    gpuDelegate!!.close()
                    gpuDelegate = null
                }
                Timber.d("Closed TFLite interpreter.")
                result.success(true)
            }
            else -> {
                result.notImplemented()
            }
        }
    }

    fun findCenterValues(a: FloatArray): Int {
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var sum = 0f
        var count = 0

        for (i in a.indices step 2) {
            val x = a[i]
            if (x > 15.0f) {
                Log.e("sss",x.toString())
                minX = minOf(minX, x)
                maxX = maxOf(maxX, x)
                sum += x
                count += 1
            }
        }
        if (sum / count < 20) {
            return 0
        } else if (sum / count > 30) {
            return 80
        }
        return 40
//        return (sum/count*2).toInt()
    }
    fun findCenterValues2(a: FloatArray): MyResult {
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE
        var maxY = Float.MIN_VALUE
        var sumX = 0f
        var countX = 0
        var sumY = 0f
        var countY = 0
        var value1 = 0f
        var value2 = 0f
        var value3 = 240f
        var value4 = 320f
        for (i in a.indices step 2) {
            val x = a[i]
//            Log.e("sss",x.toString())
            if (x >0){
                minX = minOf(minX, x)
                maxX = maxOf(maxX, x)
                sumX += x
                countX += 1
            }
        }
        for (i in a.indices step 2) {
            val y = a[i+1]
//            Log.e("sss",y.toString())
            if (y > 0){
                minY = minOf(minX, y)
                maxY = maxOf(maxX, y)
                sumY += y
                countY += 1
            }
        }
        if (countX>0 && countY>0){

        if (minX-(maxX-minX)/10>0f){
            value1 = (minX-(maxX-minX)/10)*4
        }
        else {
            value1 = 0f
        }
        if (minY-(maxY-minY)/10>0f){
            value2 = (minY-(maxY-minY)/10)*4
        }
        else {
            value2 = 0f
        }
        if ((maxX-minX)*1.2+value1>60f||(maxX-minX)*1.2<1f){
            value3 = 240f - value1
        }
        else{
            value3 = (maxX-minX)*4.8f

        }
        if ((maxY-minY)*1.2+value2>80f || (maxY-minY)*1.2<1f){
            value4 = 320f - value2
        }
        else{
            value4 = (maxY-minY)*4.8f
        }
        }
        var value11 = value1.toInt()
        var value22 = value2.toInt()
        var value33 = value3.toInt()
        var value44 = value4.toInt()

//        Log.e("width",value3.toString())
//        Log.e("height",value4.toString())
        return MyResult(value11,value22,value33,value44)

//        return (sum/count*2).toInt()
    }

    fun xvaluescale(a: FloatArray): FloatArray {
        for (i in a.indices step 2) {
            a[i] = a[i] * width / 320 + 30 * (x) / 160
        }
        return a
    }

    fun valuescale(a: FloatArray): FloatArray {
        for (i in a.indices step 2) {
            a[i] = a[i] * width / 240 + x / 4
        }
        for (i in a.indices step 2) {
            a[i+1] = a[i+1] * height / 320 + y / 4
        }
        return a
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }

    private fun buildImageProcessor(rotParam: Int): ImageProcessor {
        val resizeWidth = kotlin.math.min(inputImageWidth, inputImageHeight)
        val resizeHeight = kotlin.math.max(inputImageWidth, inputImageHeight)

        return ImageProcessor.Builder()
            .add(ResizeOp(resizeHeight, resizeWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(Rot90Op(rotParam))
            .add(
                NormalizeOp(
                    floatArrayOf(IMAGE_MEAN_R, IMAGE_MEAN_G, IMAGE_MEAN_B), floatArrayOf(
                        1f,
                        1f,
                        1f
                    )
                )
            )
            .build()
    }
}

fun matrixBitmap(bitmap: Bitmap, sx: Float, sy: Float, degree: Float): Bitmap {
    val matrix = Matrix()
    matrix.preScale(sx, sy)
    val rotateMatrix = Matrix()
    rotateMatrix.postRotate(degree)
    matrix.postConcat(rotateMatrix)

    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

private fun bitmapToFile(bitmap: Bitmap, fileName: String, context: Context): File {
    var out: OutputStream? = null

    val file =
        File("${Environment.getExternalStorageDirectory()}/" + DIRECTORY_DCIM + "/mora/$fileName.jpg")

    Log.d("filepath", file.absolutePath);
    try {
        Log.d("t", "이미지 저장 시도")
        file.parentFile.mkdirs()
        if (file.isFile) {
            file.delete()
        }
        file.createNewFile()
        out = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out)
        Log.d("t", "이미지 저장 완료")
    } finally {
        out?.close()
    }
    return file
}




