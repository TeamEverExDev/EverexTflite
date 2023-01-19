package com.example.everex_tflite


import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.view.Surface
import androidx.annotation.NonNull
import io.flutter.BuildConfig
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
import java.nio.ByteBuffer

/** EverexTflitePlugin */
class EverexTflitePlugin : FlutterPlugin, MethodCallHandler {

    private lateinit var channel: MethodChannel
    var isInitialized = false
    private lateinit var context: Context
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0
    private var modelInputSize: Int = 0
    var outputWidth: Int = 0
    var outputHeight: Int = 0
    private lateinit var heatmapOutput: Array<Array<Array<FloatArray>>>
    private lateinit var prevHeatmap: Array<Array<FloatArray>>

    private val positions = FloatArray(17 * 2)

    private var inputImageBuffer: TensorImage? = null
    private var imageprocessorRot0: ImageProcessor? = null
    private var imageprocessorRot90: ImageProcessor? = null
    private var imageprocessorRot180: ImageProcessor? = null
    private var imageprocessorRot270: ImageProcessor? = null

    private val poseEstimationUtil: PoseEstimationUtil = PoseEstimationUtil()


    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "everex_tflite")
        channel.setMethodCallHandler(this)
        context = flutterPluginBinding.applicationContext
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "loadModel" -> {
                //loadModel
                var fileName = call.arguments
                val loadModelMethod = LoadModelMethod()
                val assetManager: AssetManager = context.assets
                val model = loadModelMethod.loadModelFile(assetManager, fileName)

                //init interpreter
                val options = Interpreter.Options()
                gpuDelegate = GpuDelegate()
                var interpreter = Interpreter(model, options)

                var inputShape = interpreter!!.getInputTensor(0).shape()
                val inputDataType = interpreter!!.getInputTensor(0).dataType()
                inputImageWidth = inputShape[2]
                inputImageHeight = inputShape[1]
                modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * CHANNEL_SIZE

                outputHeight = inputImageHeight / 4
                outputWidth = inputImageWidth / 4
                prevHeatmap = Array(outputHeight) { Array(outputWidth) { FloatArray(17) } }
                heatmapOutput =
                    Array(1) { Array(outputHeight) { Array(outputWidth) { FloatArray(17) } } }

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
                var byteList: ByteBuffer? = call.arguments() as ByteBuffer?

                var bitmap: Bitmap = poseEstimationUtil.unNormalizeImage(
                    byteList!!,
                    inputImageWidth,
                    inputImageHeight
                )

                inputImageBuffer!!.load(bitmap)

                var rotation = call.arguments
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


                val startTime = SystemClock.uptimeMillis()
                interpreter?.run(byteBuffer, heatmapOutput)

                // heatmap smoothing
                poseEstimationUtil.heatmapSmoothing(heatmapOutput, prevHeatmap)

                poseEstimationUtil.getJointPositions(heatmapOutput, outputWidth, outputHeight)

                val endTime = SystemClock.uptimeMillis()
                val elapsedTime = endTime - startTime

                if (BuildConfig.DEBUG) {
                    val debugBitmap = poseEstimationUtil.unNormalizeImage(
                        byteBuffer,
                        inputImageWidth,
                        inputImageHeight
                    )
                    Triple(positions, elapsedTime, debugBitmap)
                } else {
                    Triple(positions, elapsedTime, null)
                }

                result.success(true)
            }
            "outPutReturn" -> {
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

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }

    private fun buildImageProcessor(rotParam: Int): ImageProcessor {
        val resizeHeight = kotlin.math.min(inputImageWidth, inputImageHeight)
        val resizeWidth = kotlin.math.max(inputImageWidth, inputImageHeight)

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





