import Flutter
import UIKit
import TensorFlowLite


public class SwiftEverexTflitePlugin: NSObject, FlutterPlugin {
    
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "everex_tflite", binaryMessenger: registrar.messenger())
        let instance = SwiftEverexTflitePlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    
    private var interpreter : Interpreter? = nil
    private var delegate : Delegate? = nil
    private var inputTensor: Tensor? = nil
    private var heatsTensor: Tensor? = nil
    private var personBBInInputImg: CGRect? = nil
    private var inputImgSize: CGSize? = nil
    
    private var positions : Array<Float>? = nil
    private var isInitialized = false
    
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        
        switch call.method {
        case "loadModel":
            let fileName = call.arguments as? String
            let modelPath = Bundle.main.path(forResource: fileName, ofType: "tflite")
            var options = Interpreter.Options()
            options.threadCount = 2
            
            let coreMLDelegate = CoreMLDelegate()
            var delegates : [Delegate] = [coreMLDelegate!]
            
            do {
                interpreter = try Interpreter(modelPath: modelPath ?? "", options: options, delegates: delegates)
                try interpreter?.allocateTensors()
                inputTensor = try interpreter?.input(at: 0)
                heatsTensor = try interpreter?.output(at: 0)
                
                
                guard (inputTensor?.dataType == .uInt8) == Model.isQuantized else {
                    fatalError("Unexpected Model: quantization is \(!Model.isQuantized)")
                }
                
                guard inputTensor?.shape.dimensions[0] == Model.input.batchSize,
                      inputTensor?.shape.dimensions[1] == Model.input.height,
                      inputTensor?.shape.dimensions[2] == Model.input.width,
                      inputTensor?.shape.dimensions[3] == Model.input.channelSize
                else {
                    fatalError("Unexpected Model: input shape")
                }
                
                guard heatsTensor?.shape.dimensions[0] == Model.output.batchSize,
                      heatsTensor?.shape.dimensions[1] == Model.output.height,
                      heatsTensor?.shape.dimensions[2] == Model.output.width,
                      heatsTensor?.shape.dimensions[3] == Model.output.keypointSize
                else {
                    fatalError("Unexpected Model: heat tensor")
                }
                
                personBBInInputImg = CGRect(x: 0, y: 0, width: 0, height: 0)
                inputImgSize = CGSize(width: 0, height: 0)
                
                
            } catch {
                print(error)
            }
            
            isInitialized = true
            
            
            result(true)
        case "runModel":
            
            let arg : [String:Any] = (call.arguments as? [String: Any])!
            var byteArray = arg["byteList"]
           
            positions = runPoseNet(on: byteArray as! CVPixelBuffer)
            
            result(true)
        case "outPut":
            result(positions)
        case "checkInitialize":
            result(isInitialized)
        case "close":
            isInitialized = false
            interpreter = nil
            delegate = nil
            result(true)
        default:
            result("default")
        }
    }
    
    /// Runs PoseNet model with given image with given source area to destination area.
    ///
    /// - Parameters:
    ///   - on: Input image to run the model.
    ///   - from: Range of input image to run the model.
    ///   - to: Size of view to render the result.
    /// - Returns: Result of the inference and the times consumed in every steps.
    func runPoseNet(on pixelbuffer: CVPixelBuffer)
    -> (Array<Float>)?
    {
    
        inputImgSize = pixelbuffer.size
        if personBBInInputImg!.width <= 0 {
            personBBInInputImg?.size.width = inputImgSize!.width
            personBBInInputImg?.size.height = inputImgSize!.height
        }
        
        guard let data = preprocess(of: pixelbuffer) else {
            fatalError("Preprocessing failed")
            return nil
        }
        inference(from: data)
    
        var arr2 :[Float] = postprocess()
        
        return arr2
    }
    
    private func inference(from data: Data) {
        // Copy the initialized `Data` to the input `Tensor`.
        do {
            try interpreter?.copy(data, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            try interpreter?.invoke()
            
            // Get the output `Tensor` to process the inference results.
            heatsTensor = try interpreter?.output(at: 0)
            
        } catch let error {
            fatalError(
                "Failed to invoke the interpreter with error: %s  "+error.localizedDescription)
            return
        }
    }
    
    private func preprocess(of pixelBuffer: CVPixelBuffer) -> Data? {
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
        
        // Resize `targetSquare` of input image to `modelSize`.
        let modelSize = CGSize(width: Model.input.width, height: Model.input.height)
        var cropBB = CGRect(x: 0, y: 0, width: pixelBuffer.size.width, height: pixelBuffer.size.height)

        if personBBInInputImg!.width > 0 {
            cropBB = personBBInInputImg!
        }
        
        #if DEBUG
        NSLog("cropBB: %.1f, %.1f", cropBB.width, cropBB.height)
        #endif
        guard let thumbnail = pixelBuffer.resizePixelBuffer(from: cropBB, to: modelSize)
        else {
            return nil
        }
        
        // Remove the alpha component from the image buffer to get the initialized `Data`.
        guard
            let inputData = thumbnail.rgbData(
                isModelQuantized: Model.isQuantized
            )
        else {
            fatalError("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        return inputData
    }
    
    func postprocess() -> Array<Float> {
        var result : Array<Float> = []
        // MARK: Formats output tensors
        // Convert `Tensor` to `FlatArray`. As PoseNet is not quantized, convert them to Float type
        // `FlatArray`.
        let heats = FlatArray<Float32>(tensor: heatsTensor!)
        // MARK: Find position of each key point
        // Finds the (row, col) locations of where the keypoints are most likely to be. The highest
        // `heats[0, row, col, keypoint]` value, the more likely `keypoint` being located in (`row`,
        // `col`).
        let valTh: Float32 = 50.0  // 관절 임계값 설정
        
        
        let keypointPositions = (0..<Model.output.keypointSize).map { keypoint -> (Float, Float) in
            var maxValue = heats[0, 0, 0, keypoint]
            var maxRow = 0
            var maxCol = 0
            for row in 0..<Model.output.height {
                for col in 0..<Model.output.width {
                    if heats[0, row, col, keypoint] > maxValue {
                        maxValue = heats[0, row, col, keypoint]
                        maxRow = row
                        maxCol = col
                    }
                }
            }
            
            if maxValue > valTh {
                var maxRowf:Float32 = 0.0
                var maxColf:Float32 = 0.0
                
                // subpixel refine
                if maxRow >= 1 && maxRow < Model.output.height-1 {
                    let diffY = heats[0, maxRow+1, maxCol, keypoint] - heats[0, maxRow-1, maxCol, keypoint]
                    if diffY > 0 {
                        maxRowf = Float(maxRow) + 0.25
                    }else {
                        maxRowf = Float(maxRow) - 0.25
                    }
                }else{
                    maxRowf = Float(maxRow)
                }
                
                if maxCol >= 1 && maxCol < Model.output.width-1 {
                    let diffX = heats[0, maxRow, maxCol+1, keypoint] - heats[0, maxRow, maxCol-1, keypoint]
                    if diffX > 0 {
                        maxColf = Float(maxCol) + 0.25
                    }else {
                        maxColf = Float(maxCol) - 0.25
                    }
                }else{
                    maxColf = Float(maxCol)
                }
                
                return (maxRowf, maxColf)
            }else{
                return (-1.0, -1.0)
            }
        }
       
        
        return result
    }
}


typealias FileInfo = (name: String, extension: String)

enum Model {
    static let isQuantized = false
    static let file: FileInfo = (
        name: "model_ver_v2", extension: "tflite"
    )
    static let input = (batchSize: 1, height: 320, width: 240, channelSize: 3)
    static let output = (batchSize: 1, height: 80, width: 60, keypointSize: 17)
}
