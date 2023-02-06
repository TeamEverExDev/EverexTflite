import Flutter
import UIKit
import TensorFlowLite
import Accelerate
import CoreImage



public class SwiftEverexTflitePlugin: NSObject, FlutterPlugin {
    
    private var _reg : FlutterPluginRegistrar? = nil
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "everex_tflite", binaryMessenger: registrar.messenger())
        let instance = SwiftEverexTflitePlugin()
        instance._reg = registrar
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    
    private var interpreter : Interpreter? = nil
    private var delegate : Delegate? = nil
    private var inputTensor: Tensor? = nil
    private var heatsTensor: Tensor? = nil
    
    private var positions : Array<Float>? = nil
    private var imageDatak : [Float]? = nil
    private var isInitialized = false
    private var createImage1 = false
    private var createImage2 = false
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        
        switch call.method {
        case "loadModel":
            let fileName = call.arguments as? String
            
            let key = _reg?.lookupKey(forAsset: fileName ?? "")
            let modelPath = Bundle.main.path(forResource: key, ofType: nil)
            
            var options = Interpreter.Options()
            options.threadCount = 2
            
            //GPU 사용안됨
            //var coreMLOption = CoreMLDelegate.Options()
            //coreMLOption.enabledDevices = .all
            
            //let coreMLDelegate = CoreMLDelegate(options: coreMLOption)
            //let delegates : Delegate = coreMLDelegate ?? MetalDelegate()
            let delegates : Delegate = MetalDelegate()
            
            do {
                interpreter = try Interpreter(modelPath: modelPath ?? "", options: options, delegates: [delegates])
                
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
            } catch {
                print(error)
            }
            
            isInitialized = true
        case "runModel":
            let arg = call.arguments as? NSDictionary
            let temp : [FlutterStandardTypedData] = arg?["bytesList"] as! [FlutterStandardTypedData]
            let typedData : FlutterStandardTypedData = temp.first!
            
            let data = Data(typedData.data)
            let myUInt8bytes: [UInt8] = data.toArray(type: UInt8.self)
            
            let resizeUint8Bytes : [UInt8] = resizeBgraImage(pixels: myUInt8bytes, width: 288, height: 352, targetSize: CGSize(width: 240, height: 320))
            let rgb : [UInt8] = bgraToRgb(pixels: resizeUint8Bytes, width: 240, height: 320)
            
            runPoseNet(on: rgb)
            result(true)
        case "outPut":
            result(positions)
        case "callBackImageData":
            
            result(imageDatak)
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
    func runPoseNet(on pixelbuffer: [UInt8]) {
        
        guard let data = preprocess(of: pixelbuffer) else {
            fatalError("Preprocessing failed")
        }
        
        
        inference(from: data)
        
        postprocess()
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
        }
    }
    
    private func preprocess(of pixelBuffer: [UInt8]) -> Data? {
        
        let destinationBytesPerRow = Constants.rgbPixelChannels * 240
        
        
        
        var floats : [Float] = []
        
        floats.reserveCapacity(240 * 320 * Constants.rgbPixelChannels)
        for y in 0 ..< 320 {
            for x in 0 ..< 240 {
                
                floats.append(Float(pixelBuffer[y * destinationBytesPerRow + x * 3]) - Constants.mean_R)
                floats.append(Float(pixelBuffer[y * destinationBytesPerRow + x * 3 + 1]) - Constants.mean_G)
                floats.append(Float(pixelBuffer[y * destinationBytesPerRow + x * 3 + 2]) - Constants.mean_B)
            }
        }
        
        return Data(copyingBufferOf: floats)
    }
    
    func postprocess() {
        // MARK: Formats output tensors
        // Convert `Tensor` to `FlatArray`. As PoseNet is not quantized, convert them to Float type
        // `FlatArray`.
        let heats: FlatArray<Float32> = FlatArray<Float32>(tensor: heatsTensor!)
        
        positions = Array<Float>()
        
        for i in 0...16 {
            var maxX = 0
            var maxY = 0
            var max : Float32 = 50.0
            
            for x in 0..<Model.output.width {
                for y in 0..<Model.output.height {
                    let value = heats[0, y, x, i]
                    
                    if(value > max) {
                        max = value
                        maxX = x
                        maxY = y
                    }
                }
            }
            
            var maxXf:Float = Float(maxX)
            var maxYf:Float = Float(maxY)
            
            if (maxY >= 1 && maxY < Model.output.height-1) && ( maxX >= 1 && maxX < Model.output.width-1) {
                let diffX = heats[0,  maxY+1, maxX,i] - heats[0,  maxY-1, maxX,i]
                let diffY = heats[0,  maxY, maxX+1, i] - heats[0,  maxY, maxX-1,i]
                
                if diffX > 0 {
                    maxYf = Float(maxY) + 0.25
                }else {
                    maxYf = Float(maxY) - 0.25
                }
                
                
                if diffY > 0 {
                    maxXf = Float(maxX) + 0.25
                }else {
                    maxXf = Float(maxX) - 0.25
                }
            }
            positions?.insert(maxXf, at: i * 2 + 0)
            positions?.insert(maxYf, at: i * 2 + 1)
        }
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

func bgraToRgb(pixels: [UInt8], width: Int, height: Int) -> [UInt8] {
    var rgbPixels = [UInt8](repeating: 0, count: width * height * 3)
    for i in 0..<width * height {
        let offset = i * 4
        rgbPixels[i * 3] = UInt8(pixels[offset + 2])
        rgbPixels[i * 3 + 1] = UInt8(pixels[offset + 1])
        rgbPixels[i * 3 + 2] = UInt8(pixels[offset])
    }
    return rgbPixels
}

func resizeBgraImage(pixels: [UInt8], width: Int, height: Int, targetSize: CGSize) -> [UInt8] {
    let newWidth = Int(targetSize.width)
    let newHeight = Int(targetSize.height)
    var resizedPixels = [UInt8](repeating: 0, count: newWidth * newHeight * 4)
    let xRatio = CGFloat(width) / targetSize.width
    let yRatio = CGFloat(height) / targetSize.height
    for y in 0..<newHeight {
        for x in 0..<newWidth {
            let oldX = Int(CGFloat(x) * xRatio)
            let oldY = Int(CGFloat(y) * yRatio)
            let oldIndex = (oldY * width + oldX) * 4
            let newIndex = (y * newWidth + x) * 4
            resizedPixels[newIndex] = pixels[oldIndex]
            resizedPixels[newIndex + 1] = pixels[oldIndex + 1]
            resizedPixels[newIndex + 2] = pixels[oldIndex + 2]
            resizedPixels[newIndex + 3] = pixels[oldIndex + 3]
        }
    }
    return resizedPixels
}

func normalizeRGBImage(image: [Float]) -> [Float] {
    let maxValue = image.max()!
    let minValue = image.min()!
    
    let normalizedImage = image.map { (value) -> Float in
        return (value - minValue) / (maxValue - minValue)
    }
    
    return normalizedImage
}
