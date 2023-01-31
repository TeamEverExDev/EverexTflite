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
    private var personBBInInputImg: CGRect? = nil
    private var inputImgSize: CGSize? = nil
    
    private var positions : Array<Float>? = nil
    private var isInitialized = false
    
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        
        switch call.method {
        case "loadModel":
            let fileName = call.arguments as? String
    
            let key = _reg?.lookupKey(forAsset: fileName ?? "")
            let modelPath = Bundle.main.path(forResource: key, ofType: nil)
    
            var options = Interpreter.Options()
            options.threadCount = 2
            print("1")
            let coreMLDelegate = CoreMLDelegate()
               print("5")
            var delegates : [Delegate] = [coreMLDelegate!]
               print("4")
            do {
                interpreter = try Interpreter(modelPath: modelPath ?? "", options: options, delegates: delegates)
                  print("2")
                try interpreter?.allocateTensors()
                inputTensor = try interpreter?.input(at: 0)
                heatsTensor = try interpreter?.output(at: 0)
               print("3")
                
                guard (inputTensor?.dataType == .uInt8) == Model.isQuantized else {
                    fatalError("Unexpected Model: quantization is \(!Model.isQuantized)")
                }
                 print("a")
                
                guard inputTensor?.shape.dimensions[0] == Model.input.batchSize,
                      inputTensor?.shape.dimensions[1] == Model.input.height,
                      inputTensor?.shape.dimensions[2] == Model.input.width,
                      inputTensor?.shape.dimensions[3] == Model.input.channelSize
                else {
                    fatalError("Unexpected Model: input shape")
                }
                 print("b")
                
                guard heatsTensor?.shape.dimensions[0] == Model.output.batchSize,
                      heatsTensor?.shape.dimensions[1] == Model.output.height,
                      heatsTensor?.shape.dimensions[2] == Model.output.width,
                      heatsTensor?.shape.dimensions[3] == Model.output.keypointSize

                else {
                    fatalError("Unexpected Model: heat tensor")
                }
                personBBInInputImg = CGRect(x: 0, y: 0, width: 0, height: 0)
                inputImgSize = CGSize(width: 0, height: 0)
                print("d")
            } catch {
                print("e")
                print(error)
            }
            
            isInitialized = true
        case "runModel":
            let arg = call.arguments as? NSDictionary
            var temp : [FlutterStandardTypedData] = arg?["bytesList"] as! [FlutterStandardTypedData]
            let typedData : FlutterStandardTypedData = temp.first!
            
            let data = Data(typedData.data)
            let myUInt8bytes: [UInt8] = data.toArray(type: UInt8.self)
            
            let resizeUint8Bytes : [UInt8] = resizeBgraImage(pixels: myUInt8bytes, width: 288, height: 352, targetSize: CGSize(width: 240, height: 320))
            
            //let argb : [UInt8] = bgraToArgb(bgra: myUInt8bytes)
            let argb : [UInt32] = bgraToRgb(pixels: resizeUint8Bytes, width: 320, height: 240)
            
          
            positions = runPoseNet(on: argb)
            
            
            result(true)
        case "outPut":
            positions?.insert(1.1, at: 0)
            //print(positions)
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
    func runPoseNet(on pixelbuffer: [UInt32])
    -> (Array<Float>)?
    {
        inputImgSize = CGSize(
            width: 320, height: 240
        )
        
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
    
    private func preprocess(of pixelBuffer: [UInt32]) -> Data? {
        let data = Data(bytes: pixelBuffer, count: pixelBuffer.count * MemoryLayout<UInt32>.size)
        return data
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


func bgraToArgb(bgra: [UInt8]) -> [UInt8] {
    var argb = [UInt8](repeating: 0, count: bgra.count)
    let bytesPerPixel = 4
    
    for i in 0..<bgra.count/4 {
        let offset = i*4
                  argb[offset + 0] = bgra[offset + 2]
                  argb[offset + 1] = bgra[offset + 1]
                  argb[offset + 2] = bgra[offset + 0]
                  argb[offset + 3] = bgra[offset + 3]
    }
    
    return argb
}

func bgra8888ToRgb(pixel: UInt32) -> (red: UInt8, green: UInt8, blue: UInt8) {
    let blue = UInt8(pixel & 0xff)
    let green = UInt8((pixel >> 8) & 0xff)
    let red = UInt8((pixel >> 16) & 0xff)
    return (red, green, blue)
}

func bgraToRgb(pixels: [UInt8], width: Int, height: Int) -> [UInt32] {
    var rgbPixels = [UInt32](repeating: 0, count: width * height * 3)
    for i in 0..<width * height {
        let offset = i * 4
        rgbPixels[i * 3] = UInt32(pixels[offset + 2])
        rgbPixels[i * 3 + 1] = UInt32(pixels[offset + 1])
        rgbPixels[i * 3 + 2] = UInt32(pixels[offset])
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
