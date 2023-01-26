import Flutter
import UIKit


public class SwiftEverexTflitePlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "everex_tflite", binaryMessenger: registrar.messenger())
    let instance = SwiftEverexTflitePlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    
      switch call.method {
      case "loadModel":
          result(true)
      case "runModel":
           result(true)
      case "outPut":
            result("outPut")
      case "checkInitialize":
            result("checkInitialize")
      case "close":
            result("close")
      default:
          result("default")
      }
  }
}
