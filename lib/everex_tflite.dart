import 'dart:typed_data';

import 'everex_tflite_platform_interface.dart';

class EverexTflite {
  static Future<void> loadModel(String fileName) async {
    await EverexTflitePlatform.instance.loadModel(fileName);
  }

  static Future<bool?> runModel({
    required List<Uint8List> bytesList,
    required List<int> strides,
    String? deviceOrientation,
    String? cameraLensDirection,
    int imageRotationDegree = 90,
    int imageHeight = 1280,
    int imageWidth = 720,
    int rotation = 90, // Android only
  }) {
    return EverexTflitePlatform.instance.runModel(
        strides: strides,
        bytesList: bytesList,
        deviceOrientation: deviceOrientation,
        cameraLensDirection: cameraLensDirection,
        imageRotationDegree: imageRotationDegree,
        imageHeight: imageHeight,
        imageWidth: imageWidth,
        rotation: rotation);
  }

  static Future<List<double>?> outPut() {
    return EverexTflitePlatform.instance.outPut();
  }

  static Future<bool?> close() {
    return EverexTflitePlatform.instance.close();
  }

  static Future<List?> callBackImageData() {
    return EverexTflitePlatform.instance.callBackImageData();
  }
}
