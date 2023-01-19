import 'dart:typed_data';

import 'everex_tflite_platform_interface.dart';

class EverexTflite {
  static Future<void> loadModel() async {
    await EverexTflitePlatform.instance.loadModel();
  }

  static Future<bool?> runModel({
    required List<Uint8List> bytesList,
    int imageHeight = 1280,
    int imageWidth = 720,
    int rotation = 90, // Android only
  }) {
    return EverexTflitePlatform.instance.runModel(
        bytesList: bytesList,
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
}
