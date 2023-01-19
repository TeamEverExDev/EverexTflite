import 'dart:typed_data';

import 'everex_tflite_platform_interface.dart';

class EverexTflite {
  Future<void> loadModel() async {
    await EverexTflitePlatform.instance.loadModel();
  }

  Future<bool?> runModel({
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

  Future<List<int>?> outPut() {
    return EverexTflitePlatform.instance.outPut();
  }
}
