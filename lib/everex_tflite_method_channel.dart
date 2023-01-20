import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'everex_tflite_platform_interface.dart';

class MethodChannelEverexTflite extends EverexTflitePlatform {
  @visibleForTesting
  final methodChannel = const MethodChannel('everex_tflite');

  @override
  Future<void> loadModel(String fileName) async {
    await methodChannel.invokeMethod('loadModel', fileName);
  }

  @override
  Future<bool?> runModel({
    required Uint8List bytesList,
    int imageHeight = 1280,
    int imageWidth = 720,
    int rotation = 90, // Android only
  }) async {
    bool? k = await methodChannel.invokeMethod("runModel", {
      "bytesList": bytesList,
      "imageHeight": imageHeight,
      "imageWidth": imageWidth,
      "rotation": rotation
    });
    return k;
  }

  @override
  Future<List<double>?> outPut() async {
    List<double>? k = await methodChannel.invokeMethod("outPut");
    return k;
  }

  @override
  Future<bool?> close() async {
    return await methodChannel.invokeMethod("close");
  }
}
