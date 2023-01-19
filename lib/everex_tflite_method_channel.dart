import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'everex_tflite_platform_interface.dart';

class MethodChannelEverexTflite extends EverexTflitePlatform {
  @visibleForTesting
  final methodChannel = const MethodChannel('everex_tflite');

  @override
  Future<void> loadModel() async {
    await methodChannel.invokeMethod('loadModel');
  }

  @override
  Future<bool?> runModel({
    required List<Uint8List> bytesList,
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
  Future<List<int>?> outPut() async {
    List<int>? k = await methodChannel.invokeMethod("outPut");
    return k;
  }
}
