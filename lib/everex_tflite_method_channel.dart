import 'dart:io';
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
    required List<Uint8List> bytesList,
    required List<int> strides,
    int imageRotationDegree = 90,
    String? cameraLensDirection,
    String? deviceOrientation,
    int imageHeight = 1280,
    int imageWidth = 720,
    int rotation = 90, // Android only
  }) async {
    bool? k = await methodChannel.invokeMethod("runModel", {
      "cameraLensDirection": cameraLensDirection,
      "deviceOrientation": deviceOrientation,
      "imageRotationDegree": imageRotationDegree,
      "bytesList": bytesList,
      "strides": strides,
      "imageHeight": imageHeight,
      "imageWidth": imageWidth,
      "rotation": rotation
    });
    return k;
  }

  @override
  Future<List<double>?> outPut() async {
    if (Platform.isAndroid) {
      List<double>? k = await methodChannel.invokeMethod("outPut");
      return k;
    } else if (Platform.isIOS) {
      List<Object?> iosOutPut = await methodChannel.invokeMethod("outPut");
      List<double> k = [];
      for (var s in iosOutPut) {
        k.add(s as double);
      }
      return k;
    }

    return [];
  }

  @override
  Future<bool?> close() async {
    return await methodChannel.invokeMethod("close");
  }

  @override
  Future<List?> callBackImageData() async {
    return await methodChannel.invokeMethod("callBackImageData");
  }
}
