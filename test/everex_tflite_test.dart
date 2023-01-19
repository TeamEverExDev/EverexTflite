import 'dart:typed_data';

import 'package:everex_tflite/everex_tflite.dart';
import 'package:everex_tflite/everex_tflite_method_channel.dart';
import 'package:everex_tflite/everex_tflite_platform_interface.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockEverexTflitePlatform
    with MockPlatformInterfaceMixin
    implements EverexTflitePlatform {
  @override
  Future<String?> loadModel() {
    return Future.value('42');
  }

  @override
  Future<List<double>?> outPut() {
    return Future.value([1, 2]);
  }

  @override
  Future<bool?> runModel({
    required List<Uint8List> bytesList,
    int imageHeight = 1280,
    int imageWidth = 720,
    int rotation = 90, // Android only
  }) {
    return Future.value(true);
  }
}

void main() {
  final EverexTflitePlatform initialPlatform = EverexTflitePlatform.instance;

  test('$MethodChannelEverexTflite is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelEverexTflite>());
  });

  test('runModel', () async {
    EverexTflite everexTflitePlugin = EverexTflite();
    MockEverexTflitePlatform fakePlatform = MockEverexTflitePlatform();
    EverexTflitePlatform.instance = fakePlatform;
  });
}
