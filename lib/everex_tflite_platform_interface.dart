import 'dart:typed_data';

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'everex_tflite_method_channel.dart';

abstract class EverexTflitePlatform extends PlatformInterface {
  /// Constructs a EverexTflitePlatform.
  EverexTflitePlatform() : super(token: _token);

  static final Object _token = Object();

  static EverexTflitePlatform _instance = MethodChannelEverexTflite();

  /// The default instance of [EverexTflitePlatform] to use.
  ///
  /// Defaults to [MethodChannelEverexTflite].
  static EverexTflitePlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [EverexTflitePlatform] when
  /// they register themselves.
  static set instance(EverexTflitePlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<void> loadModel() {
    throw UnimplementedError('loadModel() has not been implemented.');
  }

  Future<bool?> runModel({
    required List<Uint8List> bytesList,
    int imageHeight = 0,
    int imageWidth = 0,
    int rotation = 0, // Android only
  }) {
    throw UnimplementedError('runModel() has not been implemented.');
  }

  Future<List<int>?> outPut() {
    throw UnimplementedError('outPut() has not been implemented.');
  }
}
