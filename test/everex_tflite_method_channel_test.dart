import 'package:everex_tflite/everex_tflite_method_channel.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  MethodChannelEverexTflite platform = MethodChannelEverexTflite();
  const MethodChannel channel = MethodChannel('everex_tflite');

  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    channel.setMockMethodCallHandler((MethodCall methodCall) async {
      return '42';
    });
  });

  tearDown(() {
    channel.setMockMethodCallHandler(null);
  });

  test('runModel', () async {
    // expect(await platform.runModel(), '42');
  });
}
