import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:dio/dio.dart';
import 'package:everex_tflite/everex_tflite.dart';
import 'package:everex_tflite_example/after_layout_mix.dart';
import 'package:everex_tflite_example/functon_test_stream_controller.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:native_device_orientation/native_device_orientation.dart';

import 'main.dart';
import 'pose_painter.dart';

enum ScreenMode { liveFeed, gallery }

class CameraView extends StatefulWidget {
  const CameraView(
      {Key? key,
      required this.title,
      this.initialDirection = CameraLensDirection.back,
      this.controller})
      : super(key: key);

  final String title;
  final CameraLensDirection initialDirection;
  final CameraController? controller;

  @override
  _CameraViewState createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> with AfterLayoutMixin {
  CameraController? _controller;
  int _cameraIndex = 0;
  bool completeLoadCamera = false;
  XFile? imageFile;
  var dio = Dio();
  bool uploadData = false;

  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _controller;
    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }
    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
    } else if (state == AppLifecycleState.resumed) {
      onNewCameraSelected(cameraController.description);
    }
  }

  Future<void> onNewCameraSelected(CameraDescription cameraDescription) async {
    final CameraController? oldController = _controller;
    if (oldController != null) {
      _controller = null;
      await oldController.dispose();
    }
  }


  @override
  void initState() {
    super.initState();
    for (var i = 0; i < cameras.length; i++) {
      if (cameras[i].lensDirection == widget.initialDirection) {
        _cameraIndex = i;
      }
    }

    EverexTflite.loadModel("assets/tflite/model_ver_v2.tflite");
  }

  @override
  FutureOr<void> afterFirstLayout(BuildContext context) async {
    functionTestStream.width = MediaQuery.of(context).size.width;
    functionTestStream.height = MediaQuery.of(context).size.height;

    await _startLiveFeed();
  }

  @override
  void dispose() {
    super.dispose();
    EverexTflite.close();
    _stopLiveFeed();
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () {
        return Future(() => true); //뒤로가기 허용
      },
      child: Scaffold(
        floatingActionButton: FloatingActionButton(
          onPressed: () async {
            Navigator.of(context).pop();
          },
        ),
        body: completeLoadCamera ? _body() : Container(),
      ),
    );
  }

  Widget _body() {
    Widget body;
    body = _liveFeedBody();
    return body;
  }

  Widget _liveFeedBody() {
    return Transform.scale(
      scale: 1,
      alignment: Alignment.topCenter,
      child: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          mounted ? CameraPreview(_controller!) : Container(),
          StreamBuilder<List<double>>(
            stream: functionTestStream.poseData,
            builder:
                (BuildContext context, AsyncSnapshot<List<double>> snapshot) {
              if (snapshot.hasData) {
                final painter = PosePainter(
                  snapshot.data!,
                );
                if (snapshot.data?.isNotEmpty ?? false) {
                  return CustomPaint(painter: painter);
                } else {
                  return Container();
                }
              } else {
                return Container();
              }
            },
          ),
        ],
      ),
    );
  }

  Future _startLiveFeed() async {
    final camera = cameras[_cameraIndex];
    _controller = CameraController(camera, ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888);

    await _controller!.initialize();

    _controller?.startImageStream(_processCameraImage);

    completeLoadCamera = true;
    //_controller!.lockCaptureOrientation();

    setState(() {});
  }

  Future _stopLiveFeed() async {
    await _controller?.stopImageStream();
    await _controller?.dispose();
    _controller = null;
  }

  bool busy = false;

  Future _processCameraImage(CameraImage image) async {
    try {
      //320 * 240;
      NativeDeviceOrientation orientation =
          await NativeDeviceOrientationCommunicator()
              .orientation(useSensor: false);
      List<int> strides = Int32List(image.planes.length * 2);
      if (Platform.isAndroid) {
        int index = 0;
        List<Uint8List> data = image.planes.map((plane) {
          strides[index] = (plane.bytesPerRow);
          index++;
          strides[index] = (plane.bytesPerPixel)!;
          index++;
          return plane.bytes;
        }).toList();
      }

      // print("높이" + image.height.toString());
      // print("넓이" + image.width.toString());
      //
      // print(orientation.name);
      // print(_controller!.description.lensDirection.name);
      // print(_controller!.description.sensorOrientation);

      if (busy == false) {
        busy = true;

        bool? runComplete = await EverexTflite.runModel(
            imageRotationDegree: _controller!.description.sensorOrientation,
            cameraLensDirection: _controller!.description.lensDirection.name,
            deviceOrientation: orientation.name,
            imageHeight: image.height,
            imageWidth: image.width,
            strides: strides,
            bytesList: image.planes.map((Plane plane) => plane.bytes).toList());

        if (runComplete ?? false) {
          List<double>? result = await EverexTflite.outPut();
          // print(result);
          functionTestStream.setPoseData(result!);
        }

        busy = false;
      } else {}
    } catch (e) {
      print(e);
    }
  }
}
