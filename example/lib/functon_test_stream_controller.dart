import 'package:rxdart/rxdart.dart';

class FunctionTestStreamController {
  final BehaviorSubject<List<double>> poseData =
      BehaviorSubject<List<double>>();

  double width = 480;
  double height = 1200;

  close() {
    poseData.close();
  }

  setPoseData(List<double> poses) {
    poseData.sink.add(poses);
  }

  clearPoseData() {
    poseData.value.clear();
  }
}

final functionTestStream = FunctionTestStreamController();
