import 'package:everex_tflite_example/functon_test_stream_controller.dart';
import 'package:flutter/material.dart';

class PosePainter extends CustomPainter {
  PosePainter(this.poses);

  final List<double> poses;
  // final Size absoluteImageSize;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10.0
      ..color = Color.fromRGBO(7, 190, 184, 1);

    final backgroundPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 15.0
      ..color = Colors.green;

    final leftPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..color = Color.fromRGBO(7, 190, 184, 1);

    final anglePainter = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..color = Colors.green;

    final anglePainterBackground = Paint()
      ..style = PaintingStyle.fill
      ..strokeWidth = 7
      ..color = Color.fromRGBO(7, 190, 184, 1).withOpacity(0.5);
    //..color = Colors.black.withOpacity(0.5);

    final rightPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5
      ..color = Color.fromRGBO(7, 190, 184, 1);

    double x = 0;
    double y = 0;
    print("here");
    print(poses.length);

    for (int i = 0; i < 34; i += 2) {
      canvas.drawCircle(
          Offset(
            translateX(
              poses[i],
              functionTestStream.width,
            ),
            translateY(
              poses[i + 1],
              functionTestStream.height,
            ),
          ),
          1,
          paint);
    }

    // for (int i = 0; i < poses.length; i++) {
    //   if (poses[i] % 2 == 0) {
    //     x = poses[i];
    //   } else {
    //     y = poses[i];
    //     print("x : $x width ${functionTestStream.width}");
    //     print("y : $y height ${functionTestStream.height}");
    //     print("Draw!!");
    //     canvas.drawCircle(
    //         Offset(
    //           translateX(
    //             x,
    //             functionTestStream.width,
    //           ),
    //           translateY(
    //             y,
    //             functionTestStream.height,
    //           ),
    //         ),
    //         1,
    //         paint);
    //   }
    // }

    // for (final pose in poses) {
    //   void paintLine(
    //       PoseLandmarkType type1, PoseLandmarkType type2, Paint paintType) {
    //     final PoseLandmark joint1 = pose.landmarks[type1]!;
    //     final PoseLandmark joint2 = pose.landmarks[type2]!;
    //     canvas.drawLine(
    //         Offset(translateX(joint1.x, rotation, size, absoluteImageSize),
    //             translateY(joint1.y, rotation, size, absoluteImageSize)),
    //         Offset(translateX(joint2.x, rotation, size, absoluteImageSize),
    //             translateY(joint2.y, rotation, size, absoluteImageSize)),
    //         paintType);
    //   }
    //
    //   //Draw arms
    //   paintLine(
    //       PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow, leftPaint);
    //   paintLine(
    //       PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist, leftPaint);
    //   paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow,
    //       rightPaint);
    //   paintLine(
    //       PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist, rightPaint);
    //
    //   //Draw Body
    //   paintLine(
    //       PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, leftPaint);
    //   paintLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip,
    //       rightPaint);
    //
    //   //Draw shoulder and hip
    //   paintLine(PoseLandmarkType.leftHip, PoseLandmarkType.rightHip, leftPaint);
    //   paintLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder,
    //       rightPaint);
    //
    //   //Draw legs
    //   paintLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee, leftPaint);
    //   paintLine(
    //       PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle, leftPaint);
    //   paintLine(
    //       PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee, rightPaint);
    //   paintLine(
    //       PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle, rightPaint);
    //
    //   //Draw Circle
    //   pose.landmarks.forEach((_, landmark) {
    //     canvas.drawCircle(
    //         Offset(
    //           translateX(landmark.x, rotation, size, absoluteImageSize),
    //           translateY(landmark.y, rotation, size, absoluteImageSize),
    //         ),
    //         1,
    //         paint);
    //
    //     canvas.drawCircle(
    //       Offset(
    //         translateX(landmark.x, rotation, size, absoluteImageSize),
    //         translateY(landmark.y, rotation, size, absoluteImageSize),
    //       ),
    //       3,
    //       backgroundPaint,
    //     );
    //   });
    // }
  }

  @override
  bool shouldRepaint(covariant PosePainter oldDelegate) {
    return oldDelegate.poses != poses;
  }
}

double translateX(double x, double width) {
  return x * width / 60;
}

double translateY(double y, double height) {
  return y * height / 80;
}
