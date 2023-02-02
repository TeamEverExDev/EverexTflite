import 'package:everex_tflite_example/functon_test_stream_controller.dart';
import 'package:flutter/material.dart';

class PosePainter extends CustomPainter {
  PosePainter(this.poses);
  final List<double> poses;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10.0
      ..color = Color.fromRGBO(7, 190, 184, 1);

    final facePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 20.0
      ..color = Colors.red;
    List bodyPart = [];

    for (int i = 0; i < 34; i += 2) {
      //draw dot
      if (i == 0) {
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
            facePaint);
      }
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
    //draw line
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[0],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[1],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[2],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[3],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[2],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[3],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[6],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[7],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[0],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[1],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[4],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[5],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[4],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[5],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[8],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[9],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[10],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[11],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[14],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[15],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[14],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[15],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[18],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[19],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[12],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[13],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //         translateX(
    //           poses[16],
    //           functionTestStream.width,
    //         ),
    //         poses[17]),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[16],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[17],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[20],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[21],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[22],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[23],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[26],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[27],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    //
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[26],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[27],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[30],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[31],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    //
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[24],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[25],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[28],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[29],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    //
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[28],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[29],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[32],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[33],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);
    //
    // canvas.drawLine(
    //     Offset(
    //       translateX(
    //         poses[10],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[11],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     Offset(
    //       translateX(
    //         poses[28],
    //         functionTestStream.width,
    //       ),
    //       translateY(
    //         poses[29],
    //         functionTestStream.height,
    //       ),
    //     ),
    //     paint);

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
