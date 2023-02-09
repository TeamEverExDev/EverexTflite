package com.example.everex_tflite

class PoseEstimationUtil {
    fun getJointPositions(
        heatmap: Array<Array<Array<FloatArray>>>,
        outputHeight: Int,
        outputWidth: Int,
        numJoints: Int
    ): FloatArray {
        
        val positions = FloatArray(numJoints * 2)
        // 각 조인트별 Heatmap 에서 가장 높은 score를 가지는 x, y 좌표 선택
        for (i in 0 until numJoints) {
            var maxX = 0
            var maxY = 0
            var max = 50f
            // find keypoint coordinate through maximum values
            for (x in 0 until outputWidth) {
                for (y in 0 until outputHeight) {
                    val value = heatmap[0][y][x][i]
                    if (value > max) {
                        max = value
                        maxX = x
                        maxY = y
                    }
                }
            }
            var maxXf = maxX.toFloat()
            var maxYf = maxY.toFloat()

            // subpixel refine
            if ((maxX in 1 until outputWidth - 1) and (maxY in 1 until outputHeight - 1)) {
                val diffY = heatmap[0][maxY + 1][maxX][i] - heatmap[0][maxY - 1][maxX][i]
                val diffX = heatmap[0][maxY][maxX + 1][i] - heatmap[0][maxY][maxX - 1][i]

                if (diffY > 0) {
                    maxYf += 0.25f
                } else {
                    maxYf -= 0.25f
                }

                if (diffX > 0) {
                    maxXf += 0.25f
                } else {
                    maxXf -= 0.25f
                }
            }
            positions[i * 2 + 0] = maxXf
            positions[i * 2 + 1] = maxYf
        }
        return positions
    }
}