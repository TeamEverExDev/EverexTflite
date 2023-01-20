package com.example.everex_tflite

class PoseEstimationUtil {
    fun getJointPositions(
        heatmap: Array<Array<Array<FloatArray>>>,
        outputHeight: Int,
        outputWidth: Int
    ): FloatArray {
        val positions = FloatArray(17 * 2)

        // 각 조인트별 Heatmap 에서 가장 높은 score를 가지는 x, y 좌표 선택
        for (i in 0..16) {
            var maxX = 0
            var maxY = 0
            var max = 0f

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

            if (max > 50.0f) {  // 일정 score 이하의 관절 위치는 사용하지 않음
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
            } else {
                // 일정 score 이하의 조인트는 좌표값으로 -1 사용 (Display 시 보이지 않도록 함)
                positions[i * 2 + 0] = -1.0f
                positions[i * 2 + 1] = -1.0f
            }
        }

        return positions
    }

    fun heatmapSmoothing(
        heatmap: Array<Array<Array<FloatArray>>>,
        prevHeatmap: Array<Array<FloatArray>>
    ): Array<Array<Array<FloatArray>>> {
        // 연속된 프레임간의 Heatmap 을 weighted averaging 함
        for (i in heatmap[0].indices) {
            for (j in heatmap[0][i].indices) {
                for (k in heatmap[0][i][j].indices) {
                    heatmap[0][i][j][k] = heatmap[0][i][j][k] * 0.5f + prevHeatmap[i][j][k] * 0.5f
                    prevHeatmap[i][j][k] = heatmap[0][i][j][k]
                }
            }
        }
        return heatmap;
    }

}