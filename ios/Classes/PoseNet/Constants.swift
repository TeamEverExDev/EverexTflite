//
//  Constants.swift
//  everex_tflite
//
//  Created by 김동주 on 2023/01/26.
//

enum Constants {
  // MARK: - Constants related to the image processing
  static let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  static let rgbPixelChannels = 3
  static let mean_R: Float32 = 123.68
  static let mean_G: Float32 = 116.78
  static let mean_B: Float32 = 103.94

  // MARK: - Constants related to the model interperter
  static let defaultThreadCount = 2
  static let defaultDelegate: Delegates = .Metal
}
