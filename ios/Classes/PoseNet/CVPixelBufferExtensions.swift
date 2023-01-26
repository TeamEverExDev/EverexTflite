//
//  CVPixelBufferExtensions.swift
//  everex_tflite
//
//  Created by 김동주 on 2023/01/26.
//

import Accelerate
import Foundation

extension CVPixelBuffer {
  var size: CGSize {
    return CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
  }

  /// Returns a new `CVPixelBuffer` created by taking the self area and resizing it to the
  /// specified target size. Aspect ratios of source image and destination image are expected to be
  /// same.
  ///
  /// - Parameters:
  ///   - from: Source area of image to be cropped and resized.
  ///   - to: Size to scale the image to(i.e. image size used while training the model).
  /// - Returns: The cropped and resized image of itself.
  func resize(from source: CGRect, to size: CGSize) -> CVPixelBuffer? {
    let rect = CGRect(origin: CGPoint(x: 0, y: 0), size: self.size)
    guard rect.contains(source) else {
      os_log("Resizing Error: source area is out of index", type: .error)
      return nil
    }
    guard rect.size.width / rect.size.height - source.size.width / source.size.height < 1e-5
    else {
      os_log(
        "Resizing Error: source image ratio and destination image ratio is different",
        type: .error)
      return nil
    }

    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
    let imageChannels = 4

    CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
    defer { CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0)) }

    // Finds the address of the upper leftmost pixel of the source area.
    guard
      let inputBaseAddress = CVPixelBufferGetBaseAddress(self)?.advanced(
        by: Int(source.minY) * inputImageRowBytes + Int(source.minX) * imageChannels)
    else {
      return nil
    }

    // Crops given area as vImage Buffer.
    var croppedImage = vImage_Buffer(
      data: inputBaseAddress, height: UInt(source.height), width: UInt(source.width),
      rowBytes: inputImageRowBytes)

    let resultRowBytes = Int(size.width) * imageChannels
    guard let resultAddress = malloc(Int(size.height) * resultRowBytes) else {
      return nil
    }

    // Allocates a vacant vImage buffer for resized image.
    var resizedImage = vImage_Buffer(
      data: resultAddress,
      height: UInt(size.height), width: UInt(size.width),
      rowBytes: resultRowBytes
    )

    // Performs the scale operation on cropped image and stores it in result image buffer.
    guard vImageScale_ARGB8888(&croppedImage, &resizedImage, nil, vImage_Flags(0)) == kvImageNoError
    else {
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = { mutablePointer, pointer in
      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }

    var result: CVPixelBuffer?

    // Converts the thumbnail vImage buffer to CVPixelBuffer
    let conversionStatus = CVPixelBufferCreateWithBytes(
      nil,
      Int(size.width), Int(size.height),
      CVPixelBufferGetPixelFormatType(self),
      resultAddress,
      resultRowBytes,
      releaseCallBack,
      nil,
      nil,
      &result
    )

    guard conversionStatus == kCVReturnSuccess else {
      free(resultAddress)
      return nil
    }

    return result
  }

  /// Returns the RGB `Data` representation of the given image buffer.
  ///
  /// - Parameters:
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  func rgbData(
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(self, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
    guard let sourceData = CVPixelBufferGetBaseAddress(self) else {
      return nil
    }

    let width = CVPixelBufferGetWidth(self)
    let height = CVPixelBufferGetHeight(self)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(self)
    let destinationBytesPerRow = Constants.rgbPixelChannels * width

    // Assign input image to `sourceBuffer` to convert it.
    var sourceBuffer = vImage_Buffer(
      data: sourceData,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: sourceBytesPerRow)

    // Make `destinationBuffer` and `destinationData` for its data to be assigned.
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      os_log("Error: out of memory", type: .error)
      return nil
    }
    defer { free(destinationData) }
    var destinationBuffer = vImage_Buffer(
      data: destinationData,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: destinationBytesPerRow)

    // Convert image type.
    switch CVPixelBufferGetPixelFormatType(self) {
    case kCVPixelFormatType_32BGRA:
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32ARGB:
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    default:
      os_log("The type of this image is not supported.", type: .error)
      return nil
    }

    // Make `Data` with converted image.
    let imageByteData = Data(
      bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)

    if isModelQuantized { return imageByteData }

    let imageBytes = [UInt8](imageByteData)
    var floats : [Float] = []
    floats.reserveCapacity(width * height * Constants.rgbPixelChannels)
    for y in 0 ..< height {
      for x in 0 ..< width {
        floats.append(Float(imageBytes[y * destinationBytesPerRow + x * 3]) - Constants.mean_R)
        floats.append(Float(imageBytes[y * destinationBytesPerRow + x * 3 + 1]) - Constants.mean_G)
        floats.append(Float(imageBytes[y * destinationBytesPerRow + x * 3 + 2]) - Constants.mean_B)
      }
    }
    return Data(copyingBufferOf: floats)
  }
    // Resize with padding
    // code from https://stackoverflow.com/questions/53734335/accessing-pixels-outside-of-the-cvpixelbuffer-that-has-been-extended-with-paddin
    // modify by hobeom
    public func resizePixelBuffer(from source: CGRect, to size: CGSize) -> CVPixelBuffer? {
        let srcPixelBuffer = self
        let cropX = Int(source.minX)
        let cropY = Int(source.minY)
        let cropWidth = Int(source.width)
        let cropHeight = Int(source.height)
        let scaleWidth = Int(size.width)
        let scaleHeight = Int(size.height)
        
        let flags = CVPixelBufferLockFlags(rawValue: 0)
        let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
        guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, flags) else {
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, flags) }

        guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
            print("Error: could not get pixel buffer base address")
            return nil
        }

        let srcHeight = CVPixelBufferGetHeight(srcPixelBuffer)
        let srcWidth = CVPixelBufferGetWidth(srcPixelBuffer)
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
        let offset = cropY*srcBytesPerRow + cropX*4

        var srcBuffer: vImage_Buffer!
        var paddedSrcPixelBuffer: CVPixelBuffer!

        if (cropX < 0 || cropY < 0 || cropX + cropWidth > srcWidth || cropY + cropHeight > srcHeight) {
            let paddingLeft = abs(min(cropX, 0))
            let paddingRight = max((cropX + cropWidth) - (srcWidth - 1), 0)
            let paddingBottom = max((cropY + cropHeight) - (srcHeight - 1), 0)
            let paddingTop = abs(min(cropY, 0))

            let paddedHeight = paddingTop + srcHeight + paddingBottom
            let paddedWidth = paddingLeft + srcWidth + paddingRight

            guard kCVReturnSuccess == CVPixelBufferCreate(kCFAllocatorDefault, paddedWidth, paddedHeight, pixelFormat, nil, &paddedSrcPixelBuffer) else {
                print("failed to allocate a new padded pixel buffer")
                return nil
            }

            guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(paddedSrcPixelBuffer, flags) else {
                return nil
            }

            guard let paddedSrcData = CVPixelBufferGetBaseAddress(paddedSrcPixelBuffer) else {
                print("Error: could not get padded pixel buffer base address")
                return nil
            }

            let paddedBytesPerRow = CVPixelBufferGetBytesPerRow(paddedSrcPixelBuffer)
            for yIndex in paddingTop..<srcHeight+paddingTop {
                let dstRowStart = paddedSrcData.advanced(by: yIndex*paddedBytesPerRow).advanced(by: paddingLeft*4)
                let srcRowStart = srcData.advanced(by: (yIndex - paddingTop)*srcBytesPerRow)
                dstRowStart.copyMemory(from: srcRowStart, byteCount: srcBytesPerRow)
            }

            let paddedOffset = (cropY + paddingTop)*paddedBytesPerRow + (cropX + paddingLeft)*4
            srcBuffer = vImage_Buffer(data: paddedSrcData.advanced(by: paddedOffset),
                                      height: vImagePixelCount(cropHeight),
                                      width: vImagePixelCount(cropWidth),
                                      rowBytes: paddedBytesPerRow)

        } else {
            srcBuffer = vImage_Buffer(data: srcData.advanced(by: offset),
                                      height: vImagePixelCount(cropHeight),
                                      width: vImagePixelCount(cropWidth),
                                      rowBytes: srcBytesPerRow)
        }

        let destBytesPerRow = scaleWidth*4
        guard let destData = malloc(scaleHeight*destBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        var destBuffer = vImage_Buffer(data: destData,
                                       height: vImagePixelCount(scaleHeight),
                                       width: vImagePixelCount(scaleWidth),
                                       rowBytes: destBytesPerRow)

        let vImageFlags: vImage_Flags = vImage_Flags(kvImageEdgeExtend)
        let error = vImageScale_ARGB8888(&srcBuffer, &destBuffer, nil, vImageFlags)
        if error != kvImageNoError {
            print("Error:", error)
            free(destData)
            return nil
        }

        let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
            if let ptr = ptr {
                free(UnsafeMutableRawPointer(mutating: ptr))
            }
        }

        var dstPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreateWithBytes(nil, scaleWidth, scaleHeight,
                                                  pixelFormat, destData,
                                                  destBytesPerRow, releaseCallback,
                                                  nil, nil, &dstPixelBuffer)
        if status != kCVReturnSuccess {
            print("Error: could not create new pixel buffer")
            free(destData)
            return nil
        }

        if paddedSrcPixelBuffer != nil {
            CVPixelBufferUnlockBaseAddress(paddedSrcPixelBuffer, flags)
        }


        return dstPixelBuffer
    }
}
