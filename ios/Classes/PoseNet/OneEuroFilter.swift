//
//  OneEuroFilter.swift
//  everex_tflite
//
//  Created by 김동주 on 2023/01/26.
//

import Foundation


public class OneEuroFilter {
    var min_cutoff: Float32
    var beta: Float32
    var d_cutoff: Float32
    var num_keypoints:Int
    var t_prev: Array<Date> = []
    var x_prev: Array<(y: Float, x: Float)> = []
    var dx_prev: Array<(y: Float, x: Float)> = []
    
    init(num_keypoints:Int = 16, min_cutoff:Float32 = 1.7,  beta:Float32 = 0.3,  d_cutoff:Float32 = 1.0) {
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.num_keypoints = num_keypoints
        print(num_keypoints)
        for _ in 0..<num_keypoints{
            self.t_prev.append(Date())
            self.x_prev.append((y: -1.0, x: -1.0))
            self.dx_prev.append((y: -1.0, x: -1.0))
        }
    }
    
    func smoothing_factor(t_e: Float32, cutoff: Float32) -> Float32 {
        let r = 2 * Float32.pi * cutoff * t_e
        return r / (r + 1)
    }
    
    func exponential_smoothing(a: Float32, x: Float32, x_prev: Float32) -> Float32 {
        return a * x + (1 - a) * x_prev
    }
    
    public func Filter(coords :Array<(y: Float, x: Float)>) -> Array<(y: Float, x: Float)> {
        
        var coords_hat:Array<(y: Float, x: Float)> = []
        var coords_dx:Array<(y: Float, x: Float)> = []
        var t:Array<Date> = []
        for i in 0..<self.num_keypoints{
            let t_e = Float32(Date().timeIntervalSince(self.t_prev[i]) * 1000)
            // The filtered derivative of the signal.
            let a_d = self.smoothing_factor(t_e: t_e, cutoff: self.d_cutoff)
            
            let dx = (coords[i].x - self.x_prev[i].x) / t_e
            let dx_hat = self.exponential_smoothing(a: a_d, x: dx, x_prev: self.dx_prev[i].x)
            let x_cutoff = self.min_cutoff + self.beta * abs(dx_hat)
            let xa = self.smoothing_factor(t_e: t_e, cutoff: x_cutoff)
            let x_hat = self.exponential_smoothing(a: xa, x: coords[i].x, x_prev: x_prev[i].x)
            
            let dy = (coords[i].y - self.x_prev[i].y) / t_e
            let dy_hat = self.exponential_smoothing(a: a_d, x: dy, x_prev: self.dx_prev[i].y)
            let y_cutoff = self.min_cutoff + self.beta * abs(dy_hat)
            let ya = self.smoothing_factor(t_e: t_e, cutoff: y_cutoff)
            let y_hat = self.exponential_smoothing(a: ya, x: coords[i].y, x_prev: x_prev[i].y)
            
            coords_hat.append((y: y_hat, x: x_hat))
            coords_dx.append((y: dy_hat, x: dx_hat))
            t.append(Date())
        }
        
        // Memorize the previous values.
        self.x_prev = coords_hat
        self.dx_prev = coords_dx
        self.t_prev = t
        
        return coords_hat
    }
}
