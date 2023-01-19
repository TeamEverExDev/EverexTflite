#import "EverexTflitePlugin.h"
#if __has_include(<everex_tflite/everex_tflite-Swift.h>)
#import <everex_tflite/everex_tflite-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "everex_tflite-Swift.h"
#endif

@implementation EverexTflitePlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftEverexTflitePlugin registerWithRegistrar:registrar];
}
@end
