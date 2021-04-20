//
//  CaptureManager.h
//  DeepCode
//
//  Created by Vu Tran on 21/11/18.
//  Copyright © 2018 Vu Tran. All rights reserved.
//

#ifndef CaptureManager_h
#define CaptureManager_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
@import CoreMedia;

typedef NS_ENUM(NSUInteger, CameraType) {
    CameraTypeBack,
    CameraTypeFront,
};

typedef NS_ENUM(NSUInteger, OutputMode) {
    OutputModeVideoData,
    OutputModeMovieFile,
};


@protocol CaptureManagerDelegate <NSObject>
- (void) startRecording:(NSURL *)outputFileURL;
- (void) finishRecording:(NSURL *)outputFileURL error:(NSError *)error;
- (void) pauseRecoding;
- (void) updateConnStatus:(BOOL) status;
@end


@interface CaptureManager : NSObject

@property (nonatomic, assign) id<CaptureManagerDelegate> delegate;
@property (nonatomic, copy) void (^onBuffer)(CMSampleBufferRef sampleBuffer);

- (instancetype)initWithPreviewView:(UIView *)previewView
                preferredCameraType:(CameraType)cameraType
                         outputMode:(OutputMode)outputMode;
- (void) startRecording;
- (void) stopRecording;
- (BOOL) isRecording;
- (void) switchFormatWithDesiredFPS:(CGFloat)desiredFPS;
- (BOOL) toggleAE;
- (BOOL) toggleAF;
- (void) connect: (NSString *)ipaddr requestPort:(int) port;
- (void) disconnect;
- (BOOL) isConnected;
- (void) resume;

@end

#endif /* CaptureManager_h */
