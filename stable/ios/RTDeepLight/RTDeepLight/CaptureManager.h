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
- (void) updateResult: (NSString *) text;
- (void) updateScreenCorners: (float*) corners;
- (void) clearResult;
@end


@interface CaptureManager : NSObject

@property (nonatomic, assign) id<CaptureManagerDelegate> delegate;
@property (nonatomic, copy) void (^onBuffer)(CMSampleBufferRef sampleBuffer);

- (instancetype)initWithPreviewView:(UIView *)previewView
                preferredCameraType:(CameraType)cameraType
                         outputMode:(OutputMode)outputMode;
- (void) startProcessing;
- (void) stopProcessing;
- (void) switchFormatWithDesiredFPS:(CGFloat)desiredFPS;
- (BOOL) toggleAE;
- (BOOL) toggleAF;

@end

#endif /* CaptureManager_h */
