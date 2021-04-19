//
//  CaptureManager.m
//  DeepCode
//
//  Created by Vu Tran on 21/11/18.
//  Copyright © 2018 Vu Tran. All rights reserved.
//

#import "CaptureManager.h"
#import <AVFoundation/AVFoundation.h>
#import "RTProcessing.h"

#define FRAME_BUFFER_SIZE 32

@interface CaptureManager ()
<AVCaptureFileOutputRecordingDelegate, AVCaptureVideoDataOutputSampleBufferDelegate>
{
    BOOL aeLocked;
    BOOL afLocked;
    BOOL processingRequested;
    uint64_t prevTimestamp;
    uint32_t numframe;
    dispatch_queue_t processingQueue;
    dispatch_queue_t movieWritingQueue;
    CVPixelBufferRef  imgBuffer[FRAME_BUFFER_SIZE];
    uint32_t frameIndex;
    uint64_t tsBuffer[FRAME_BUFFER_SIZE];
    Processing* deeplightProcessor;
    int temdata[100];
    float corners[8];
    NSString *currentText;
}
@property (nonatomic, strong) NSURL *videoFileURL;
@property (nonatomic, strong) NSMutableArray *tsArray;
@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic, strong) AVCaptureDevice *videoDevice;
@property (nonatomic, strong) AVCaptureConnection *videoConnection;
@end


@implementation CaptureManager

- (instancetype)initWithPreviewView:(UIView *)previewView
                preferredCameraType:(CameraType)cameraType
                         outputMode:(OutputMode)outputMode
{
    self = [super init];
    
    if (self) {
        NSError *error;
        self.captureSession = [[AVCaptureSession alloc] init];
        self.captureSession.sessionPreset = AVCaptureSessionPresetInputPriority;
        self.videoDevice = cameraType == CameraTypeFront ? [CaptureManager frontCaptureDevice] : [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        AVCaptureDeviceInput *videoIn = [AVCaptureDeviceInput deviceInputWithDevice:self.videoDevice error:&error];
        
        if (error) {
            NSLog(@"Video input creation failed");
            return nil;
        }
        
        if (![self.captureSession canAddInput:videoIn]) {
            NSLog(@"Video input add-to-session failed");
            return nil;
        }
        [self.captureSession addInput:videoIn];
        NSLog(@"videoDevice.activeFormat:%@", self.videoDevice.activeFormat);
        
        if (previewView) {
            self.previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self.captureSession];
            self.previewLayer.frame = previewView.bounds;
            self.previewLayer.contentsGravity = kCAGravityResizeAspectFill;
            self.previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
            [previewView.layer insertSublayer:self.previewLayer atIndex:0];
            [[self.previewLayer connection] setVideoOrientation:AVCaptureVideoOrientationLandscapeRight];
        }
        
        AVCaptureVideoDataOutput *videoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
        [self.captureSession addOutput:videoDataOutput];
        
        [videoDataOutput setVideoSettings:[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey]];
        
        dispatch_queue_t videoCaptureQueue = dispatch_queue_create("capturequeue", NULL);
        [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
        [videoDataOutput setSampleBufferDelegate:self queue:videoCaptureQueue];
        
        processingQueue = dispatch_queue_create("writingqueue", DISPATCH_QUEUE_SERIAL);
        movieWritingQueue = dispatch_queue_create("cacacaca", DISPATCH_QUEUE_SERIAL);
        
        self.videoConnection = [videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
        
        [self.captureSession startRunning];
        deeplightProcessor = [[Processing alloc] initWithCamera:1920 originalHeight:1080 extractorWidth:256 extractorHeight:256 decoderWidth:299 decoderHeight:299];
        processingRequested = NO;
        prevTimestamp = 0;
        currentText = @"";
    }
    return self;
}

+ (AVCaptureDevice *)frontCaptureDevice {
    
    //    NSArray *videoDevices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceDiscoverySession *captureDeviceDiscoverySession = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionBack];
    NSArray *videoDevices = [captureDeviceDiscoverySession devices];
    for (AVCaptureDevice *device in videoDevices)
    {
        if (device.position == AVCaptureDevicePositionFront)
        {
            return device;
        }
    }
    
    return [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
}

- (void)switchFormatWithDesiredFPS:(CGFloat)desiredFPS
{
    BOOL isRunning = self.captureSession.isRunning;
    
    if (isRunning)  [self.captureSession stopRunning];
    
    AVCaptureDevice *videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceFormat *selectedFormat = nil;
    int32_t desiredWidth = 1920;
    int32_t desiredHeight = 1080;
    AVFrameRateRange *frameRateRange = nil;
    
    for (AVCaptureDeviceFormat *format in [videoDevice formats]) {
//        NSLog(@"supported format:%@", format);
        for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
            
            CMFormatDescriptionRef desc = format.formatDescription;
            CMVideoDimensions dimensions = CMVideoFormatDescriptionGetDimensions(desc);
            int32_t width = dimensions.width;
            int32_t height = dimensions.height;
            if (range.minFrameRate <= desiredFPS && desiredFPS <= range.maxFrameRate && width == desiredWidth && height == desiredHeight) {
                
                selectedFormat = format;
                frameRateRange = range;
            }
        }
    }
    
    if (selectedFormat) {
        
        if ([videoDevice lockForConfiguration:nil]) {
            NSLog(@"selected format:%@", selectedFormat);
            [videoDevice setAutomaticallyAdjustsVideoHDREnabled:NO];
            [videoDevice setVideoHDREnabled:NO];
            NSLog(@"HDREnabled:%i", videoDevice.isVideoHDREnabled);
            NSLog(@"Auto HDR enabled:%i", videoDevice.automaticallyAdjustsVideoHDREnabled);
            videoDevice.activeFormat = selectedFormat;
            videoDevice.activeVideoMinFrameDuration = CMTimeMake(1, (int32_t)desiredFPS);
            videoDevice.activeVideoMaxFrameDuration = CMTimeMake(1, (int32_t)desiredFPS);
            [videoDevice unlockForConfiguration];
        }
    }
    
    if (isRunning) [self.captureSession startRunning];
}

- (BOOL) toggleAE
{
    aeLocked = !aeLocked;
    NSError *error;
    [self.videoDevice lockForConfiguration:&error];
    if(aeLocked) {
        [self.videoDevice setExposureMode:AVCaptureExposureModeLocked];
//        [self.videoDevice setExposureMode:AVCaptureExposureModeCustom];
        [self.videoDevice setExposureModeCustomWithDuration:CMTimeMake(4, 1000) ISO:100 completionHandler:nil];
    } else {
        [self.videoDevice setExposureMode:AVCaptureExposureModeAutoExpose];
    }
    [self.videoDevice unlockForConfiguration];
    NSLog(@"Toggle AE");
    return aeLocked;
}

- (BOOL) toggleAF
{
    afLocked = !afLocked;
    NSError *error;
    [self.videoDevice lockForConfiguration:&error];
    if(afLocked) {
        [self.videoDevice setFocusMode:AVCaptureFocusModeLocked];
    } else {
        [self.videoDevice setFocusMode:AVCaptureFocusModeContinuousAutoFocus];
    }
    [self.videoDevice unlockForConfiguration];
    NSLog(@"Toggle AF");
    return afLocked;
}

// =============================================================================
#pragma mark - AVCaptureFileOutputRecordingDelegate

- (void)                 captureOutput:(AVCaptureFileOutput *)captureOutput
    didStartRecordingToOutputFileAtURL:(NSURL *)videoFileURL
                       fromConnections:(NSArray *)connections
{
    NSLog(@"Delegate Start Recording");
}

- (void)                 captureOutput:(AVCaptureFileOutput *)captureOutput
   didFinishRecordingToOutputFileAtURL:(NSURL *)outputvideoFileURL
                       fromConnections:(NSArray *)connections error:(NSError *)error
{
    //    [self saveRecordedFile:outputvideoFileURL];
    NSLog(@"Delegate Finish Recording");
}
// =============================================================================
#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate

- (void)    captureOutput:(AVCaptureOutput *)captureOutput
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection
{
    if (self.onBuffer) {
        self.onBuffer(sampleBuffer);
    }
//    dispatch_async(movieWritingQueue, ^{
        if(self->numframe > 0) {
            CFRetain(sampleBuffer);
            CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
            CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
            CVPixelBufferLockBaseAddress(imageBuffer, 0);
            int bufferWidth = (int)CVPixelBufferGetWidth(imageBuffer);
            int bufferHeight = (int)CVPixelBufferGetHeight(imageBuffer);
            size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
            uint8_t *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
            if(self->imgBuffer[self->frameIndex] == nil) CVPixelBufferCreate(kCFAllocatorDefault, bufferWidth, bufferHeight, kCVPixelFormatType_32BGRA, NULL, &self->imgBuffer[self->frameIndex]);
            CVPixelBufferLockBaseAddress(self->imgBuffer[self->frameIndex], 0);
            uint8_t *copyBaseAddress = CVPixelBufferGetBaseAddress(self->imgBuffer[self->frameIndex]);
            memcpy(copyBaseAddress, baseAddress, bufferHeight * bytesPerRow);
            CVPixelBufferUnlockBaseAddress(self->imgBuffer[self->frameIndex], 0);
            CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
            CFRelease(sampleBuffer);
            self->tsBuffer[self->frameIndex] = timestamp.value;
            self->numframe--;
            self->frameIndex = (self->frameIndex + 1) % FRAME_BUFFER_SIZE;
//            NSLog(@"Recording %d frames", self->numframe);
        }
//    });
}

- (void) startProcessing
{
//    processingRequested = YES;
    dispatch_async(processingQueue, ^{
//        while(self->processingRequested) {
//            uint32_t index = self->frameIndex - 1;
//            if(index < 0) index = FRAME_BUFFER_SIZE - 1;
        dispatch_async(dispatch_get_main_queue(), ^{[self.delegate updateScreenCorners:self->corners];
            [self.delegate clearResult];
        });
        NSString *all_text = @"";
        [self->deeplightProcessor detectScreen: self->imgBuffer[0]];
        [self->deeplightProcessor getScreenCorners:self->corners];
        for(int offset = 0; offset < FRAME_BUFFER_SIZE-3; offset++) {
            [self->deeplightProcessor setFrames:self->imgBuffer start_index:offset size:FRAME_BUFFER_SIZE];
            [self->deeplightProcessor extractScreen];
            [self->deeplightProcessor decode: self->temdata];
            NSString *text = [self->deeplightProcessor correct: self->temdata];
            NSLog(@"%@", text);
            self->prevTimestamp = self->tsBuffer[offset];
            all_text = [all_text stringByAppendingString:text];
        }
        dispatch_async(dispatch_get_main_queue(), ^{[self.delegate updateScreenCorners:self->corners];
            [self.delegate updateResult:all_text];
        });
//        }
        NSLog(@"!!!!!!!!!!!!!!!!!! STOP PROCESSING !!!!!!!!!!!!!!!!!!\n");
        self->numframe = FRAME_BUFFER_SIZE;
    });
}

- (void) stopProcessing
{
    numframe = FRAME_BUFFER_SIZE;
    frameIndex = 0;
//    processingRequested = NO;
}

@end
