//
//  CaptureManager.m
//  DeepCode
//
//  Created by Vu Tran on 21/11/18.
//  Copyright © 2018 Vu Tran. All rights reserved.
//

#import "CaptureManager.h"
#import <AVFoundation/AVFoundation.h>

@interface CaptureManager ()
<AVCaptureFileOutputRecordingDelegate, AVCaptureVideoDataOutputSampleBufferDelegate, NSStreamDelegate>
{
    BOOL recordingRequested;
    BOOL recording;
    BOOL paused;
    BOOL aeLocked;
    BOOL afLocked;
    int exposure;
    uint32_t numframe;
    dispatch_queue_t movieWritingQueue;
    BOOL connected;
    CFReadStreamRef readStream;
    CFWriteStreamRef writeStream;
    NSInputStream   *inputStream;
    NSOutputStream  *outputStream;
}
@property (nonatomic, strong) AVAssetWriter *assetWriter;
@property (nonatomic, strong) AVAssetWriterInput *assetWriterVideoInput;
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
        recordingRequested = NO;
        recording = NO;
        paused = NO;
        connected = NO;
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
        
        movieWritingQueue = dispatch_queue_create("writingqueue", DISPATCH_QUEUE_SERIAL);
        
        self.videoConnection = [videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
        
        [self.captureSession startRunning];
    }
    NSLog(@"White balance gain: %f", [self.videoDevice maxWhiteBalanceGain]);
    NSLog(@"Current white balance gain: %f, %f, %f", [self.videoDevice deviceWhiteBalanceGains].redGain, [self.videoDevice deviceWhiteBalanceGains].greenGain, [self.videoDevice deviceWhiteBalanceGains].blueGain);
    return self;
}

- (BOOL)setupAssetWriterVideoInput:(CMFormatDescriptionRef)currentFormatDescription
{
    CMVideoDimensions dimensions = CMVideoFormatDescriptionGetDimensions(currentFormatDescription);
    NSDictionary *videoCompressionSettings = [NSDictionary dictionaryWithObjectsAndKeys:
                                                  AVVideoCodecTypeJPEG, AVVideoCodecKey,
                                                  [NSNumber numberWithInteger:dimensions.width], AVVideoWidthKey,
                                                  [NSNumber numberWithInteger:dimensions.height], AVVideoHeightKey,
                                                  [NSDictionary dictionaryWithObjectsAndKeys:
    //                                               [NSNumber numberWithInteger:bitsPerSecond], AVVideoAverageBitRateKey,
                                                   [NSNumber numberWithFloat:1.0],
                                                   AVVideoQualityKey,
                                                   [NSNumber numberWithInteger:1], AVVideoMaxKeyFrameIntervalKey,
                                                   nil], AVVideoCompressionPropertiesKey,
                                                  nil];
    
    NSLog(@"videoCompressionSetting:%@", videoCompressionSettings);
    
    if ([self.assetWriter canApplyOutputSettings:videoCompressionSettings forMediaType:AVMediaTypeVideo]) {
        
        self.assetWriterVideoInput = [[AVAssetWriterInput alloc] initWithMediaType:AVMediaTypeVideo
                                                                    outputSettings:videoCompressionSettings];
        
        self.assetWriterVideoInput.expectsMediaDataInRealTime = YES;
//        self.assetWriterVideoInput.transform = [self transformFromCurrentVideoOrientationToOrientation:referenceOrientation];
        
        if ([self.assetWriter canAddInput:self.assetWriterVideoInput]) {
            
            [self.assetWriter addInput:self.assetWriterVideoInput];
        }
        else {
            
            NSLog(@"Couldn't add asset writer video input.");
            return NO;
        }
    }
    else {
        
        NSLog(@"Couldn't apply video output settings.");
        return NO;
    }
    
    return YES;
}

- (void)writeSampleBuffer:(CMSampleBufferRef)sampleBuffer
                   ofType:(NSString *)mediaType
{
    CMTime timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    if (self.assetWriter.status == AVAssetWriterStatusUnknown) {
        if ([self.assetWriter startWriting]) {
            [self.assetWriter startSessionAtSourceTime:timestamp];
        }
        else {
            
            NSLog(@"AVAssetWriter startWriting error:%@", self.assetWriter.error);
        }
    }
    
    if (self.assetWriter.status == AVAssetWriterStatusWriting) {
        if (mediaType == AVMediaTypeVideo) {
            if (self.assetWriterVideoInput.readyForMoreMediaData) {
                if ([self.assetWriterVideoInput appendSampleBuffer:sampleBuffer]) {
                    self->numframe += 1;
                    [self.tsArray addObject:[NSNumber numberWithLong:timestamp.value]];
                } else {
                    NSLog(@"AVAssetWriterInput video appendSapleBuffer error:%@", self.assetWriter.error);
                }
            }
        }
    }
}

- (void) startRecording {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSString *filePath = [documentsDirectory stringByAppendingPathComponent:@"/record.mov"];
    self.videoFileURL = [NSURL URLWithString:[@"file://" stringByAppendingString:filePath]];
    NSError *error;
    [[NSFileManager defaultManager] removeItemAtPath:filePath error:&error];
    self.assetWriter = [[AVAssetWriter alloc] initWithURL:self.videoFileURL
                                                 fileType:AVFileTypeQuickTimeMovie
                                                    error:&error];
    self->numframe = 0;
    self.tsArray = [NSMutableArray array];
    if ([self.delegate respondsToSelector:@selector(startRecording:)]) {
        [self.delegate startRecording:self.videoFileURL];
    }
    recording = YES;
    recordingRequested = YES;
}

- (void) stopRecording {
    dispatch_async(movieWritingQueue, ^{
        [self.assetWriter finishWritingWithCompletionHandler:^{
            self.assetWriterVideoInput = nil;
            self.assetWriter = nil;
            dispatch_async(dispatch_get_main_queue(), ^{
                NSFileHandle *myHandle;
                NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask, YES);
                NSString *documentsDirectory = [paths objectAtIndex:0];
                NSString *tsFilePath = [documentsDirectory stringByAppendingPathComponent:@"/record.ts"];
                if([[NSFileManager defaultManager] fileExistsAtPath:tsFilePath]) {
                    NSError *error;
                    [[NSFileManager defaultManager] removeItemAtPath:tsFilePath error:&error];
                }
                [[NSFileManager defaultManager] createFileAtPath:tsFilePath contents:nil attributes:nil];
                myHandle = [NSFileHandle fileHandleForWritingAtPath:tsFilePath];
//                [myHandle seekToFileOffset:0];
                [myHandle writeData:[NSData dataWithBytes:&(self->numframe) length:sizeof(self->numframe)]];
                for(int i = 0; i < self.tsArray.count; i++) {
                    int64_t value = ((NSNumber *)[self.tsArray objectAtIndex:i]).longLongValue;
                    [myHandle writeData:[NSData dataWithBytes:&(value) length:sizeof(value)]];
                }
                [myHandle closeFile];
                if ([self.delegate respondsToSelector:@selector(finishRecording:error:)]) {
                    [self.delegate finishRecording:self.videoFileURL error:nil];
                }
            });
        }];
    });
    recording = NO;
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
//                maxWidth = width;
            }
        }
    }
    
    if (selectedFormat) {
        
        if ([videoDevice lockForConfiguration:nil]) {
            [videoDevice setAutomaticallyAdjustsVideoHDREnabled:NO];
            [videoDevice setVideoHDREnabled:NO];
            NSLog(@"HDREnabled:%i", videoDevice.isVideoHDREnabled);
            NSLog(@"Auto HDR enabled:%i", videoDevice.automaticallyAdjustsVideoHDREnabled);
            NSLog(@"selected format:%@", selectedFormat);
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
        AVCaptureWhiteBalanceGains tem = [self.videoDevice deviceWhiteBalanceGains];
        // 1.858154, 1.000000, 2.241943
        tem.blueGain = 2.0;
        tem.greenGain = 1.0;
        tem.redGain = 1.8;
        [self.videoDevice setWhiteBalanceModeLockedWithDeviceWhiteBalanceGains:tem completionHandler:nil];
    } else {
        [self.videoDevice setWhiteBalanceMode:AVCaptureWhiteBalanceModeAutoWhiteBalance];
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

- (void) resume
{
    paused = NO;
    uint8_t code = 0x55;
    [outputStream write:(uint8_t*)(&code) maxLength:sizeof(code)];
}

- (BOOL) isRecording
{
    return recording;
}

- (BOOL) isConnected
{
    return connected;
}

// =============================================================================
#pragma mark - AVCaptureFileOutputRecordingDelegate

- (void)                 captureOutput:(AVCaptureFileOutput *)captureOutput
    didStartRecordingToOutputFileAtURL:(NSURL *)videoFileURL
                       fromConnections:(NSArray *)connections
{
    NSLog(@"Delegate Start Recording");
    if ([self.delegate respondsToSelector:@selector(startRecording:)]) {
        [self.delegate startRecording:videoFileURL];
    }
}

- (void)                 captureOutput:(AVCaptureFileOutput *)captureOutput
   didFinishRecordingToOutputFileAtURL:(NSURL *)outputvideoFileURL
                       fromConnections:(NSArray *)connections error:(NSError *)error
{
    //    [self saveRecordedFile:outputvideoFileURL];
    NSLog(@"Delegate Finish Recording");
    if ([self.delegate respondsToSelector:@selector(finishRecording:error:)]) {
        [self.delegate finishRecording:outputvideoFileURL error:error];
    }
}
// =============================================================================
#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate

- (void)    captureOutput:(AVCaptureOutput *)captureOutput
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection
{
    if(paused) return;
    if (self.onBuffer) {
        self.onBuffer(sampleBuffer);
    }
    if(recording) {
        CFRetain(sampleBuffer);
        dispatch_async(movieWritingQueue, ^{
            BOOL ready = NO;
            if(self->recordingRequested) {
                self->recordingRequested = NO;
                CMFormatDescriptionRef formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer);
                ready = [self setupAssetWriterVideoInput: formatDescription];
            }
            [self writeSampleBuffer:sampleBuffer ofType:AVMediaTypeVideo];
            CFRelease(sampleBuffer);
        });
    }
}

//========================================================================
- (void) connect:(NSString *)ipaddr requestPort:(int) port
{
    NSLog(@"Setting up connection to %@ : %i", ipaddr, port);
    CFStreamCreatePairWithSocketToHost(kCFAllocatorDefault, (__bridge CFStringRef) ipaddr, port, &readStream, &writeStream);
    outputStream = (__bridge NSOutputStream *)writeStream;
    inputStream = (__bridge NSInputStream *)readStream;
    
    [outputStream setDelegate:self];
    [inputStream setDelegate:self];
    [outputStream scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    [inputStream scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    [outputStream open];
    [inputStream open];
}

- (void) disconnect
{
    NSLog(@"================ Ip Disconnect Request =================\n");
    [outputStream close];
    [inputStream close];
    [inputStream removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    [outputStream removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    [inputStream setDelegate:nil];
    [outputStream setDelegate:nil];
    inputStream = nil;
    outputStream = nil;
    connected = NO;
    [self.delegate updateConnStatus:connected];
}

- (void)stream:(NSStream *)theStream handleEvent:(NSStreamEvent)streamEvent {
    
    NSLog(@"stream event %lu", streamEvent);
    
    switch (streamEvent) {
            
        case NSStreamEventOpenCompleted:
            NSLog(@"Stream opened");
            connected = YES;
            paused = YES;
            break;
        case NSStreamEventHasBytesAvailable:
            NSLog(@"New data arrived");
            if (theStream == inputStream)
            {
                uint8_t command = 0;
                while ([inputStream hasBytesAvailable])
                {
                    [inputStream read:(uint8_t*)(&command) maxLength:sizeof(command)];
                    if(command == 0xAA) {
                        paused = YES;
                        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(5.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{[self.delegate pauseRecoding];});
                        NSLog(@"Pause recording");
                    }
                }
            }
            break;
            
        case NSStreamEventHasSpaceAvailable:
            NSLog(@"Stream has space available now");
            break;
            
        case NSStreamEventErrorOccurred:
            NSLog(@"Stream Error: %@",[theStream streamError].localizedDescription);
            [theStream close];
            [theStream removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
            connected = NO;
            break;
            
        case NSStreamEventEndEncountered:
            NSLog(@"Stream End: %@",[theStream streamError].localizedDescription);
            [theStream close];
            [theStream removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
            connected = NO;
            break;
        case NSStreamStatusClosed:
            NSLog(@"================ Remote Disconnected =================\n");
            connected = NO;
            break;
        default:
            NSLog(@"Unknown event");
    }
    [self.delegate updateConnStatus:connected];
}

@end
