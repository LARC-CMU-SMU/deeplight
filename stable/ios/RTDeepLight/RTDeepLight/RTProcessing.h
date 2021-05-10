//
//  Processing.h
//  RTDeepLight
//
//  Created by Vu Tran on 14/6/20.
//  Copyright © 2020 Vu Tran. All rights reserved.
//

#ifndef Processing_h
#define Processing_h

#import <UIKit/UIKit.h>

@interface Processing : NSObject

- (id) initWithCamera: (int)orgWidth originalHeight:(int)orgHeight extractorWidth:(int)extWidth extractorHeight:(int)extHeight decoderWidth:(int)decwidth decoderHeight:(int)decHeight;

- (void) setFrames: (CVPixelBufferRef*)imageBuffer start_index: (uint32_t) offset size: (uint32_t)buffer_size;

//- (void) nextFrame: (CVPixelBufferRef) frame;

- (void) detectScreen: (CVPixelBufferRef) frame;

- (void) extractScreen;

- (void) decode: (int*)data;

- (NSString*) correct: (int*)data;

- (void) getScreenCorners: (float *) corners;

@end

#endif /* Processing_h */
