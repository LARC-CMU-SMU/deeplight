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

- (id) initWithFilePath: (NSString *)filePath originalWidth:(int)orgWidth originalHeight:(int)orgHeight extractorWidth:(int)extWidth extractorHeight:(int)extHeight decoderWidth:(int)decwidth decoderHeight:(int)decHeight;

- (void) nextFrame;

-(void) resetVideo;

- (UIImage *) getFrame: (int) type;

- (void) detectScreen;

- (void) drawBorder;

- (void) extractScreen;

- (void) decode: (int*) data;

- (NSString*) correct: (int*) data;

@end

#endif /* Processing_h */
