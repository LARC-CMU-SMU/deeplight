//
//  ViewController.m
//  OLDeepLight
//
//  Created by Vu Tran on 5/7/20.
//  Copyright © 2020 Vu Tran. All rights reserved.
//

#import "ViewController.h"
#import "Processing.h"
#import <CoreMotion/CoreMotion.h>

@interface ViewController ()
{
    UIImage *play_icon;
    UIImage *next_icon;
    int temdata[100];
    NSMutableArray* elapse_times;
    NSMutableArray* str_data;
    dispatch_queue_t processingQueue;
    BOOL stopTesting;
}
@property (nonatomic, strong) Processing* deeplightProcessor;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    play_icon = [[UIImage imageNamed:@"ShutterButtonPlay"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    next_icon = [[UIImage imageNamed:@"ShutterButtonNext"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    [self.btnPlay setBackgroundImage:play_icon forState:UIControlStateNormal];
    [self.btnPlay setTintColor:[UIColor redColor]];
    [self.btnNext setBackgroundImage:next_icon forState:UIControlStateNormal];
    [self.btnNext setTintColor:[UIColor redColor]];
    [self.btnTest setBackgroundImage:play_icon forState:UIControlStateNormal];
    [self.btnTest setTintColor:[UIColor redColor]];
    NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0];
    NSLog(@"Directory path: %@", documentsDirectory);
    NSString* recFileName = [documentsDirectory stringByAppendingPathComponent:@"/hand.mov"];
    self.deeplightProcessor = [[Processing alloc] initWithFilePath:recFileName originalWidth:1280 originalHeight:720 extractorWidth:256 extractorHeight:256 decoderWidth:299 decoderHeight:299];
    elapse_times = [NSMutableArray arrayWithCapacity:10000];
    str_data = [NSMutableArray arrayWithCapacity:10000];
    processingQueue = dispatch_queue_create("processingqueue", DISPATCH_QUEUE_SERIAL);
    stopTesting = NO;
}

- (IBAction)btnPlayTapped:(id)sender {
    //Start the first dummy
    [self.deeplightProcessor nextFrame];
    [self.deeplightProcessor detectScreen];
    [self.deeplightProcessor drawBorder];
    [self.deeplightProcessor extractScreen];
    [self.deeplightProcessor decode: self->temdata];
    self.ivMain.image = [self.deeplightProcessor getFrame:2];
    dispatch_async(processingQueue, ^{
        int detect_interval = 1;
        int frame_index = 0;
        while(!self->stopTesting) {
            [self.deeplightProcessor nextFrame];
            if(frame_index % detect_interval == 0) [self.deeplightProcessor detectScreen];
            [self.deeplightProcessor drawBorder];
            [self.deeplightProcessor extractScreen];
            [self.deeplightProcessor decode: self->temdata];
//            [self.RSDecoder sendBits:self->temdata numBits:100];
            frame_index++;
            if((frame_index % 100) == 0) [self.deeplightProcessor resetVideo];
            if((frame_index % 100) == 0) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self.lbDecodedText setText:[NSString stringWithFormat:@"%d", frame_index]];
                });
            }
        }
        NSLog(@"Finish Testing Power Consumption\n");
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.lbDecodedText setText:@"Finish"];
        });
    });
}

- (IBAction)btnNextTapped:(id)sender {
    //Start the first dummy
    [self.deeplightProcessor nextFrame];
    [self.deeplightProcessor detectScreen];
    [self.deeplightProcessor drawBorder];
    [self.deeplightProcessor extractScreen];
    [self.deeplightProcessor decode: self->temdata];
    self.ivMain.image = [self.deeplightProcessor getFrame:2];
    dispatch_async(processingQueue, ^{
        NSString* all_text = @"";
        uint64_t total_elapse = 0;
        uint64_t ave_read_time = 0;
        uint64_t ave_detect_time = 0;
        uint64_t ave_extract_time = 0;
        uint64_t ave_decode_time = 0;
        uint64_t ave_rscode_time = 0;
        int detect_interval = 1;
        for(int i = 0; i < 100; i++) {
            uint64_t init_time = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            [self.deeplightProcessor nextFrame];
            uint64_t start_detect = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            if(i % detect_interval == 0) [self.deeplightProcessor detectScreen];
            uint64_t start_extract = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        //    [self.deeplightProcessor drawBorder];
            [self.deeplightProcessor extractScreen];
            uint64_t start_decode = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            [self.deeplightProcessor decode: self->temdata];
            uint64_t start_rscode = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            NSString* result = [self.deeplightProcessor correct: self->temdata];
            uint64_t end = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            ave_read_time += start_detect - init_time;
            ave_detect_time += start_extract - start_detect;
            ave_extract_time += start_decode - start_extract;
            ave_decode_time += start_rscode - start_decode;
            ave_rscode_time += end - start_rscode;
            total_elapse += end - start_detect;
            NSString* log_text = [NSString stringWithFormat:@"%lld, %lld, %lld, %lld\n", start_extract - start_detect, start_decode - start_extract, start_rscode - start_decode, end - start_rscode];
            [self->elapse_times addObject:log_text];
            NSString *tem_txt = @"";
            for(int bit_index = 0; bit_index < 100; bit_index++) {
                tem_txt = [tem_txt stringByAppendingFormat:@"%d ", self->temdata[bit_index]];
            }
            tem_txt = [tem_txt stringByAppendingString:@"\n"];
            [self->str_data addObject:tem_txt];
            NSLog(@"%@", result);
            all_text = [all_text stringByAppendingFormat:@"%@", result];
        }
        NSLog(@"%@", all_text);
        NSFileHandle *myHandle;
        NSFileHandle *dataHandle;
        NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask, YES);
        NSString *documentsDirectory = [paths objectAtIndex:0];
        NSString *tsFilePath = [documentsDirectory stringByAppendingString:@"/log_processing_time_ios.txt"];
        NSString *dataFilePath = [documentsDirectory stringByAppendingString:@"/temdata.txt"];
        if([[NSFileManager defaultManager] fileExistsAtPath:tsFilePath]) {
            NSError *error;
            [[NSFileManager defaultManager] removeItemAtPath:tsFilePath error:&error];
        }
        if([[NSFileManager defaultManager] fileExistsAtPath:dataFilePath]) {
            NSError *error;
            [[NSFileManager defaultManager] removeItemAtPath:dataFilePath error:&error];
        }
        [[NSFileManager defaultManager] createFileAtPath:tsFilePath contents:nil attributes:nil];
        [[NSFileManager defaultManager] createFileAtPath:dataFilePath contents:nil attributes:nil];
        myHandle = [NSFileHandle fileHandleForWritingAtPath:tsFilePath];
        dataHandle = [NSFileHandle fileHandleForWritingAtPath:dataFilePath];
        for(int i = 0; i < self->elapse_times.count; i++) {
            NSString* tem = [self->elapse_times objectAtIndex:i];
            [myHandle writeData:[tem dataUsingEncoding:NSUTF8StringEncoding]];
        }
        for(int i = 0; i < 100; i++) {
            NSString* tem = [self->str_data objectAtIndex:i];
            [dataHandle writeData:[tem dataUsingEncoding:NSUTF8StringEncoding]];
        }
        [myHandle closeFile];
        [dataHandle closeFile];
        double denorm = 1000000000.0;
        NSLog(@"========== Reading Frames, Detecting, Extracting, Decoding, RSDecoding: %f, %f, %f, %f, %f\n",ave_read_time/denorm , ave_detect_time/denorm, ave_extract_time/denorm, ave_decode_time/denorm, ave_rscode_time/denorm);
        NSLog(@"Total Elapse Time: %f\n", total_elapse/denorm);
    });
}
- (IBAction)btnTestTapped:(id)sender {
    self->stopTesting = YES;
}

-(void) updateNewData:(NSString *)data
{
    NSLog(@"============== Decoded Text: %@\n", data);
}

@end
