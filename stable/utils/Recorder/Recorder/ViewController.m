//
//  ViewController.m
//  Recorder
//
//  Created by Vu Tran on 24/1/20.
//  Copyright © 2020 Vu Tran. All rights reserved.
//

#import "ViewController.h"
#import "CaptureManager.h"
#import <AssetsLibrary/AssetsLibrary.h>
#import <Photos/Photos.h>

@interface ViewController ()
<CaptureManagerDelegate, UITextFieldDelegate>
{
    BOOL afLocked;
    BOOL aeLocked;
    uint32_t pauseCount;
    UIImage *rec_icon;
    UIImage *pse_icon;
    UIImage *stp_icon;
    UIImage *ply_icon;
    UIImage *con_icon;
    UIImage *dis_icon;
    UIImage *chk_icon;
    UIImage *emp_icon;
}
@property (nonatomic, strong) CaptureManager *captureManager;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
//    [self cleanDocuments];
    afLocked = NO;
    aeLocked = NO;
    pauseCount = 0;
    self.captureManager = [[CaptureManager alloc] initWithPreviewView:self.preView preferredCameraType:CameraTypeBack
        outputMode:OutputModeVideoData];
    self.captureManager.delegate = self;
    self.txtAddr.delegate = self;
    rec_icon = [[UIImage imageNamed:@"ShutterButtonRec"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    pse_icon = [[UIImage imageNamed:@"ShutterButtonPause"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    stp_icon = [[UIImage imageNamed:@"ShutterButtonStop"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    ply_icon = [[UIImage imageNamed:@"ShutterButtonPlay"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    con_icon = [[UIImage imageNamed:@"ButtonConnected"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    dis_icon = [[UIImage imageNamed:@"ButtonDisconnected"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    chk_icon = [[UIImage imageNamed:@"CheckButtonChecked"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    emp_icon = [[UIImage imageNamed:@"CheckButtonEmpty"] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    [self.btnRecord setBackgroundImage:rec_icon forState:UIControlStateNormal];
    [self.btnRecord setTintColor:[UIColor redColor]];
    [self.btnAF setBackgroundImage:emp_icon forState:UIControlStateNormal];
    [self.btnAF setTintColor:[UIColor greenColor]];
    [self.btnAE setBackgroundImage:emp_icon forState:UIControlStateNormal];
    [self.btnAE setTintColor:[UIColor greenColor]];
    [self.btnConnect setBackgroundImage:dis_icon forState:UIControlStateNormal];
    [self.btnConnect setTintColor:[UIColor redColor]];
}

- (void) cleanDocuments {
    NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0];
    NSArray *filelist = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:documentsDirectory error:nil];
    unsigned long count = [filelist count];
    for(int i = 0; i < count; i++) {
        [[NSFileManager defaultManager] removeItemAtPath:[documentsDirectory stringByAppendingPathComponent:[filelist objectAtIndex:i]] error:nil];
    }
}

- (void)startRecording:(NSURL *)outputFileURL {
    if([self.captureManager isConnected]) {
        [self.btnRecord setBackgroundImage:ply_icon forState:UIControlStateNormal];
        [self.btnRecord setTintColor:[UIColor greenColor]];
    } else {
        [self.btnRecord setBackgroundImage:stp_icon forState:UIControlStateNormal];
        [self.btnRecord setTintColor:[UIColor redColor]];
    }
    NSLog(@"Start Recording");
}

- (void)finishRecording:(NSURL *)outputFileURL error:(NSError *)error {
    [self.btnRecord setBackgroundImage:rec_icon forState:UIControlStateNormal];
    [self.btnRecord setTintColor:[UIColor greenColor]];
    NSLog(@"Finish Recording");
}

- (void) pauseRecoding
{
    pauseCount++;
    if(pauseCount < 20) [self.captureManager resume];
    else {
        [self.btnRecord setEnabled:YES];
        [self.btnRecord setBackgroundImage:ply_icon forState:UIControlStateNormal];
        [self.btnRecord setTintColor:[UIColor greenColor]];
    }
//    sleep(3);
//    [self.captureManager resume];
//    [self.btnRecord setBackgroundImage:pse_icon forState:UIControlStateNormal];
//    [self.btnRecord setTintColor:[UIColor redColor]];
//    [self.btnRecord setEnabled:NO];
}

- (void)updateConnStatus:(BOOL)status {
    if(status) {
        [self.btnConnect setBackgroundImage:con_icon forState:UIControlStateNormal];
        [self.btnConnect setTintColor:[UIColor greenColor]];
    } else {
        [self.btnConnect setBackgroundImage:dis_icon forState:UIControlStateNormal];
        [self.btnConnect setTintColor:[UIColor redColor]];
        if([self.captureManager isRecording])
            [self.btnRecord setBackgroundImage:stp_icon forState:UIControlStateNormal];
        [self.btnRecord setTintColor:[UIColor redColor]];
    }
}

- (IBAction)btnRecPressed:(UIButton *)sender {
    if([self.captureManager isRecording]) {
        if([self.captureManager isConnected]) {
            sleep(1);
            [self.captureManager resume];
            [self.btnRecord setBackgroundImage:pse_icon forState:UIControlStateNormal];
            [self.btnRecord setTintColor:[UIColor redColor]];
            [self.btnRecord setEnabled:NO];
        } else {
            [self.captureManager stopRecording];
        }
    } else {
        [self.captureManager startRecording];
    }
}

- (IBAction)btnConnPressed:(UIButton *)sender {
    if([self.captureManager isConnected])
        [self.captureManager disconnect];
    else
        if([self.captureManager isRecording]) {
            NSLog(@"Cannot connect while recording");
        } else {
            [self.captureManager connect:self.txtAddr.text requestPort:4747];
        }
}

- (IBAction)segFpsChanged:(UISegmentedControl *)sender {
    CGFloat desiredFps = 30.0;;
    switch (self.segFps.selectedSegmentIndex) {
        case 0:
        default:
        {
            break;
        }
        case 1:
            desiredFps = 60.0;
            break;
        case 2:
            desiredFps = 120.0;
            break;
        case 3:
            desiredFps = 240.0;
            break;
    }
    [self.captureManager switchFormatWithDesiredFPS:desiredFps];
}

- (IBAction)btnAFTapped:(id)sender {
    if ([self.captureManager toggleAF]) {
        [self.btnAF setTitle:@"MF" forState:UIControlStateNormal];
        [self.btnAF setTintColor:[UIColor redColor]];
    } else {
        [self.btnAF setTitle:@"AF" forState:UIControlStateNormal];
        [self.btnAF setTintColor:[UIColor greenColor]];
    }
}
- (IBAction)btnAETapped:(id)sender {
    if ([self.captureManager toggleAE]) {
        [self.btnAE setTitle:@"ME" forState:UIControlStateNormal];
        [self.btnAE setTintColor:[UIColor redColor]];
    } else {
        [self.btnAE setTitle:@"AE" forState:UIControlStateNormal];
        [self.btnAE setTintColor:[UIColor greenColor]];
    }
}

- (BOOL)textFieldShouldReturn:(UITextField *)textField {
    [textField resignFirstResponder];
    return NO;
}

@end
