//
//  ViewController.m
//  Recorder
//
//  Created by Vu Tran on 24/1/20.
//  Copyright © 2020 Vu Tran. All rights reserved.
//

#import "ViewController.h"
#import "CaptureManager.h"

@interface ViewController ()
<CaptureManagerDelegate>
{
    BOOL afLocked;
    BOOL aeLocked;
    UIImage *rec_icon;
    UIImage *pse_icon;
    UIImage *stp_icon;
    UIImage *ply_icon;
    UIImage *con_icon;
    UIImage *dis_icon;
    UIImage *chk_icon;
    UIImage *emp_icon;
    uint8_t rawData[100];
}
@property (nonatomic, strong) CaptureManager *captureManager;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
//    [self cleanDocuments];
    afLocked = NO;
    aeLocked = NO;
    self.captureManager = [[CaptureManager alloc] initWithPreviewView:self.preView preferredCameraType:CameraTypeBack
        outputMode:OutputModeVideoData];
    self.captureManager.delegate = self;
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
    [self.btnLeftTop setBackgroundImage:rec_icon forState:UIControlStateNormal];
    [self.btnLeftTop setTintColor:[UIColor greenColor]];
    [self.btnRightTop setBackgroundImage:rec_icon forState:UIControlStateNormal];
    [self.btnRightTop setTintColor:[UIColor greenColor]];
    [self.btnRightBottom setBackgroundImage:rec_icon forState:UIControlStateNormal];
    [self.btnRightBottom setTintColor:[UIColor greenColor]];
    [self.btnLeftBottom setBackgroundImage:rec_icon forState:UIControlStateNormal];
    [self.btnLeftBottom setTintColor:[UIColor greenColor]];
}

- (void) cleanDocuments {
    NSString *documentsDirectory = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0];
    NSArray *filelist = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:documentsDirectory error:nil];
    unsigned long count = [filelist count];
    for(int i = 0; i < count; i++) {
        [[NSFileManager defaultManager] removeItemAtPath:[documentsDirectory stringByAppendingPathComponent:[filelist objectAtIndex:i]] error:nil];
    }
}

//- (void)startRecording {
//    [self.btnRecord setBackgroundImage:stp_icon forState:UIControlStateNormal];
//    [self.btnRecord setTintColor:[UIColor redColor]];
//    NSLog(@"Start Recording");
//}
//
//- (void)finishRecording {
//    [self.btnRecord setBackgroundImage:rec_icon forState:UIControlStateNormal];
//    [self.btnRecord setTintColor:[UIColor greenColor]];
//    NSLog(@"Finish Recording");
//}

- (void) updateScreenCorners: (float*) corners
{
    float width = self.view.frame.size.width;
    float height = self.view.frame.size.height;
    float ratio = width/1920.0;
    float raw_height = 1080 * ratio;
    
    [self.btnLeftTop setCenter:CGPointMake(width*(corners[0] - 960)/960/2.0 + width/2.0, raw_height*(corners[1] - 540)/540/2.0 + height/2.0)];
    [self.btnRightTop setCenter:CGPointMake(width*(corners[2] - 960)/960/2.0 + width/2.0, raw_height*(corners[3] - 540)/540/2.0 + height/2.0)];
    [self.btnRightBottom setCenter:CGPointMake(width*(corners[4] - 960)/960/2.0 + width/2.0, raw_height*(corners[5] - 540)/540/2.0 + height/2.0)];
    [self.btnLeftBottom setCenter:CGPointMake(width*(corners[6] - 960)/960/2.0 + width/2.0, raw_height*(corners[7] - 540)/540/2.0 + height/2.0)];
    NSLog(@"CACACCACACACA: %f, %f, %f, %f, %f, %f, %f, %f\n", self.btnLeftTop.frame.origin.x, self.btnLeftTop.frame.origin.y, self.btnRightTop.frame.origin.x, self.btnRightTop.frame.origin.y, self.btnRightBottom.frame.origin.x, self.btnRightBottom.frame.origin.y, self.btnLeftBottom.frame.origin.x, self.btnLeftBottom.frame.origin.y);
}

- (void)updateResult:(NSString *)text {
    NSString* currentText = self.lbResult.text;
    [self.lbResult setText:[currentText stringByAppendingString:text]];
}

- (void) clearResult
{
    [self.lbResult setText:@""];
}

- (IBAction)btnRecPressed:(UIButton *)sender {
    [self.captureManager startProcessing];
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
    if ([self.captureManager toggleAE]) {
        [self.btnAE setTitle:@"ME" forState:UIControlStateNormal];
        [self.btnAE setTintColor:[UIColor redColor]];
    } else {
        [self.btnAE setTitle:@"AE" forState:UIControlStateNormal];
        [self.btnAE setTintColor:[UIColor greenColor]];
    }
}
- (IBAction)btnAETapped:(id)sender {
    NSLog(@"Start capture N frames\n");
    [self.captureManager stopProcessing];
    [self.lbResult setText:@""];
}

@end
