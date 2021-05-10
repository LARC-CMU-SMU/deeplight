//
//  ViewController.h
//  Recorder
//
//  Created by Vu Tran on 24/1/20.
//  Copyright © 2020 Vu Tran. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController

@property (strong, nonatomic) IBOutlet UIView *mainView;
@property (strong, nonatomic) IBOutlet UIView *preView;
@property (strong, nonatomic) IBOutlet UIButton *btnRecord;
@property (strong, nonatomic) IBOutlet UISegmentedControl *segFps;
@property (strong, nonatomic) IBOutlet UIButton *btnAF;
@property (strong, nonatomic) IBOutlet UIButton *btnAE;
@property (strong, nonatomic) IBOutlet UIActivityIndicatorView *savingProgress;
@property (strong, nonatomic) IBOutlet UILabel *lbResult;
@property (strong, nonatomic) IBOutlet UIButton *btnLeftTop;
@property (strong, nonatomic) IBOutlet UIButton *btnRightTop;
@property (strong, nonatomic) IBOutlet UIButton *btnRightBottom;
@property (strong, nonatomic) IBOutlet UIButton *btnLeftBottom;

@end

