//
//  Processing.m
//  RTDeepLight
//
//  Created by Vu Tran on 14/6/20.
//  Copyright © 2020 Vu Tran. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs/ios.h>
#import <Foundation/Foundation.h>
#import "RTProcessing.h"
#import <CoreML/CoreML.h>
#import "screennet.h"
#import "deeplight.h"
#include "reedsolo.h"

using namespace cv;
using namespace std;

@interface Processing ()
{
    BOOL debug_en;
    VideoCapture *capture;
    CVPixelBufferRef extractorInput;
    CVPixelBufferRef decoderInput;
    Mat temCamera;
    Mat temBGRA;
    Mat extractorBGRA;
    Mat extractorNorm;
    Mat extractorScale;
    Mat extractorGray;
    Mat extractorThreshold;
    Mat extractorDilate;
    Mat extractorKernel;
    vector<cv::Point_<std::float_t>> srcScreenCorners;
    vector<cv::Point_<std::float_t>> dstScreenCorners;
    
    Mat decoderCombined;
    Mat decoderWarped;
    Mat decoderChannels[3][4];
    Mat decoderDummyAlpha;
    int currentIndex;
    
    screennet *extractorModel;
    deeplight *decoderModel;
    
    int seq_num;
    int pre_seq_num;
}
@end

@implementation Processing

- (id) initWithCamera: (int)orgWidth originalHeight:(int)orgHeight extractorWidth:(int)extWidth extractorHeight:(int)extHeight decoderWidth:(int)decWidth decoderHeight:(int)decHeight {
    self = [super init];
    debug_en = YES;
    
    CVReturn code = CVPixelBufferCreate(kCFAllocatorDefault, extWidth, extHeight, kCVPixelFormatType_32BGRA, NULL, &extractorInput);
    NSLog(@"Creating Extractor PixelBuffer. Return code: %id", code);
    CVPixelBufferLockBaseAddress(extractorInput, 0);
    void* extractorInputBaseaddr = CVPixelBufferGetBaseAddress(extractorInput);
    extractorBGRA = Mat(extHeight, extWidth, CV_8UC4, extractorInputBaseaddr);
    CVPixelBufferUnlockBaseAddress(extractorInput, 0);
    extractorNorm = Mat::zeros(extHeight, extWidth, CV_32FC1);
    extractorScale = Mat::zeros(extHeight, extWidth, CV_32FC1);
    extractorGray = Mat::zeros(extHeight, extWidth, CV_8UC1);
    extractorKernel = Mat::ones(3,3,CV_8UC1);
    
    srcScreenCorners.push_back(cv::Point(0, 0));
    srcScreenCorners.push_back(cv::Point(decWidth, 0));
    srcScreenCorners.push_back(cv::Point(decWidth, decHeight));
    srcScreenCorners.push_back(cv::Point(0, decHeight));
    
    dstScreenCorners.push_back(cv::Point(0, 0));
    dstScreenCorners.push_back(cv::Point(decWidth, 0));
    dstScreenCorners.push_back(cv::Point(decWidth, decHeight));
    dstScreenCorners.push_back(cv::Point(0, decHeight));
    
    code = CVPixelBufferCreate(kCFAllocatorDefault, decWidth, decHeight, kCVPixelFormatType_32BGRA, NULL, &decoderInput);
    NSLog(@"Creating Decoder PixelBuffer. Return code: %id, %zu, %zu, %zu", code, CVPixelBufferGetWidth(decoderInput), CVPixelBufferGetHeight(decoderInput), CVPixelBufferGetBytesPerRow(decoderInput));
    CVPixelBufferLockBaseAddress(decoderInput, 0);
    void *decoderInputBaseaddr = CVPixelBufferGetBaseAddress(decoderInput);
    decoderWarped = Mat(304, 304, CV_8UC4, decoderInputBaseaddr);
    CVPixelBufferUnlockBaseAddress(decoderInput, 0);
    decoderCombined = Mat(orgHeight, orgWidth, CV_8UC4);
    NSLog(@"Cac1");
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 4; j++)
            decoderChannels[i][j] = Mat::zeros(orgHeight, orgWidth, CV_8UC1);
    decoderDummyAlpha = Mat(orgHeight, orgWidth, CV_8UC1, Scalar(255));
    currentIndex = 0;
    NSLog(@"Cac2");
    extractorModel = [[screennet alloc] init];
    decoderModel = [[deeplight alloc] init];
    decoderModel.model.configuration.computeUnits = MLComputeUnitsAll;
    extractorModel.model.configuration.computeUnits = MLComputeUnitsAll;
    rscode_init(5, 20, 10);
    seq_num = -1;
    pre_seq_num = -1;
    NSLog(@"Finish initializing Processing");
    return self;
}

- (void) setFrames: (CVPixelBufferRef*)imageBuffer start_index: (uint32_t) offset size: (uint32_t) buffer_size
{
    //======== Image_1
    uint32_t index = offset;
    CVPixelBufferLockBaseAddress(imageBuffer[index], 0);
    int temWidth = (int)CVPixelBufferGetWidth(imageBuffer[index]);
    int temHeight = (int)CVPixelBufferGetHeight(imageBuffer[index]);
    void* temInputBaseaddr = CVPixelBufferGetBaseAddress(imageBuffer[index]);
    temCamera = Mat(temHeight, temWidth, CV_8UC4, temInputBaseaddr);
    cvtColor(temCamera, temBGRA, COLOR_BGR2RGBA);
    CVPixelBufferUnlockBaseAddress(imageBuffer[index], 0);
    split(temBGRA, decoderChannels[0]);
    //======== Image_2
    index = offset + 1;
    CVPixelBufferLockBaseAddress(imageBuffer[index], 0);
    temInputBaseaddr = CVPixelBufferGetBaseAddress(imageBuffer[index]);
    temBGRA = Mat(temHeight, temWidth, CV_8UC4, temInputBaseaddr);
    CVPixelBufferUnlockBaseAddress(imageBuffer[index], 0);
    split(temBGRA, decoderChannels[1]);
    //======== Current frame
    index = offset + 2;
    CVPixelBufferLockBaseAddress(imageBuffer[index], 0);
    temInputBaseaddr = CVPixelBufferGetBaseAddress(imageBuffer[index]);
    temBGRA = Mat(temHeight, temWidth, CV_8UC4, temInputBaseaddr);
    CVPixelBufferUnlockBaseAddress(imageBuffer[index], 0);
    split(temBGRA, decoderChannels[2]);
//    if (debug_en) NSLog(@"Current Index: %d\n", currentIndex);
}

vector<float> intersect(float a1, float b1, float c1, float a2, float b2, float c2)
{
    vector<float> result(2);
    if(a1*b2-a2*b1 == 0) {
        result[0] = -1;
        result[1] = -1;
    } else {
        result[0] = (c2*b1-c1*b2)/(a1*b2-a2*b1);
        result[1] = (c2*a1-c1*a2)/(b1*a2-b2*a1);
    }
    return result;
}

vector<float> fitting(vector<cv::Point> points, int direction, float dist, int size, int step)
{
//A RANSAC inspired (Without ramdom sample as the contour points are sequential) fitting algorithm to find the border.
    unsigned long max_fit = 0;
    vector<float> max_arg(3);
    unsigned long head_index = 0;
    while(head_index < points.size()) {
        vector<cv::Point> tem_points(size);
        if(head_index + size > points.size()) {
            unsigned long first_size = points.size()-head_index;
            unsigned long second_size = size - first_size;
            memcpy(&tem_points[0], &points[head_index],first_size*sizeof(cv::Point));
            memcpy(&tem_points[first_size], &points[0], second_size*sizeof(cv::Point));
        } else {
            memcpy(&tem_points[0], &points[head_index],size*sizeof(cv::Point));
        }
        vector<float> line(4);
        cv::fitLine(tem_points, line, cv::DIST_L2, 0, 0.01, 0.01);
        float a = line[1];
        float b = -line[0];
        float c = line[0]*line[3]-line[1]*line[2];
        head_index += step;
        if(direction == 0){
            if((b != 0)&&(abs(a/b) <= 1.0)) continue; //direction = 0 for vertical
        }
        else {
            if((a != 0)&&(abs(b/a) <= 1.0)) continue; //direction = 1 for horizontal
        }
        vector<cv::Point> inliners;
        float norm = a*a + b*b;
        float s = dist*dist;
        for(int i = 0; i < points.size(); i++) {
            float tem1 = a*points[i].x + b*points[i].y + c;
            if(tem1*tem1/norm < s) inliners.push_back(points[i]);
        }
        if(inliners.size() > max_fit) {
            max_fit = inliners.size();
            cv::fitLine(inliners, line, cv::DIST_L2, 0, 0.01, 0.01);
            float a = line[1];
            float b = -line[0];
            float c = line[0]*line[3]-line[1]*line[2];
            max_arg[0] = a;
            max_arg[1] = b;
            max_arg[2] = c;
        }
    }
    return max_arg;
}

-(void) detectScreen: (CVPixelBufferRef) frame
{
    int temWidth = (int)CVPixelBufferGetWidth(frame);
    int temHeight = (int)CVPixelBufferGetHeight(frame);
    CVPixelBufferLockBaseAddress(frame, 0);
    void* temInputBaseaddr = CVPixelBufferGetBaseAddress(frame);
    temCamera = Mat(temHeight, temWidth, CV_8UC4, temInputBaseaddr);
    cvtColor(temCamera, temBGRA, COLOR_BGR2RGBA);
    CVPixelBufferUnlockBaseAddress(frame, 0);
    resize(temBGRA, extractorBGRA, cv::Size(extractorBGRA.cols, extractorBGRA.rows));
    screennetOutput *output = [extractorModel predictionFromInput:extractorInput error:nil];
    
    int srcWidth = extractorBGRA.cols;
    int srcHeight = extractorBGRA.rows;

    void* rawdata = output.output.dataPointer;
    Mat rawExtract(srcHeight, srcWidth, CV_32FC1, rawdata, 0);
    scaleAdd(rawExtract, 255, extractorNorm, extractorScale);
    extractorScale.convertTo(extractorGray, CV_8UC1);
    
    threshold(extractorGray, extractorThreshold, 180, 255, THRESH_BINARY);
    vector<vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(extractorThreshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSString *greyFilePath = [documentsDirectory stringByAppendingPathComponent:@"/grey.jpg"];
    NSString *maskFilePath = [documentsDirectory stringByAppendingPathComponent:@"/mask.jpg"];
    imwrite(maskFilePath.UTF8String, extractorThreshold);
    imwrite(greyFilePath.UTF8String, extractorGray);
    float max_area = -1;
    int max_index = -1;
    for(int i = 0; i < contours.size(); i++) {
        float tem_area = contourArea(contours[i]);
        if(tem_area > max_area) {
            max_area = tem_area;
            max_index = i;
        }
    }
    if(max_index >= 0) {
        vector<cv::Point> left_points;
        vector<cv::Point> top_points;
        vector<cv::Point> right_points;
        vector<cv::Point> bottom_points;
        vector<cv::Point> max_contour = contours[max_index];
        Moments mm = moments(max_contour);
        float px = mm.m10/mm.m00;
        float py = mm.m01/mm.m00;
        
        int pre_x = max_contour[max_contour.size()-1].x;
        int pre_y = max_contour[max_contour.size()-1].y;
        for(int i = 0; i < max_contour.size(); i++) {
            int tem_x = max_contour[i].x;
            int tem_y = max_contour[i].y;
            if(pre_x == tem_x) {
                int x = pre_x;
                int y = pre_y;
                int dy = 1;
                if(pre_y > tem_y) dy = -1;
                while(y != tem_y) {
                    if(x < px) left_points.push_back(cv::Point(x, y));
                    if(x > px) right_points.push_back(cv::Point(x, y));
                    if(y < py) top_points.push_back(cv::Point(x, y));
                    if(y > py) bottom_points.push_back(cv::Point(x, y));
                    y = y + dy;
                }
            } else if(pre_y == tem_y) {
                int x = pre_x;
                int y = pre_y;
                int dx = 1;
                if(pre_x > tem_x) dx = -1;
                while(x != tem_x) {
                    if(x < px) left_points.push_back(cv::Point(x, y));
                    if(x > px) right_points.push_back(cv::Point(x, y));
                    if(y < py) top_points.push_back(cv::Point(x, y));
                    if(y > py) bottom_points.push_back(cv::Point(x, y));
                    x = x + dx;
                }
            } else { //dx == dy
                int x = pre_x;
                int y = pre_y;
                int dx = 1;
                int dy = 1;
                if(pre_x > tem_x) dx = -1;
                if(pre_y > tem_y) dy = -1;
                while(x != tem_x) {
                    if(x < px) left_points.push_back(cv::Point(x, y));
                    if(x > px) right_points.push_back(cv::Point(x, y));
                    if(y < py) top_points.push_back(cv::Point(x, y));
                    if(y > py) bottom_points.push_back(cv::Point(x, y));
                    x = x + dx;
                    y = y + dy;
                }
            }
            pre_x = tem_x;
            pre_y = tem_y;
        }
        
        vector<float> left = fitting(left_points, 0, 4, 40, 10);
        vector<float> right = fitting(right_points, 0, 4, 40, 10);
        vector<float> top = fitting(top_points, 1, 4, 40, 10);
        vector<float> bottom = fitting(bottom_points, 1, 4, 40, 10);
        NSLog(@"==========LEFT========== %f, %f, %f", left[0], left[1], left[2]);
        NSLog(@"==========RIGHT========== %f, %f, %f", right[0], right[1], right[2]);
        NSLog(@"==========TOP========== %f, %f, %f", top[0], top[1], top[2]);
        NSLog(@"==========BOTTOM========== %f, %f, %f", bottom[0], bottom[1], bottom[2]);
        vector<float> lt = intersect(left[0], left[1], left[2], top[0], top[1], top[2]);
        vector<float> rt = intersect(right[0], right[1], right[2], top[0], top[1], top[2]);
        vector<float> rb = intersect(right[0], right[1], right[2], bottom[0], bottom[1], bottom[2]);
        vector<float> lb = intersect(left[0], left[1], left[2], bottom[0], bottom[1], bottom[2]);
        srcScreenCorners[0].x = lt[0]*temBGRA.cols/extractorBGRA.cols;
        srcScreenCorners[0].y = lt[1]*temBGRA.rows/extractorBGRA.rows;
        srcScreenCorners[1].x = rt[0]*temBGRA.cols/extractorBGRA.cols;
        srcScreenCorners[1].y = rt[1]*temBGRA.rows/extractorBGRA.rows;
        srcScreenCorners[2].x = rb[0]*temBGRA.cols/extractorBGRA.cols;
        srcScreenCorners[2].y = rb[1]*temBGRA.rows/extractorBGRA.rows;
        srcScreenCorners[3].x = lb[0]*temBGRA.cols/extractorBGRA.cols;
        srcScreenCorners[3].y = lb[1]*temBGRA.rows/extractorBGRA.rows;
        if(debug_en) NSLog(@"=================== Corners: %f, %f : %f, %f : %f, %f : %f, %f\n", srcScreenCorners[0].x, srcScreenCorners[0].y, srcScreenCorners[1].x, srcScreenCorners[1].y, srcScreenCorners[2].x, srcScreenCorners[2].y, srcScreenCorners[3].x, srcScreenCorners[3].y);
    }
}

-(void) extractScreen
{
    Mat transform = getPerspectiveTransform(srcScreenCorners, dstScreenCorners);
    vector<Mat> blueChannels;
    blueChannels.push_back(decoderChannels[2][0]);
    blueChannels.push_back(decoderChannels[1][0]);
    blueChannels.push_back(decoderChannels[0][0]);
    blueChannels.push_back(decoderDummyAlpha);
    merge(blueChannels, decoderCombined);
    warpPerspective(decoderCombined, decoderWarped, transform, cv::Size(decoderWarped.cols, decoderWarped.rows));
}

-(void) decode: (int*) data
{
    deeplightOutput *output = [decoderModel predictionFromInput:decoderInput error:nil];
    float32_t *rawData = (float32_t*)output.output.dataPointer;
    
    for(int s = 0; s < 100; s++) {
        if(rawData[s] >= 0.5) data[s] = 1;
        else data[s] = 0;
    }
}

-(NSString*) correct:(int *)data
{
    int sym_bits = 5;
    int block_size = 20;
    int ecc_size = 10;
    int temdata[50];
    int size;
    int code = rscode_decode_bits(data, 100, temdata, &size);
    int data_size_bits = (block_size - ecc_size)*sym_bits;
    if(code < 0) {
        return @"";
    } else {
        NSString *result = @"";
        int seq_num = 0;
        for(int bit_index = 6; bit_index < 11; bit_index++) {
            seq_num = seq_num << 1;
            seq_num = seq_num + temdata[data_size_bits - bit_index];
        }
        
        int csum_field = 0;
        for(int bit_index = 1; bit_index < 6; bit_index++) {
            csum_field = csum_field << 1;
            csum_field = csum_field + temdata[data_size_bits-bit_index];
        }

        int csum_calc = 0;
        for(int sym_index = 0; sym_index < block_size - ecc_size - 1; sym_index++) {
            int value = 0;
            for(int bit_index = sym_index * sym_bits; bit_index < sym_index * sym_bits + sym_bits; bit_index++) {
                value = value << 1;
                value += temdata[bit_index];//[sym_index*sym_bits+sym_bits - bit_index];
            }
            csum_calc = ((csum_calc & 0x1F) >> 1) + ((csum_calc & 0x1) << 4);
            csum_calc = (csum_calc + value) & 0x1F;
        }
        if(csum_calc == csum_field) {
            for(int text_index = 0; text_index < 5; text_index++) {
                int sym_bits_upper = text_index*8+7;
                int value = 0;
                for(int bit_index = 0; bit_index < 8; bit_index++) {
                    value = value << 1;
                    value += temdata[sym_bits_upper - bit_index];
                }
                if(seq_num != pre_seq_num) result = [result stringByAppendingFormat:@"%c", value];
            }
            pre_seq_num = seq_num;
        } else {
            return @"";
        }
        return result;
    }
}

- (void) getScreenCorners: (float *) corners
{
    corners[0] = srcScreenCorners[0].x;
    corners[1] = srcScreenCorners[0].y;
    corners[2] = srcScreenCorners[1].x;
    corners[3] = srcScreenCorners[1].y;
    corners[4] = srcScreenCorners[2].x;
    corners[5] = srcScreenCorners[2].y;
    corners[6] = srcScreenCorners[3].x;
    corners[7] = srcScreenCorners[3].y;
}

@end
