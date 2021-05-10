#include "video.h"
#include <pthread.h>
#include <semaphore.h> 
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#define BUFLEN 1000
uint8_t* buffer[BUFLEN];
int rdptr = 0;
// int wrptr = -1;
AVFormatContext *pFormatCtx = NULL;
AVCodecContext *pCodecCtx = NULL;
struct SwsContext *sws_ctx = NULL;
int videoStream;
int frameFinished;
int frame_width = 0;
int frame_height = 0;
AVFrame *rawFrame = NULL;
AVFrame *rgbFrame = NULL;
AVPacket packet;
// uint8_t * buffer;

int open_video(const char *filename) {
    av_register_all();

    // Open video file
    if(avformat_open_input(&pFormatCtx, filename, NULL, NULL)!=0){
      printf("Cannot open input\n");
      return -1; // Couldn't open file
    }
    // Retrieve stream information
    if(avformat_find_stream_info(pFormatCtx, NULL)<0) {
      printf("Stream info not found\n");
      return -1; // Couldn't find stream information
    }

    // Find the first video stream
    videoStream=-1;
    for(int i=0; i<pFormatCtx->nb_streams; i++)
      if(pFormatCtx->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_VIDEO) {
        videoStream=i;
        break;
      }
    if(videoStream==-1) {
      printf("Video stream not found\n");
      return -1; // Didn't find a video stream
    }
    frame_width = pFormatCtx->streams[videoStream]->codecpar->width;
    frame_height = pFormatCtx->streams[videoStream]->codecpar->height;

    // Get a pointer to the codec context for the video stream
    AVCodec *pCodec=avcodec_find_decoder(pFormatCtx->streams[videoStream]->codecpar->codec_id);
    if(pCodec==NULL) {
      fprintf(stderr, "Unsupported codec!\n");
      return -1; // Codec not found
    }
    // Copy context
    pCodecCtx = avcodec_alloc_context3(pCodec);
    
    // Open codec
    if(avcodec_open2(pCodecCtx, pCodec, NULL)<0) {
      printf("Cannot open codec\n");
      return -1; // Could not open codec
    }

    // Allocate video frame
    rawFrame = av_frame_alloc();
    rgbFrame = av_frame_alloc();
    for(int i = 0; i < BUFLEN; i++)
      buffer[i] = (uint8_t *)av_malloc(av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame_width, frame_height, 16));
    rgbFrame->width = frame_width;
    rgbFrame->height = frame_height;
    rgbFrame->format = AV_PIX_FMT_BGR24;
    rgbFrame->linesize[0] = rgbFrame->width *3;
    return 0;
}

void close_video() {
  av_free(rawFrame);
  av_free(rgbFrame);
  for(int i = 0; i < BUFLEN; i++) free(buffer[i]);
  avcodec_close(pCodecCtx);
  avformat_close_input(&pFormatCtx);
}

int read_frame(unsigned char** data, int* width, int* height) {
  *data = buffer[rdptr];
  rdptr = (rdptr + 1) % BUFLEN;
  *width = frame_width;
  *height = frame_height;
  return 0;
}

void fill_buffer(int start) 
{ 
  int wrptr = 0;
  int frame_no = 0;
  while((wrptr < BUFLEN) && (av_read_frame(pFormatCtx, &packet)>=0)) {
    // Is this a packet from the video stream?
    if(packet.stream_index==videoStream) {
      // Decode video frame
      avcodec_decode_video2(pCodecCtx, rawFrame, &frameFinished, &packet);
      // Did we get a video frame?
      if(frameFinished) {
        if(frame_no >= start) {
          sws_ctx = sws_getContext(rawFrame->width, rawFrame->height,
          pCodecCtx->pix_fmt,
          rgbFrame->width, rgbFrame->height,
          AV_PIX_FMT_RGB24,
          SWS_BILINEAR,
          NULL,
          NULL,
          NULL
          );
          rgbFrame->data[0] = buffer[wrptr];
          sws_scale(sws_ctx, (uint8_t const * const *)rawFrame->data, rawFrame->linesize, 0, rawFrame->height, rgbFrame->data, rgbFrame->linesize);
          wrptr++;
          if(wrptr >= BUFLEN) break;
        }
        frame_no++;
      }
    }
    // Free the packet that was allocated by av_read_frame
    av_packet_unref(&packet);
  }
  return NULL; 
} 
