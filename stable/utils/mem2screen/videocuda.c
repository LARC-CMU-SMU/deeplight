#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libavcodec/avcodec.h"

#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mathematics.h"

#define INBUF_SIZE 4096

AVFormatContext *pFormatCtx;
AVCodecContext *pCodecCtx;
AVCodecParserContext *parser;
const AVCodec *codec;
FILE *fd;
int frame_index;
AVFrame *rgbFrame;
AVPacket *packet;
uint8_t buffer[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];

int open_video(const char *filename) {
    avcodec_register_all();

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged MPEG streams) */
    memset(buffer + INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    /* find the MPEG-1 video decoder */
    codec = avcodec_find_decoder(AV_CODEC_ID_MPEG1VIDEO);
    if (!codec) {
        fprintf(stderr, "codec not found\n");
        exit(1);
    }

    parser = av_parser_init(codec->id);
    if (!parser) {
        fprintf(stderr, "parser not found\n");
        exit(1);
    }

    pCodecCtx = avcodec_alloc_context3(codec);

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */

    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "could not open codec\n");
        exit(1);
    }

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", filename);
        exit(1);
    }

    // Allocate video frame
    packet = av_packet_alloc();
    rgbFrame = av_frame_alloc();
    fd = fopen(filename, "rb");
    if (!fd) {
        fprintf(stderr, "could not open %s\n", filename);
        exit(1);
    }
    return 0;
}

void close_video() {
    fclose(fd);
    av_parser_close(parser);
    avcodec_free_context(&pCodecCtx);
    av_frame_free(&rgbFrame);
    av_packet_free(&packet);
}

static void decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt,
                   const char *filename)
{
    char buf[1024];
    int ret;

    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding\n");
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            exit(1);
        }

        printf("saving frame %3d\n", dec_ctx->frame_number);
        fflush(stdout);

        /* the picture is allocated by the decoder. no need to
           free it */
        snprintf(buf, sizeof(buf), filename, dec_ctx->frame_number);
        pgm_save(frame->data[0], frame->linesize[0],
                 frame->width, frame->height, buf);
    }
}

int read_frame(unsigned char** data, int* width, int* height) {
    while (!feof(fd)) {
        /* read raw data from the input file */
        int data_size = fread(inbuf, 1, INBUF_SIZE, f);
        if (!data_size)
            break;

        /* use the parser to split the data into frames */
        uint8_t *data = buffer;
        while (data_size > 0) {
            int ret = av_parser_parse2(parser, c, &packet->data, &packet->size,
                                   data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (ret < 0) {
                fprintf(stderr, "Error while parsing\n");
                exit(1);
            }
            data      += ret;
            data_size -= ret;

            if (pkt->size) {
                // decode(c, picture, pkt, outfilename);
                int ret = avcodec_send_packet(pCodecCtx, packet);
                if (ret < 0) {
                    fprintf(stderr, "Error sending a packet for decoding\n");
                    exit(1);
                }

                while(ret >= 0) {
                    ret = avcodec_receive_frame(pCodecCtx, rgbFrame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                        return;
                    else if (ret < 0) {
                        fprintf(stderr, "Error during decoding\n");
                        exit(1);
                    }
                }
                return 0;
            }
        }
    }
    return -1;
}