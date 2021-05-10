#ifndef __VIDEO_H__
#define __VIDEO_H__

int open_video(const char* filename);
void close_video();
int read_frame(unsigned char** data, int* width, int* height);
void fill_buffer();

#endif /* __VIDEO_H__ */
