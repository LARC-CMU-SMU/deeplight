CC = gcc
LD = $(CC)
CFLAGS += -Wall -O3 -std=gnu99
LDFLAGS += -L/usr/local/cuda/lib64 -L/usr/lib/nvidia-440 -lXNVCtrl -lX11 -lGL -lGLU -lglut -lGLEW -lavutil -lavformat -lavcodec -lavutil -lswscale -lswresample -lcuda -lnvcuvid -lpthread -lfreeimage -lm

TARGETS = mem2screen

.PHONY: default
default: $(TARGETS)

.PHONY: clean
clean:
	-rm -rf *.o core.* *~ $(TARGETS)

mem2screen: main.o gsync.o vsync.o video.o
	$(LD) $^ $(LDFLAGS) -o $@

main.o: main.c gsync.h vsync.h
gsync.o: gsync.c gsync.h
vsync.o: vsync.c vsync.h
video.o: video.c video.h
