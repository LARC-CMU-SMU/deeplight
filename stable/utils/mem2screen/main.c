#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <sys/resource.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "gsync.h"
#include "vsync.h"
#include "FreeImage.h"

#include "video.h"

#define MAX_FRAME_RATE 240
#define MIN_FRAME_RATE 1
#define FRAME_RATE_STEP 5
#define BASE_LINE_SEP 20

/**
 * Clock
 */

struct Clock
{
  double currentTimeSec;
  double lastTimeSec;
  double deltaSec; /* Delta Time */
  double aveDelta;
  double lowDelta;
  double highDelta;
  int start;
};

void initializeClock(struct Clock *clock)
{
  clock->currentTimeSec = 0;
  clock->lastTimeSec = 0;
  clock->deltaSec = 0;
  clock->aveDelta = 0;
  clock->lowDelta = 0;
  clock->highDelta = 0;
  clock->start = 100;
}

void updateClock(struct Clock *clock)
{
  clock->lastTimeSec = clock->currentTimeSec;
  clock->currentTimeSec = (double)glutGet(GLUT_ELAPSED_TIME) * 0.001f;
  clock->deltaSec = clock->currentTimeSec - clock->lastTimeSec;
  clock->aveDelta = 0.8*clock->aveDelta + 0.2*clock->deltaSec;
  if(clock->start == 0) {
    if((clock->lowDelta == 0) || (clock->lowDelta > clock->deltaSec)) clock->lowDelta = clock->deltaSec;
    if((clock->highDelta < clock->deltaSec) || (clock->lastTimeSec == 0)) clock->highDelta = clock->deltaSec;
  } else {
    clock->start--;
  }
}

/**
 * Application
 */

struct Application
{
  struct Clock clock;

  struct GSyncController gsyncController;
  struct VSyncController vsyncController;
  FILE *tsfile;
  unsigned int framecount;
  double framerate;
  int imgWidth;
  int imgHeight;
  unsigned char *data;
  bool started;
} app;

void initializeApplication(struct Application *app)
{
  initializeClock(&app->clock);
  vsyncInitialize(&app->vsyncController);
  app->framerate = 30;
  app->framecount = 0;
  app->started = false;
  /* OpenGL initialization */
  glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
}

void toggleGSync(struct Application *app)
{
  gsyncSetAllowed(&app->gsyncController, !gsyncIsAllowed(&app->gsyncController));
}

void toggleVSync(struct Application *app)
{
  vsyncSetEnabled(&app->vsyncController, !vsyncIsEnabled(&app->vsyncController));
}

int printText(int x, int y, float r, float g, float b, const char *format, ...)
{
  char buffer[1024];
  va_list arg;
  int ret;

  /* Format the text */
  va_start(arg, format);
    ret = vsnprintf(buffer, sizeof(buffer), format, arg);
  va_end(arg);

  glColor3f(r, g, b); 
  glRasterPos2f(x, y);
  glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char *)buffer);

  return ret;
}

void drawScene()
{
  // read_frame(&(app.data), &(app.imgWidth), &(app.imgHeight));
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glColor3f(1.0f, 1.0f, 1.0f);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, app.imgWidth, app.imgHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, app.data);
  glBegin(GL_QUADS);
  glNormal3f(0.0, 0.0, 1.0);
  glTexCoord2f(0.0f, 1.0f);glVertex3f(0.0, 0.0, 0.0);
  glTexCoord2f(0.0f, 0.0f);glVertex3f(0.0, glutGet(GLUT_WINDOW_HEIGHT), 0.0);
  glTexCoord2f(1.0f, 0.0f);glVertex3f(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), 0.0);
  glTexCoord2f(1.0f, 1.0f);glVertex3f(glutGet(GLUT_WINDOW_WIDTH), 0.0, 0.0);
  glEnd();
  if(!app.started) {
    printText(10, BASE_LINE_SEP, 0.0f, 1.0f, 0.0f, "Current FPS: %f", 1.0/app.clock.deltaSec);
    printText(10, BASE_LINE_SEP*2, 0.0f, 1.0f, 0.0f, "Low FPS: %f", app.clock.highDelta);
    printText(10, BASE_LINE_SEP*3, 0.0f, 1.0f, 0.0f, "High FPS: %f", app.clock.lowDelta);
    printText(10, BASE_LINE_SEP*4, 0.0f, 1.0f, 0.0f, "Expected FPS: %f", app.framerate);
    if(gsyncIsAllowed(&(app.gsyncController))) printText(10, BASE_LINE_SEP*5, 0.0f, 1.0f, 0.0f, "G-SYNC: YES");
    else printText(10, BASE_LINE_SEP*5, 0.0f, 1.0f, 0.0f, "G-SYNC: NO");
    if(vsyncIsEnabled(&(app.vsyncController))) printText(10, BASE_LINE_SEP*6, 0.0f, 1.0f, 0.0f, "V-SYNC: YES");
    else printText(10, BASE_LINE_SEP*6, 0.0f, 1.0f, 0.0f, "V-SYNC: NO");
  }
  // printf("%f\n", 1.0/app.clock.deltaSec);
}

void loadTextures(const char* filename, unsigned char **data, int *width, int *height) {
  printf("LOADING TEXTURES...\n");
  FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(filename);
  if (format == FIF_UNKNOWN) {
    printf("Unknown file type for texture image file %s\n", filename);
    return;
  }
  FIBITMAP* bitmap = FreeImage_Load(format, filename, 0);
  if (!bitmap) {
    printf("Failed to load image %s\n", filename);
    return;
  }
  FIBITMAP* bitmap2 = FreeImage_ConvertTo24Bits(bitmap);
  FreeImage_Unload(bitmap);
  *data = FreeImage_GetBits(bitmap2);
  *width = FreeImage_GetWidth(bitmap2);
  *height = FreeImage_GetHeight(bitmap2);
  if (*data) {
    printf("Texture image loaded from file %s, size %dx%d\n", filename, *width, *height);
  }
  else {
    printf("Failed to get texture data from %s\n", filename);
  }
  printf("DONE\n\n");
}

/**
 * GLUT's reshape callback function.
 * Update the viewport and the projection matrix.
 */
void reshape(int width, int height)
{
  if (height == 0) {
    height = 1;
  }

  glViewport(0, 0, (GLsizei)width, (GLsizei)height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, width, 0, height, -1.0f, 1.0f);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glutPostRedisplay();
}

/**
 * GLUT's timer callback function.
 * Force redraw of the main OpenGL scene.
 */
void timerCallBack(int value)
{
  updateClock(&app.clock);
  // printf("test: %li\n", clock());
  /* Register next redraw */
  glutTimerFunc((int)(1000/app.framerate), timerCallBack, value);
  /* Redraw scene */
  // if(app.started) read_frame(&(app.data), &(app.imgWidth), &(app.imgHeight));
  glutPostRedisplay();
}

/**
 * GLUT's display callback function.
 * Render the main OpenGL scene.
 */
void display()
{
  glClear(GL_COLOR_BUFFER_BIT);

  drawScene();

  glutSwapBuffers();
  struct timespec timestamp;
  if(app.started) {
    clock_gettime(CLOCK_REALTIME, &timestamp);
    fwrite(&(timestamp.tv_sec), sizeof(timestamp.tv_sec), 1, app.tsfile);
    fwrite(&(timestamp.tv_nsec), sizeof(timestamp.tv_nsec), 1, app.tsfile);
    app.framecount++;
    read_frame(&(app.data), &(app.imgWidth), &(app.imgHeight));
  }
  glutPostRedisplay();
}

/**
 * GLUT's Key press callback function.
 * Called when user press a key.
 */
void keyPress(unsigned char key, int x, int y)
{
  switch (key)
    {
    case 27: /* Escape */
    case 'q': glutLeaveMainLoop(); break;
    case 'v': toggleVSync(&app); break;
    case 'g': toggleGSync(&app); break;
    case ' ': app.started = !app.started; break;
    case 'a': if (app.framerate > 0) app.framerate--; break;
    case 'd': app.framerate++; break;
  }
}

/**
 * GLUT's Key press callback function.
 * Called when user press a special key.
 */
void specialKeyPress(int key, int x, int y)
{
  switch (key) {
  case GLUT_KEY_UP:        
    if(app.framerate <= MAX_FRAME_RATE-FRAME_RATE_STEP) app.framerate += FRAME_RATE_STEP; 
    break;
  case GLUT_KEY_DOWN:      
    if(app.framerate >= MIN_FRAME_RATE-FRAME_RATE_STEP) app.framerate -= FRAME_RATE_STEP;
    break;
  }
}

int main(int argc, char *argv[])
{
  setpriority(PRIO_PROCESS, 0, -20);
  if(open_video(argv[1]) < 0) {
    printf("Error open video file\n");
    return -1;
  }
  if(argc > 2) fill_buffer(atoi(argv[2]));
  else fill_buffer(0);
  app.tsfile = fopen("txtimestamp.ts", "wb");
  fwrite(&(app.framecount), sizeof(app.framecount), 1, app.tsfile);
  gsyncInitialize(&app.gsyncController);
  /* Force G-SYNC Visual Indicator
     For an unknown reason, we must do it twice to make it work...
     (the second call enables the first value) */
  // gsyncShowVisualIndicator(&app.gsyncController, true);
  // gsyncShowVisualIndicator(&app.gsyncController, true);

  /* Initialize GLUT */
  glutInit(&argc, argv);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  /* Create an OpenGL window */
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutEnterGameMode();

  // ===========================================================================
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // (Actually, this one is the default)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glEnable(GL_TEXTURE_2D);
  // ===========================================================================

  glewInit();

  /* Initialize application */
  initializeApplication(&app);
  read_frame(&(app.data), &(app.imgWidth), &(app.imgHeight));
  // loadTextures("test.jpg", &(app.data), &(app.imgWidth), &(app.imgHeight));

  /* Setup GLUT's callback functions */
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutKeyboardFunc(keyPress);
  glutSpecialFunc(specialKeyPress);

  // glutTimerFunc(0, timerCallBack, 0);

  /* Enter the main loop */
  glutMainLoop();

  glutExit();

  gsyncFinalize(&app.gsyncController);
  close_video();
  fseek(app.tsfile, 0, SEEK_SET);
  fwrite(&(app.framecount), sizeof(app.framecount), 1, app.tsfile);
  fflush(app.tsfile);
  fclose(app.tsfile);
  printf("Number of frames: %d\n", app.framecount);
  return 0;
}
