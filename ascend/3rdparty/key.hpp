#ifndef __KEY_H_
#define __KEY_H_


#include "multi_button.hpp"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "fcntl.h"
#include "unistd.h"
#include "led.hpp"
#include "pthread.h"

namespace thirdparty {

typedef enum {
  S2,
  S3
} KeyNum;

typedef struct TKeyhandlerTab {
  int fd;
} Keyhandler;

void Key_Init(KeyNum gpioNum, Keyhandler *key);
char Key_Status(Keyhandler *key);
void Key_close(Keyhandler *key);

}

#endif //__KEY_H_