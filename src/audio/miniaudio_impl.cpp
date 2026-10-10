// Single translation unit that instantiates miniaudio (audio capture + codecs).
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#include "miniaudio.h"
