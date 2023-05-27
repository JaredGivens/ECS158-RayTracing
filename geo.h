#ifndef DRYF_GEO_H
#define DRYF_GEO_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int32_t const kTsfSize, kTriSize, kSegSize;

typedef union {
  struct {
    float x, y, z;
  };
  float floats[3];
} vecf_t;

typedef union {
  struct {
    int32_t x, y, z;
  };
  int32_t ints[3];
} veci_t;

typedef union {
  struct {
    float x, y, z, w;
  };
  float floats[4];
} quat_t;

typedef struct {
  vecf_t pos;
  quat_t rot;
} tsf_t;

typedef union {
  struct {
    vecf_t a, b, c;
  };
  float floats[9];
  vecf_t vecs[3];
} tri_t;

#ifdef __cplusplus
}
#endif

#endif