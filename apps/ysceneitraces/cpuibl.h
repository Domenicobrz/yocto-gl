#include <yocto/yocto_sceneio.h>

namespace cpuibl {

using namespace yocto;

inline vec4f getTexel(image<vec4f> img, vec2f uv) {
  auto size = img.imsize();

  int tx = uv.x * size.x;
  int ty = (1.0 - uv.y) * size.y;

  tx = clamp(tx, 0, size.x - 1);
  ty = clamp(ty, 0, size.y - 1);

  return img[ty * size.x + tx];
}

inline void init_cpu_ibl(trace_scene* scene) {
  auto img    = scene->environments[0]->emission_tex->hdr.data_vector();
  auto extent = scene->environments[0]->emission_tex->hdr.imsize();

  // modify env texture
  for (int i = 0; i < img.size(); i++) {
    int tx = i % extent.x;
    int ty = i / extent.y;

    // uv.y needs to be flipped
    vec2f uv = {float(tx) / float(extent.x), 1.0 - float(ty) / float(extent.y)};
    vec4f vectest = {uv.x, uv.y, 0, 1};

    img[i] *= vectest;
  }

  // assign new env texture
  int i = 0;
  for (vec4f* it = scene->environments[0]->emission_tex->hdr.begin();
       it != scene->environments[0]->emission_tex->hdr.end(); it++, i++) {
    scene->environments[0]->emission_tex->hdr[i] = img[i];
  }
}
}  // namespace cpuibl
