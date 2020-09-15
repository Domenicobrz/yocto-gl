#include <yocto/yocto_sceneio.h>

namespace cpuibl {

using namespace yocto;

inline vec4f getTexel(vector<vec4f>& img, vec2f uv, vec2i size) {
  int tx = uv.x * size.x;
  int ty = (1.0 - uv.y) * size.y;

  tx = clamp(tx, 0, size.x - 1);
  ty = clamp(ty, 0, size.y - 1);

  return img[ty * size.x + tx];
}

inline vec4f getTexelFromDir(vector<vec4f>& img, vec3f dir, vec2i size) {
  vec2f uv = {
      yocto::atan2(dir.z, dir.x) / (2.0f * pi), 1.0 - yocto::acos(dir.y) / pi};

  if (uv.x < 0) uv.x = 1.0 + uv.x;

  int tx = uv.x * size.x;
  int ty = (1.0 - uv.y) * size.y;

  tx = clamp(tx, 0, size.x - 1);
  ty = clamp(ty, 0, size.y - 1);

  return img[ty * size.x + tx];
}

inline vector<vec4f> computeIrradianceTexture(
    vector<vec4f>& envmap, vec2i extent) {
  vector<vec4f> img;

  int size = envmap.size();

  // modify env texture
  for (int i = 0; i < size; i++) {
    int tx = i % extent.x;
    int ty = i / extent.x;

    // uv.y needs to be flipped
    vec2f uv = {float(tx) / float(extent.x), 1.0 - float(ty) / float(extent.y)};

    float dir_theta = (1.0 - uv.y) * pi;
    float dir_phi   = uv.x * pi * 2.0;
    vec3f dir       = {yocto::sin(dir_theta) * yocto::cos(dir_phi),
        yocto::cos(dir_theta), yocto::sin(dir_theta) * yocto::sin(dir_phi)};
    dir             = normalize(dir);

    vec4f vectest = getTexelFromDir(envmap, dir, extent);

    // ***************************
    // ***************************

    vec3f normal     = dir;
    vec4f irradiance = {0, 0, 0, 0};

    vec3f up    = {0.0, 1.0, 0.0};
    vec3f right = normalize(cross(up, normal));
    up          = normalize(cross(normal, right));

    // float sampleDelta = 0.025;
    float sampleDelta = 0.015;
    float nrSamples   = 0.0;
    for (float phi = 0.0; phi < 2.0 * pi; phi += sampleDelta) {
      for (float theta = 0.0; theta < 0.5 * pi; theta += sampleDelta) {
        // spherical to cartesian (in tangent space)
        vec3f tangentSample = {yocto::sin(theta) * yocto::cos(phi),
            yocto::sin(theta) * yocto::sin(phi), yocto::cos(theta)};

        // tangent space to world
        vec3f sampleVec = tangentSample.x * right + tangentSample.y * up +
                          tangentSample.z * normal;

        irradiance += getTexelFromDir(envmap, sampleVec, extent) *
                      yocto::cos(theta) * yocto::sin(theta);
        nrSamples++;
      }
    }
    irradiance = pi * irradiance * (1.0 / float(nrSamples));

    img.push_back({irradiance.x, irradiance.y, irradiance.z, 1});

    // ***************************
    // ***************************
  }

  return img;
}

inline void init_cpu_ibl(trace_scene* scene) {
  auto img    = scene->environments[0]->emission_tex->hdr.data_vector();
  auto extent = scene->environments[0]->emission_tex->hdr.imsize();

  auto irradianceMap = computeIrradianceTexture(img, extent);

  // assign new env texture
  int i = 0;
  for (vec4f* it = scene->environments[0]->emission_tex->hdr.begin();
       it != scene->environments[0]->emission_tex->hdr.end(); it++, i++) {
    scene->environments[0]->emission_tex->hdr[i] = irradianceMap[i];
  }
}
}  // namespace cpuibl
