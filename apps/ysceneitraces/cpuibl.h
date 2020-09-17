#include <yocto/yocto_parallel.h>
#include <yocto/yocto_sceneio.h>

namespace cpuibl {

using namespace yocto;

float radical_inverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10;  // / 0x100000000
}

inline void parallel_for(int size, const std::function<void(int)>& f) {
  auto threads = vector<std::thread>(size);
  for (int i = 0; i < size; i++) {
    threads[i] = std::thread(f, i);
  }
  for (int i = 0; i < size; i++) {
    threads[i].join();
  }
}

inline void serial_for(int size, const std::function<void(int)>& f) {
  for (int i = 0; i < size; i++) {
    f(i);
  }
}

vec2f hammersley(uint i, int N) {
  return vec2f{float(i) / float(N), radical_inverse(i)};
}

vec3f sample_ggx(vec2f rn, vec3f N, float roughness) {
  float a = roughness * roughness;

  float phi       = 2.0f * pif * rn.x;
  float cos_theta = yocto::sqrt((1.0f - rn.y) / (1.0f + (a * a - 1.0f) * rn.y));
  float sin_theta = yocto::sqrt(1.0f - cos_theta * cos_theta);

  // from spherical coordinates to cartesian coordinates
  vec3f H;
  H.x = yocto::cos(phi) * sin_theta;
  H.y = yocto::sin(phi) * sin_theta;
  H.z = cos_theta;

  // from tangent-space vector to world-space sample vector
  vec3f up = yocto::abs(N.z) < 0.999 ? vec3f{0.0f, 0.0f, 1.0f}
                                     : vec3f{1.0f, 0.0f, 0.0f};
  vec3f tangent   = normalize(cross(up, N));
  vec3f bitangent = normalize(cross(N, tangent));

  vec3f result = tangent * H.x + bitangent * H.y + N * H.z;
  return normalize(result);
}

inline vec4f getTexel(vector<vec4f>& img, vec2f uv, vec2i size) {
  int tx = uv.x * size.x;
  int ty = (1.0f - uv.y) * size.y;

  tx = clamp(tx, 0, size.x - 1);
  ty = clamp(ty, 0, size.y - 1);

  return img[ty * size.x + tx];
}

inline vec4f getTexelFromDir(
    const vector<vec4f>& img, const vec3f& dir, const vec2i& size) {
  vec2f uv = {yocto::atan2(dir.z, dir.x) / (2.0f * pif),
      1.0f - yocto::acos(dir.y) / pif};

  if (uv.x < 0) uv.x = 1.0f + uv.x;

  int tx = uv.x * size.x;
  int ty = (1.0f - uv.y) * size.y;

  tx = clamp(tx, 0, size.x - 1);
  ty = clamp(ty, 0, size.y - 1);

  return img[ty * size.x + tx];
}

inline vector<vec4f> computeIrradianceTexture(
    const vector<vec4f>& envmap, const vec2i& extent) {
  auto size = envmap.size();
  auto img  = vector<vec4f>(size);

  // modify env texture
  // for (int i = 0; i < size; i++) {
  auto f = [&](int i) {
    int tx = i % extent.x;
    int ty = i / extent.x;
    printf("%d, %d\n", tx, ty);

    // uv.y needs to be flipped
    vec2f uv = {
        float(tx) / float(extent.x), 1.0f - float(ty) / float(extent.y)};

    float dir_theta = (1.0f - uv.y) * pif;
    float dir_phi   = uv.x * pif * 2.0;
    vec3f dir       = {yocto::sin(dir_theta) * yocto::cos(dir_phi),
        yocto::cos(dir_theta), yocto::sin(dir_theta) * yocto::sin(dir_phi)};
    dir             = normalize(dir);

    vec4f vectest = getTexelFromDir(envmap, dir, extent);

    // ***************************
    // ***************************

    vec3f normal     = dir;
    vec4f irradiance = {0, 0, 0, 0};

    vec3f up    = {0.0, 1.0f, 0.0};
    vec3f right = normalize(cross(up, normal));
    up          = normalize(cross(normal, right));

    // float sampleDelta = 0.025;
    float sampleDelta = 0.015;
    float nrSamples   = 0.0;
    for (float phi = 0.0; phi < 2.0 * pif; phi += sampleDelta) {
      for (float theta = 0.0; theta < 0.5 * pif; theta += sampleDelta) {
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
    irradiance = pif * irradiance * (1.0f / float(nrSamples));

    img[i] = {irradiance.x, irradiance.y, irradiance.z, 1};

    // ***************************
    // ***************************
  };

  parallel_for(size, f);
  // serial_for(size, f);

  return img;
}

inline vector<vector<vec4f>> computePrefilteredTextures(
    vector<vec4f>& envmap, vec2i extent, int levels) {
  vector<vector<vec4f>> imgs;

  int size        = envmap.size();
  int num_samples = 512;  // 1024;

  for (int l = 0; l < levels; l++) {
    imgs.push_back(vector<vec4f>());

    vector<vec4f>& img = imgs[l];

    float roughness = float(l) / float(levels - 1);

    // modify env texture
    for (int i = 0; i < size; i++) {
      int tx = i % extent.x;
      int ty = i / extent.x;

      // uv.y needs to be flipped
      vec2f uv = {
          float(tx) / float(extent.x), 1.0f - float(ty) / float(extent.y)};

      float dir_theta = (1.0f - uv.y) * pif;
      float dir_phi   = uv.x * pif * 2.0;
      vec3f dir       = {yocto::sin(dir_theta) * yocto::cos(dir_phi),
          yocto::cos(dir_theta), yocto::sin(dir_theta) * yocto::sin(dir_phi)};
      dir             = normalize(dir);

      vec3f N = normalize(dir);
      vec3f R = N;
      vec3f V = N;

      float total_weight = 0.0f;
      vec4f result       = vec4f{0, 0, 0, 0};

      for (uint i = 0u; i < uint(num_samples); i++) {
        vec2f rn = hammersley(i, num_samples);
        vec3f H  = sample_ggx(rn, N, roughness);
        vec3f L  = normalize(
            reflect(V, H));  // reflect in yocto-gl returns the opposite
                              // direction of reflect glsl
        float NdotL = dot(N, L);
        if (NdotL > 0.0f) {
          result += getTexelFromDir(envmap, L, extent) * NdotL;
          total_weight += NdotL;
        }
      }
      result = result / total_weight;

      img.push_back({result.x, result.y, result.z, 1.0f});
    }
  }

  return imgs;
}

inline void init_cpu_ibl(trace_scene* scene) {
  auto& img    = scene->environments[0]->emission_tex->hdr.data_vector();
  auto  extent = scene->environments[0]->emission_tex->hdr.imsize();

  auto irradianceMap = computeIrradianceTexture(img, extent);
  // auto prefilteredMaps = computePrefilteredTextures(img, extent, 5);
  scene->environments[0]->emission_tex->hdr.data_vector() = irradianceMap;

  // assign new env texture
  //  int i = 0;
  //  for (vec4f* it = scene->environments[0]->emission_tex->hdr.begin();
  //       it != scene->environments[0]->emission_tex->hdr.end(); it++, i++) {
  // scene->environments[0]->emission_tex->hdr[i] = irradianceMap[i];
  //    scene->environments[0]->emission_tex->hdr[i] = prefilteredMaps[2][i];
  //  }
}
}  // namespace cpuibl
