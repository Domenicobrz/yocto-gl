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

float GeometrySchlickGGX(float NdotV, float roughness) {
  float a = roughness;
  float k = (a * a) / 2.0f;

  float nom   = NdotV;
  float denom = NdotV * (1.0f - k) + k;

  return nom / denom;
}

float GeometrySmith(vec3f N, vec3f V, vec3f L, float roughness) {
  float NdotV = max(dot(N, V), 0.0f);
  float NdotL = max(dot(N, L), 0.0f);
  float ggx2  = GeometrySchlickGGX(NdotV, roughness);
  float ggx1  = GeometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
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

inline void parallel_for(
    int num_threads, int size, const std::function<void(int)>& f) {
  auto threads = vector<std::thread>(num_threads);
  auto batch   = [&](int k) {
    int from = k * (size / num_threads);
    int to   = min(from + (size / num_threads), size);
    for (int i = from; i < to; i++) f(i);
  };

  for (int k = 0; k < num_threads; k++) {
    threads[k] = std::thread(batch, k);
  }
  for (int k = 0; k < num_threads; k++) {
    threads[k].join();
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

vec3f sample_ggx(const vec2f& rn, const vec3f& N, float roughness) {
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

inline vec4f getTexel(
    const vector<vec4f>& img, const vec2f& uv, const vec2i& size) {
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

vec2i                ir_extent = vec2i{256, 128};
inline vector<vec4f> computeIrradianceTexture(
    const vector<vec4f>& envmap, const vec2i& extent) {
  int  ir_size = ir_extent.x * ir_extent.y;
  auto img     = vector<vec4f>(ir_size);

  // modify env texture
  auto f = [&](int i) {
    int tx = i % (ir_extent.x);
    int ty = i / (ir_extent.x);

    // uv.y needs to be flipped
    vec2f uv = {
        float(tx) / float(ir_extent.x), 1.0f - float(ty) / float(ir_extent.y)};

    float dir_theta = (1.0f - uv.y) * pif;
    float dir_phi   = uv.x * pif * 2.0;
    vec3f dir       = {yocto::sin(dir_theta) * yocto::cos(dir_phi),
        yocto::cos(dir_theta), yocto::sin(dir_theta) * yocto::sin(dir_phi)};
    dir             = normalize(dir);

    vec4f vectest = getTexelFromDir(envmap, dir, extent);

    vec3f normal     = dir;
    vec4f irradiance = {0, 0, 0, 0};

    vec3f up    = {0.0, 1.0f, 0.0};
    vec3f right = normalize(cross(up, normal));
    up          = normalize(cross(normal, right));

    float sampleDelta = 0.025;
    // float sampleDelta = 0.00625;
    float nrSamples = 0.0;
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
  };

  parallel_for(16, ir_size, f);

  return img;
}

vec2i                        prefiltered_maps_extent = vec2i{256, 128};
int                          prefiltered_maps_levels = 5;
inline vector<vector<vec4f>> computePrefilteredTextures(
    const vector<vec4f>& envmap, const vec2i& extent, int levels) {
  vector<vector<vec4f>> imgs;

  int size        = prefiltered_maps_extent.x * prefiltered_maps_extent.y;
  int num_samples = 1024;  // 300024;

  for (int l = 0; l < levels; l++) {
    imgs.push_back(vector<vec4f>(size));

    vector<vec4f>& img = imgs[l];

    float roughness = float(l) / float(levels - 1);
    roughness       = roughness * roughness;

    // modify env texture
    auto f = [&](int i) {
      int tx = i % prefiltered_maps_extent.x;
      int ty = i / prefiltered_maps_extent.x;

      // uv.y needs to be flipped
      vec2f uv = {float(tx) / float(prefiltered_maps_extent.x),
          1.0f - float(ty) / float(prefiltered_maps_extent.y)};

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
          // speed-up hack that reuses previously computed levels
          // result += getTexelFromDir(l == 0 ? envmap : imgs[l - 1], L, extent)
          result += getTexelFromDir(envmap, L, extent) * NdotL;
          total_weight += NdotL;
        }
      }
      result = result / total_weight;

      img[i] = {result.x, result.y, result.z, 1.0f};
    };

    parallel_for(16, size, f);
  }

  return imgs;
}

vec2i                brdf_extent = vec2i{256, 256};
inline vector<vec4f> computeBRDFLUT() {
  int  size = brdf_extent.x * brdf_extent.y;
  auto img  = vector<vec4f>(size);

  // modify env texture
  auto f = [&](int i) {
    int tx = i % brdf_extent.x;
    int ty = i / brdf_extent.x;

    // uv.y needs to be flipped
    vec2f uv = {float(tx) / float(brdf_extent.x),
        1.0f - float(ty) / float(brdf_extent.y)};

    float NdotV     = uv.x;
    float roughness = uv.y;

    vec3f V;
    V.x = yocto::sqrt(1.0f - NdotV * NdotV);
    V.y = 0.0f;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    vec3f N = vec3f{0.0f, 0.0f, 1.0f};

    const uint SAMPLE_COUNT = 1024;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
      vec2f Xi = hammersley(i, SAMPLE_COUNT);
      vec3f H  = sample_ggx(Xi, N, roughness);
      vec3f L  = normalize(2.0f * dot(V, H) * H - V);

      float NdotL = max(L.z, 0.0);
      float NdotH = max(H.z, 0.0);
      float VdotH = max(dot(V, H), 0.0);

      if (NdotL > 0.0) {
        float G     = GeometrySmith(N, V, L, roughness);
        float G_Vis = (G * VdotH) / (NdotH * NdotV);
        float Fc    = pow(1.0 - VdotH, 5.0);

        A += (1.0 - Fc) * G_Vis;
        B += Fc * G_Vis;
      }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);

    img[i] = {A, B, 0, 1};
  };

  parallel_for(16, size, f);

  return img;
}

struct trace_environment_precompute {
  trace_texture* environment    = new trace_texture{};
  trace_texture* irradiance_map = new trace_texture{};
  trace_texture* specular_map   = new trace_texture{};
  trace_texture* brdf_lut       = new trace_texture{};
} trace_env_precumpute;

bool        compute_trace_env_maps = false;
inline void init_cpu_ibl(trace_scene* scene) {
  auto& img    = scene->environments[0]->emission_tex->hdr.data_vector();
  auto  extent = scene->environments[0]->emission_tex->hdr.imsize();

  if (compute_trace_env_maps) {
    auto& irradianceMap  = scene->trace_env->irradiance_map->hdr;
    irradianceMap.extent = ir_extent;
    irradianceMap.pixels = computeIrradianceTexture(img, extent);

    auto& BRDFLUT  = scene->trace_env->brdf_lut->hdr;
    BRDFLUT.extent = brdf_extent;
    BRDFLUT.pixels = computeBRDFLUT();

    auto& prefilteredMapsVector = scene->trace_env->specular_map;
    auto  prefilteredMaps       = computePrefilteredTextures(
        img, extent, prefiltered_maps_levels);
    for (int i = 0; i < prefiltered_maps_levels; i++) {
      trace_texture* spec_level_texture = new trace_texture{};
      spec_level_texture->hdr.extent    = prefiltered_maps_extent;
      spec_level_texture->hdr.pixels    = prefilteredMaps[i];
      prefilteredMapsVector.push_back(spec_level_texture);
    }
  } else {
    // load trace env maps from existing images
    scene->trace_env->environment = scene->environments[0]->emission_tex;

    string err;
    yocto::load_image("BRDFLut.hdr", scene->trace_env->brdf_lut->hdr, err);
    yocto::load_image(
        "irradiance.hdr", scene->trace_env->irradiance_map->hdr, err);

    for (int i = 0; i < prefiltered_maps_levels; i++) {
      trace_texture* spec_level_texture = new trace_texture{};

      char buff[100];
      snprintf(buff, sizeof(buff), "prefiltered_%d.hdr", i);
      std::string path = buff;
      yocto::load_image(buff, spec_level_texture->hdr, err);

      scene->trace_env->specular_map.push_back(spec_level_texture);
    }
  }
}
}  // namespace cpuibl
