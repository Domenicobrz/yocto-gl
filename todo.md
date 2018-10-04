# Notes on future improvements of Yocto/GL

This file contains notes on future improvements of Yocto.
Please consider this to be just development notes and not any real planning.

## Next features

- generic
    - to_string
    - vec: consider constructors/conversions
    - cmdline with to_string and parse
    - min, lerp, clamp
- move ext files
- path tracing with pbrt MIS
- simple renderers for book
    - simple intersection
    - eyelight
    - anti-aliasing
    - naive path tracing
    - mirrors / glass
    - volume ?
    - product path tracing
- consider introducing trace_point
- refactor renderers into smaller pieces for readability
- direct with deltas
- split apps

## Next features

- LUTs
- tone curve
    - from web?
    - no implementation of Photoshop curve
    - maybe try monotonic spline interpolation
    - otherwise a simple Bezier with constraints
    - use lookup tables for this

## Scene changes

- simpler shapes functions
    - add functions on quads using embree parametrization
    - quad tangent field
    - add quads to shape
- embree
- yscene
    - convert everything
        - mcguire
        - gltf

## Examples

- bent floor
- crane subdivs
- utah teapot
- bunny embed

## Trace

- clarify light pdfs
- one path
    - compute light pdfs with intersection
- orthonormal basis
    - file:///Users/fabio/Downloads/Duff2017Basis.pdf
- volumes
    - make material models clear
    - start with a simple design
- bump mapping
    - convert bump maps to normal maps internally on load
- displacement
    - apply displacement on tesselation or on triangle vertices directly
- examples
    - hair
    - subdiv
    - displacement
- put back double sided
    - one sided shading
- refraction
    - rough surface refraction
- intersection
    - methods with projections?
        - why shear?
        - triangle
        - line
        - point
    - line/point parametrization that matches pbrt and embree
    - radius in offsetting rays
        - check embree to avoid
- bvh
    - sah
    - opacity
    - embree integration
- see mini path tracer features and include them all
- simple denoiser
    - joint bilateral denoiser
    - non-local means denoiser
    - denoiser from paris

## Port scenes

- bitterli
    - hair
    - render exclude list
    - kitchen
        - bump map
    - car
        - shading bug
        - enable kr
        - metallic paint
    - car2
        - shading bug
        - enable kr
        - metallic paint
    - flip double sided normals on at least large objects
- mcguire
    - exclude list
    - bad textures
    - bad emission
        - lost empire
        - sportscar
    - cameras
    - lights
- pbrt
    - crown material or re-export
    - landscape materials
    - pbrt include parser
- gltf exports

## Test scenes: simplify shape generation

- bent floor
- 0 roughness

## BVH

- add cutout to trace
- simplify build functions
- maybe put axis with internal
- simplify partition and nth_element function
    - include wrapper functions

## OpenMP

On Apple Clang, you need to add several options to use OpenMP's front end
instead of the standard driver option. This usually looks like
  -Xpreprocessor -fopenmp -lomp

You might need to make sure the lib and include directories are discoverable
if /usr/local is not searched:

  -L/usr/local/opt/libomp/lib -I/usr/local/opt/libomp/include

For CMake, the following flags will cause the OpenMP::OpenMP_CXX target to
be set up correctly:
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib

## Book

- apps
    - tonemap
    - naivetrace
    - pathtrace
    - itrace
- lib
    - trace.h / trace.cpp
    - traceio.h / traceio.cpp