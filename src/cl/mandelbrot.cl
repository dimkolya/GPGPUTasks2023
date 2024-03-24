#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters,
                         unsigned int antialiasing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    float avg_result = 0.0f;
    if (i < width && j < height) {
        for (unsigned int k = 1; k <= antialiasing; ++k) {
            for (unsigned int l = 1; l <= antialiasing; ++l) {
                float x0 = fromX + (i + 1.0f / (antialiasing + 1) * k) * sizeX / width;
                float y0 = fromY + (j + 1.0f / (antialiasing + 1) * l) * sizeY / height;

                float x = x0;
                float y = y0;

                int iter = 0;
                for (; iter < iters; ++iter) {
                    float xPrev = x;
                    x = x * x - y * y + x0;
                    y = 2.0f * xPrev * y + y0;
                    if ((x * x + y * y) > threshold2) {
                        break;
                    }
                }
                float result = 1.0f * iter / iters;
                avg_result += result;
            }
        }
    }
    results[j * width + i] = avg_result / (antialiasing * antialiasing);
}
