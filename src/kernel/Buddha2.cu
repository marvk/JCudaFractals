extern "C"

#define ITERATIONS 10000

__global__ void exec(int iterations, int size,
                float* inputR,  float* inputI, // Real/Imaginary input
                int* output                    // Output image in one dimension
                ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float cR = inputR[i];
    float cI = inputI[i];

    float q = ((cR - (1.0 / 4.0)) * (cR - (1.0 / 4.0))) + (cI * cI);

    if (q * (q + (cR - (1.0 / 4.0))) < (1.0 / 4.0) * (cI * cI)
                    || (cR + 1.0) * (cR + 1.0) + (cI * cI) < (1.0 / 16.0))
                    return;

    float x = 0;
    float y = 0;

    float outX[ITERATIONS];
    float outY[ITERATIONS];

    for (int j = 0; j < iterations; j++) {
        outX[j] = x;
        outY[j] = y;

        float xNew = (x * x) - (y * y) + cR;
        float yNew = (2 * x * y) + cI;

        if (xNew * xNew + yNew * yNew > 4) {
            for (int k = 1; k < j; k++) {
                int curX = (outX[k] + 2 ) * size / 4;
                int curY = (outY[k] + 2 ) * size / 4;

                int idx = curX + size * curY;

                output[idx]++;
                output[idx]++;
            }
            return;
        }

        x = xNew;
        y = yNew;
    }
}