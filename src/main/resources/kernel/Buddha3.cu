extern "C"

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
    float xNew = 0;
    float yNew = 0;

    int divergeIndex = 0;



    for (int j = 0; j < iterations; j++) {
        xNew = (x * x) - (y * y) + cR;
        yNew = (2 * x * y) + cI;

        if (xNew * xNew + yNew * yNew > 4) {
            divergeIndex = j;
            break;
        }

        x = xNew;
        y = yNew;
    }

    if (divergeIndex == 0) {
        return;
    }

    x = 0;
    y = 0;
    xNew = 0;
    yNew = 0;

    int curX = 0;
    int curY = 0;
    int idx = 0;



    for (int j = 0; j < divergeIndex; j++) {
        xNew = (x * x) - (y * y) + cR;
        yNew = (2 * x * y) + cI;

        curX = (xNew + 2 ) * size / 4;
        curY = (yNew + 2 ) * size / 4;

        idx = curX + size * curY;

        output[idx]++;
        output[idx]++;

        x = xNew;
        y = yNew;
    }
}