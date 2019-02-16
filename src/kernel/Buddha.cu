extern "C"

#define ERR -1

__global__ void exec(int iterations, int size,
                float* inputR,  float* inputI, // Real/Imaginary input
                int* outputX, int* outputY // Location Results
                ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float cR = inputR[i];
    float cI = inputI[i];

    float x = 0;
    float y = 0;

    int startIdx = i * iterations;

    for (int j = 0; j < iterations; j++) {
        outputX[startIdx + j] = ((x + 2) * size) / 4;
        outputY[startIdx + j] = ((y + 2) * size) / 4;

        float xNew = (x * x) - (y * y) + cR;
        float yNew = (2 * x * y) + cI;

        if (xNew * xNew + yNew * yNew > 4) {
            outputX[startIdx] = j-1;

            for (int k = j; k < iterations; k++) {
                outputX[startIdx+k] = 0;
                outputY[startIdx+k] = 0;
            }

            return;
        }

        x = xNew;
        y = yNew;
    }

    outputX[startIdx] = ERR;
}