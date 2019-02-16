extern "C"
__global__ void add(int n, float *cRarr, float *cIarr, int *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float cR = cRarr[i];
        float cI = cIarr[i];

        int n = 0;
        float x = 0;
        float y = 0;

        for(n = 0; (y*y) < 4 && n < 255; n++) {
            float xNew = (x * x) - (y * y) + cR;

            y = (2 * x * y) + cI;
            x = xNew;
        }

        result[i] = n;
    }
}