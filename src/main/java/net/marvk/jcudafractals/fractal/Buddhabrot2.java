package net.marvk.jcudafractals.fractal;

import net.marvk.jcudafractals.cudahelper.CUHelper;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.SplittableRandom;

import static jcuda.driver.JCudaDriver.*;

/**
 * Created by Marvin on 25.05.2016.
 */
public class Buddhabrot2 {
    public static int SIZE;// = 4096;
    public static long NUM_POINTS;// = 128L * 128L * 128L * 128L * 16L;
    public static int ITERATIONS;// = 5000;

    public static final int BLOCK_SIZE = 512;
    public static final int STAGE_SIZE = 2048 * 8192;

    public static int[] buddhabrot(int iterations, long numPoints, int size) {
        SIZE = size;
        NUM_POINTS = numPoints;
        ITERATIONS = iterations;

        CUfunction function = null;

        try {
            function = CUHelper.prepareContext("Buddha3.cu", true);
        } catch (IOException | URISyntaxException e) {
            e.printStackTrace();
            System.exit(1);
        }

        float[] inR = new float[STAGE_SIZE];
        float[] inI = new float[STAGE_SIZE];

        int[] out = new int[SIZE * SIZE];

        CUdeviceptr deviceInputR = CUHelper.allocDevicePointer(inR);
        CUdeviceptr deviceInputI = CUHelper.allocDevicePointer(inI);

        CUdeviceptr deviceOutput = CUHelper.allocDevicePointer(out);

        for (long i = 0; i < NUM_POINTS; i += STAGE_SIZE) {

            System.out.println(((double) i / (double) NUM_POINTS) * 100.0 + "%");

//            for (int j = 0; j < STAGE_SIZE; j++) {
//                float cR = random.nextFloat() * 4f - 2f;
//                float cI = random.nextFloat() * 4f - 2f;
//
//                inR[j] = cR;
//                inI[j] = cI;
//            }

            //System.out.println("RANDOM START");

            double[] doubleR = new SplittableRandom().doubles(STAGE_SIZE, -2.0, 2.0).toArray();
            double[] doubleI = new SplittableRandom().doubles(STAGE_SIZE, -2.0, 2.0).toArray();

            for (int j = 0; j < STAGE_SIZE; j++) {
                inR[j] = (float) doubleR[j];
                inI[j] = (float) doubleI[j];
            }

            //System.out.println("RANDOM END");

            //System.out.println("GPU START");

            CUHelper.copyArrayToGPU(inR, deviceInputR);
            CUHelper.copyArrayToGPU(inI, deviceInputI);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(new int[]{ITERATIONS}),
                    Pointer.to(new int[]{SIZE}),
                    Pointer.to(deviceInputR),
                    Pointer.to(deviceInputI),
                    Pointer.to(deviceOutput)
            );

            int gridSize = (int) Math.ceil(((double) STAGE_SIZE) / ((double) BLOCK_SIZE));

            cuLaunchKernel(function,
                    gridSize, 1, 1,
                    BLOCK_SIZE, 1, 1,
                    0, null,
                    kernelParameters, null
            );

            cuCtxSynchronize();

            //System.out.println("GPU END");
        }

        CUHelper.copyArrayFromGPU(out, deviceOutput);

        cuMemFree(deviceInputI);
        cuMemFree(deviceInputR);
        cuMemFree(deviceOutput);

        return out;
    }
}
