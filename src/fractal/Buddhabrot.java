package fractal;

import cudahelper.CUHelper;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Created by Marvin on 25.05.2016.
 */
public class Buddhabrot {
    public static final int SIZE = 256;
    public static final long NUM_POINTS = 128 * 128 * 32;
    public static final int ITERATIONS = 1000;

    public static final int BLOCK_SIZE = 512;
    public static final int STAGE_SIZE = 2048 * 16;

    public static final Random random = new Random();

    public Buddhabrot() {
        CUfunction function;

        int[][] result = new int[SIZE][SIZE];

        //noinspection ConstantConditions
        if ((long) ITERATIONS * (long) STAGE_SIZE > Integer.MAX_VALUE) {
            throw new IllegalStateException("Stage size to big for iteration count");
        }

        try {
            function = CUHelper.prepareContext("Buddha.cu", true);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return;
        }

        float[] inR = new float[STAGE_SIZE];
        float[] inI = new float[STAGE_SIZE];

        int[] outX = new int[STAGE_SIZE * ITERATIONS];
        int[] outY = new int[STAGE_SIZE * ITERATIONS];

        CUdeviceptr deviceInputR = CUHelper.allocDevicePointer(inR);
        CUdeviceptr deviceInputI = CUHelper.allocDevicePointer(inI);

        CUdeviceptr deviceOutputX = CUHelper.allocDevicePointer(outX);
        CUdeviceptr deviceOutputY = CUHelper.allocDevicePointer(outY);

        for (long i = 0; i < NUM_POINTS; i += STAGE_SIZE) {
            System.out.println("Stage " + i);

            for (int j = 0; j < STAGE_SIZE; ) {
                float cR = random.nextFloat() * 4f - 2f;
                float cI = random.nextFloat() * 4f - 2f;

                if (!FractalHelper.isInCardiodicOrP2Bulb(cR, cI)) {
                    inR[j] = cR;
                    inI[j] = cI;

                    j++;
                }
            }

            System.out.println("GPU START");

            CUHelper.copyArrayToGPU(inR, deviceInputR);
            CUHelper.copyArrayToGPU(inI, deviceInputI);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(new int[]{ITERATIONS}),
                    Pointer.to(new int[]{SIZE}),
                    Pointer.to(deviceInputR),
                    Pointer.to(deviceInputI),
                    Pointer.to(deviceOutputX),
                    Pointer.to(deviceOutputY)
            );

            int gridSize = (int) Math.ceil(((double) STAGE_SIZE) / ((double) BLOCK_SIZE));

            System.out.println(gridSize);

            cuLaunchKernel(function,
                    gridSize, 1, 1,
                    BLOCK_SIZE, 1, 1,
                    0, null,
                    kernelParameters, null
            );

            cuCtxSynchronize();


            CUHelper.copyArrayFromGPU(outX, deviceOutputX);
            CUHelper.copyArrayFromGPU(outY, deviceOutputY);

            System.out.println("GPU END");

            for (int j = 0; j < STAGE_SIZE; j++) {
                int start = j * ITERATIONS;
                int it = outX[start];

                for (int k = start + 1; k <= start + it; k++) {
                    try {
                        result[outX[k]][outY[k]]++;
                    } catch (ArrayIndexOutOfBoundsException ignored) {
                    }
                }
            }
        }

        BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);

        Color[] colors = IntStream.range(0, 256).mapToObj(i -> new Color(i, i, i)).toArray(Color[]::new);

//        for (int i = 2046 * 64; i < 2047 * 64; i++) {
//            System.out.println(Arrays.toString(Arrays.copyOfRange(outX, ITERATIONS * i, ITERATIONS * (i + 1))));
//            System.out.println(Arrays.toString(Arrays.copyOfRange(outY, ITERATIONS * i, ITERATIONS * (i + 1))));
//            System.out.println();
//        }

        float max = Integer.MIN_VALUE;

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                max = Math.max(max, result[j][i]);
            }
        }

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                image.setRGB(j, i, colors[(int) (((float) result[j][i]) / max * 255)].getRGB());
            }
        }

        try {
            ImageIO.write(image, "png", new File("out.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
