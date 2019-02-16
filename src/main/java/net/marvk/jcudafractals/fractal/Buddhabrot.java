package net.marvk.jcudafractals.fractal;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import net.marvk.jcudafractals.cudahelper.CUHelper;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Random;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Created by Marvin on 25.05.2016.
 */
public class Buddhabrot {
    private static final int SIZE = 256;
    private static final long NUM_POINTS = 128 * 128 * 32;
    private static final int ITERATIONS = 1000;

    private static final int BLOCK_SIZE = 512;
    private static final int STAGE_SIZE = 2048 * 16;

    private static final Random RANDOM = new Random();

    public Buddhabrot() {
        //noinspection ConstantConditions

        if ((long) ITERATIONS * STAGE_SIZE > Integer.MAX_VALUE) {
            throw new IllegalStateException("Stage size to big for iteration count");
        }

        final CUfunction function;
        try {
            function = CUHelper.prepareContext("Buddha.cu", true);
        } catch (IOException | URISyntaxException e) {
            e.printStackTrace();
            System.exit(1);
            return;
        }

        final float[] inR = new float[STAGE_SIZE];
        final float[] inI = new float[STAGE_SIZE];

        final int[] outX = new int[STAGE_SIZE * ITERATIONS];
        final int[] outY = new int[STAGE_SIZE * ITERATIONS];

        final CUdeviceptr deviceInputR = CUHelper.allocDevicePointer(inR);
        final CUdeviceptr deviceInputI = CUHelper.allocDevicePointer(inI);

        final CUdeviceptr deviceOutputX = CUHelper.allocDevicePointer(outX);
        final CUdeviceptr deviceOutputY = CUHelper.allocDevicePointer(outY);

        final int[][] result = new int[SIZE][SIZE];
        for (long l = 0; l < NUM_POINTS; l += STAGE_SIZE) {
            System.out.println("Stage " + l);

            for (int j = 0; j < STAGE_SIZE; ) {
                final float cR = RANDOM.nextFloat() * 4f - 2f;
                final float cI = RANDOM.nextFloat() * 4f - 2f;

                if (!FractalHelper.isInCardiodicOrP2Bulb(cR, cI)) {
                    inR[j] = cR;
                    inI[j] = cI;

                    //noinspection AssignmentToForLoopParameter
                    j++;
                }
            }

            System.out.println("GPU START");

            CUHelper.copyArrayToGPU(inR, deviceInputR);
            CUHelper.copyArrayToGPU(inI, deviceInputI);

            final Pointer kernelParameters = Pointer.to(
                    Pointer.to(new int[]{ITERATIONS}),
                    Pointer.to(new int[]{SIZE}),
                    Pointer.to(deviceInputR),
                    Pointer.to(deviceInputI),
                    Pointer.to(deviceOutputX),
                    Pointer.to(deviceOutputY)
            );

            final int gridSize = (int) Math.ceil(((double) STAGE_SIZE) / BLOCK_SIZE);

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
                final int start = j * ITERATIONS;
                final int it = outX[start];

                for (int k = start + 1; k <= start + it; k++) {
                    try {
                        result[outX[k]][outY[k]]++;
                    } catch (final ArrayIndexOutOfBoundsException ignored) {
                    }
                }
            }
        }

        final BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);

        final Color[] colors = IntStream.range(0, 256).mapToObj(i -> new Color(i, i, i)).toArray(Color[]::new);

        float max = Integer.MIN_VALUE;

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                max = Math.max(max, result[j][i]);
            }
        }

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                image.setRGB(j, i, colors[(int) (result[j][i] / max * 255)].getRGB());
            }
        }

        try {
            ImageIO.write(image, "png", new File("out.png"));
        } catch (final IOException e) {
            e.printStackTrace();
        }
    }
}
