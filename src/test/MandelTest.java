package test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.*;

/**
 * Created by Marvin on 25.05.2016.
 */
public class MandelTest {
    public static final int SIZE = 16384;
    public static final int BLOCK_SIZE = 512;
    public static final int STAGE_SIZE = 2048 * 64;

    public static final String ptxFile = "B:\\Marvin\\IdeaProjects\\CudaFractals\\src\\kernel\\Mandel";

    public static void main(String[] args) {
        int[] result = new int[SIZE * SIZE];

        CUfunction function = prepareContext(ptxFile);

        System.out.println("START GPU");

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < SIZE * SIZE; i += STAGE_SIZE) {
            System.out.println("Stage " + i);
            float[] inX = new float[STAGE_SIZE];
            float[] inY = new float[STAGE_SIZE];

            int[] blockResult = new int[STAGE_SIZE];

            for (int j = 0; j < STAGE_SIZE; j++) {
                inX[j] = (((i + j) % SIZE) / (float) SIZE) * 4f - 2f;
                inY[j] = (((i + j) / SIZE) / (float) SIZE) * 4f - 2f;
            }

            CUdeviceptr deviceInputX = new CUdeviceptr();
            cuMemAlloc(deviceInputX, inX.length * Sizeof.FLOAT);
            cuMemcpyHtoD(deviceInputX, Pointer.to(inX), STAGE_SIZE * Sizeof.FLOAT);

            CUdeviceptr deviceInputY = new CUdeviceptr();
            cuMemAlloc(deviceInputY, inY.length * Sizeof.FLOAT);
            cuMemcpyHtoD(deviceInputY, Pointer.to(inY), STAGE_SIZE * Sizeof.FLOAT);

            CUdeviceptr deviceOutput = new CUdeviceptr();
            cuMemAlloc(deviceOutput, STAGE_SIZE * Sizeof.INT);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(new int[]{STAGE_SIZE}),
                    Pointer.to(deviceInputX),
                    Pointer.to(deviceInputY),
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

            cuMemcpyDtoH(Pointer.to(blockResult), deviceOutput, STAGE_SIZE * Sizeof.INT);

            for (int j = 0; j < STAGE_SIZE; j++) {
                result[i + j] = blockResult[j];
            }

            cuMemFree(deviceInputX);
            cuMemFree(deviceInputY);
            cuMemFree(deviceOutput);
        }

        System.out.println("END GPU");
        System.out.println("Time elapsed: " + ((double)(System.currentTimeMillis()-startTime) / 1000.0) + "s");

        BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);

        Color[] colors = IntStream.range(0, 256).mapToObj(i -> new Color(i, i, i)).toArray(Color[]::new);

        for (int i = 0; i < SIZE * SIZE; i++) {
            image.setRGB(i % SIZE, i / SIZE, colors[result[i]].getRGB());
        }

        try {
            ImageIO.write(image, "png", new File("out.png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("Duplicates")
    public static CUfunction prepareContext(String ptxFileName) {
        compileKernel(ptxFileName);

        setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName + ".ptx");

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        return function;
    }

    public static void compileKernel(String ptxFileName) {
        String command = "nvcc -ptx " + ptxFileName + ".cu -o " + ptxFileName + ".ptx";
        try {
            Runtime.getRuntime().exec(command).waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
}
