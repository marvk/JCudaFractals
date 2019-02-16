package net.marvk.jcudafractals.cudahelper;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Scanner;

import static jcuda.driver.JCudaDriver.*;

/**
 * Created by Marvin on 25.05.2016.
 */
public final class CUHelper {
    private CUHelper() {
        throw new AssertionError("No instances of utility class " + CUHelper.class);
    }

    public static CUfunction prepareContext(final String fileName, final boolean compilePtx) throws IOException, URISyntaxException {
        if (compilePtx) {
            compileKernel(fileName);
        }

        final String fn = fileName.substring(0, fileName.lastIndexOf(".")) + ".ptx";

        final File ptxFile = new File(ClassLoader.getSystemClassLoader().getResource("kernel/" + fn).toURI());

        setExceptionsEnabled(true);
        cuInit(0);

        final CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        final CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        final CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFile.getAbsolutePath());

        final CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "exec");

        cuCtxSetLimit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE, 4096);

        return function;
    }

    public static void compileKernel(final String fileName) throws IOException, URISyntaxException {
        final ClassLoader cl = ClassLoader.getSystemClassLoader();
        final URL resource = cl.getResource("kernel/" + fileName);

        final File cuFile = new File(resource.toURI());
        final String fn = fileName.substring(0, fileName.lastIndexOf(".")) + ".ptx";

        final File ptxFile = new File(cl.getResource("kernel/" + fn).toURI());

        if (!cuFile.exists()) {
            throw new IOException(cuFile + " could not be found");
        }

        if (ptxFile.exists() && ptxFile.lastModified() > cuFile.lastModified()) {
            return;
        }

        final String command = "nvcc -ptx " + cuFile.getAbsolutePath() + " -o " + ptxFile.getAbsolutePath();

        System.out.println(command);

        try {
            final Process process = Runtime.getRuntime().exec(command);
            process.waitFor();

            final Scanner sc = new Scanner(process.getErrorStream());

            if (sc.hasNext()) {
                while (sc.hasNextLine()) {
                    System.out.println(sc.nextLine());
                }

                throw new IllegalStateException("Could not compile " + cuFile.getName());
            }

        } catch (final IOException | InterruptedException | IllegalStateException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static CUdeviceptr allocDevicePointer(final float[] array) {
        final CUdeviceptr pointer = new CUdeviceptr();
        cuMemAlloc(pointer, array.length * Sizeof.FLOAT);
        return pointer;
    }

    public static CUdeviceptr allocDevicePointer(final int[] array) {
        final CUdeviceptr pointer = new CUdeviceptr();
        cuMemAlloc(pointer, array.length * Sizeof.INT);
        return pointer;
    }

    public static CUdeviceptr allocDevicePointerAndCopy(final float[] array) {
        final CUdeviceptr pointer = allocDevicePointer(array);
        copyArrayToGPU(array, pointer);
        return pointer;
    }

    public static CUdeviceptr allocDevicePointerAndCopy(final int[] array) {
        final CUdeviceptr pointer = allocDevicePointer(array);
        copyArrayToGPU(array, pointer);
        return pointer;
    }

    public static void copyArrayToGPU(final float[] array, final CUdeviceptr pointer) {
        cuMemcpyHtoD(pointer, Pointer.to(array), array.length * Sizeof.FLOAT);
    }

    public static void copyArrayToGPU(final int[] array, final CUdeviceptr pointer) {
        cuMemcpyHtoD(pointer, Pointer.to(array), array.length * Sizeof.INT);
    }

    public static void copyArrayFromGPU(final float[] array, final CUdeviceptr pointer) {
        cuMemcpyDtoH(Pointer.to(array), pointer, array.length * Sizeof.FLOAT);
    }

    public static void copyArrayFromGPU(final int[] array, final CUdeviceptr pointer) {
        cuMemcpyDtoH(Pointer.to(array), pointer, array.length * Sizeof.INT);
    }
}
