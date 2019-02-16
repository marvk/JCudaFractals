package cudahelper;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import static jcuda.driver.JCudaDriver.*;

/**
 * Created by Marvin on 25.05.2016.
 */
public class CUHelper {
    private static final String kernelFolder = "src/kernel/";

    private CUHelper() {
        //no instance
    }

    public static CUfunction prepareContext(String fileName, boolean compilePtx) throws IOException {
        if (compilePtx)
            compileKernel(fileName);

        File ptxFile = new File(kernelFolder + fileName.substring(0, fileName.lastIndexOf(".")) + ".ptx");

        setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFile.getAbsolutePath());

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "exec");

        cuCtxSetLimit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE, 4096);

        return function;
    }

    public static void compileKernel(String fileName) throws IOException {
        File cuFile = new File(kernelFolder + fileName);
        File ptxFile = new File(kernelFolder + fileName.substring(0, fileName.lastIndexOf(".")) + ".ptx");

        if (!cuFile.exists())
            throw new IOException(cuFile + " could not be found");

        if (ptxFile.exists() && ptxFile.lastModified() > cuFile.lastModified())
            return;

        String command = "nvcc -ptx " + cuFile.getAbsolutePath() + " -o " + ptxFile.getAbsolutePath();

        System.out.println(command);

        try {
            Process process = Runtime.getRuntime().exec(command);
            process.waitFor();

            Scanner sc = new Scanner(process.getErrorStream());

            if (sc.hasNext()) {
                while (sc.hasNextLine())
                    System.out.println(sc.nextLine());

                throw new IllegalStateException("Could not compile " + cuFile.getName());
            }

        } catch (IOException | InterruptedException | IllegalStateException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static CUdeviceptr allocDevicePointer(float[] array) {
        CUdeviceptr pointer = new CUdeviceptr();
        cuMemAlloc(pointer, array.length * Sizeof.FLOAT);
        return pointer;
    }

    public static CUdeviceptr allocDevicePointer(int[] array) {
        CUdeviceptr pointer = new CUdeviceptr();
        cuMemAlloc(pointer, array.length * Sizeof.INT);
        return pointer;
    }

    public static CUdeviceptr allocDevicePointerAndCopy(float[] array) {
        CUdeviceptr pointer = allocDevicePointer(array);
        copyArrayToGPU(array, pointer);
        return pointer;
    }

    public static CUdeviceptr allocDevicePointerAndCopy(int[] array) {
        CUdeviceptr pointer = allocDevicePointer(array);
        copyArrayToGPU(array, pointer);
        return pointer;
    }

    public static void copyArrayToGPU(float[] array, CUdeviceptr pointer) {
        cuMemcpyHtoD(pointer, Pointer.to(array), array.length * Sizeof.FLOAT);
    }

    public static void copyArrayToGPU(int[] array, CUdeviceptr pointer) {
        cuMemcpyHtoD(pointer, Pointer.to(array), array.length * Sizeof.INT);
    }

    public static void copyArrayFromGPU(float[] array, CUdeviceptr pointer) {
        cuMemcpyDtoH(Pointer.to(array), pointer, array.length * Sizeof.FLOAT);
    }

    public static void copyArrayFromGPU(int[] array, CUdeviceptr pointer) {
        cuMemcpyDtoH(Pointer.to(array), pointer, array.length * Sizeof.INT);
    }
}
