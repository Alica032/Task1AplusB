#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main()
{
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");


    cl_device_id deviceId;
    int gpu_use = -1;

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex){
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            size_t deviceParameterSize = 0;

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceParameterSize));
            std::vector<unsigned char> deviceName(deviceParameterSize);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceParameterSize, deviceName.data(), nullptr));
            std::cout << "      Device name: " << deviceName.data() << std::endl;

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));

            if(deviceType == CL_DEVICE_TYPE_GPU){
                deviceId = device;
                gpu_use = 1;
                break;
            }

            deviceId = device;
            gpu_use = 0;
        }
        if(gpu_use==1)
            break;
    }

    if (gpu_use == -1)
        throw std::runtime_error("devices not found!");


    cl_int errcode_ret;
    cl_context context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceId, 0, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);


    unsigned int n = 100*1000*1000;

    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    cl_mem asBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                        n*sizeof(float), as.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    cl_mem bsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                        n*sizeof(float), bs.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    cl_mem csBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        n*sizeof(float), nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);


    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
//        std::cout << kernel_sources << std::endl;
    }


    auto string_content = kernel_sources.c_str();
        const size_t kernel_size = kernel_sources.length();
    cl_program program = clCreateProgramWithSource(context, 1, &string_content, &kernel_size, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);


    errcode_ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    OCL_SAFE_CALL(errcode_ret);


    size_t log_size = 0;

    OCL_SAFE_CALL(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));

    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;

    }

    cl_kernel kernel = clCreateKernel(program, "aplusb", &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(asBuffer), &asBuffer);
        clSetKernelArg(kernel, i++, sizeof(bsBuffer), &bsBuffer);
        clSetKernelArg(kernel, i++, sizeof(csBuffer), &csBuffer);
        clSetKernelArg(kernel, i++, sizeof(n), &n);
    }

        {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;

        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr,
                                                 &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
        }
        t.nextLap();
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GFlops: " << n / 1e9 / t.lapAvg() << std::endl;
        std::cout << "VRAM bandwidth: " <<  3*n*sizeof(float) / pow(1024, 3) / t.lapAvg() << " GB/s" << std::endl;

        }

        {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {

        cl_event event;
        OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue, csBuffer, CL_TRUE, 0,
                                          sizeof(float)*n, cs.data(), 0, nullptr, &event));

        OCL_SAFE_CALL(clWaitForEvents(1, &event));

        t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n*sizeof(float) / pow(1024, 3) / t.lapAvg() << " GB/s" << std::endl;
        }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseMemObject(asBuffer));
    OCL_SAFE_CALL(clReleaseMemObject(bsBuffer));
    OCL_SAFE_CALL(clReleaseMemObject(csBuffer));

    OCL_SAFE_CALL(clReleaseCommandQueue(commandQueue));
    OCL_SAFE_CALL(clReleaseContext(context));


    return 0;
}