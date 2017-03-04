/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

struct vertexColor
{
	float x,y;
	float r,g,b;
};
	
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

const unsigned int window_width = 512;
const unsigned int window_height = 512;

__global__ void triangle_kernel(vertexColor* pos, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //Calculate the clip coordinates
    float u = 2.0 * x / (float) width - 1;
    float v = 1 - 2.0 * y / (float) height;

    //Calculate a color
    if(u<=1 && u>=-1 && v<=1 && v>=-1)
    {
	if(v <= u+0.5 && v <= -u+0.5 && v >= -0.5)
	{
		pos[x*width+y].x = u;
		pos[x*width+y].y = v;
		pos[x*width+y].r = 255;
		pos[x*width+y].g = 0;
		pos[x*width+y].b = 0;
	}
	else
	{
		pos[x*width+y].x = u;
		pos[x*width+y].y = v;
		pos[x*width+y].r = 0;
		pos[x*width+y].g = 0;
		pos[x*width+y].b = 0;
	}
    }
}

void launch_kernel(vertexColor *pos,  unsigned int width,
                   unsigned int height)
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    triangle_kernel<<< grid, block>>>(pos, width, height);
}

void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	vertexColor *dptr;
	cudaGraphicsMapResources(1, vbo_resource, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource);
	launch_kernel(dptr, window_width, window_height);

	cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	//Initialize VBO
	unsigned int size = window_width * window_height * sizeof(vertexColor);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Register VBO with CUDA
	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}


int main(void)
{
    //------ InitGL---------------//
    GLFWwindow* window;
    
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(window_width, window_height, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);
    glewInit();
    glfwSwapInterval(1);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0,0,window_width, window_height);
    //----------InitGL--------------//

    cudaGLSetGLDevice(0);

    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    
    runCuda(&cuda_vbo_resource);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, sizeof(float)*5, 0);
    glColorPointer(3, GL_FLOAT,sizeof(float)*5, (GLvoid*)(sizeof(float)*2));
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArrays(GL_POINTS, 0, window_width * window_height);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}
  
