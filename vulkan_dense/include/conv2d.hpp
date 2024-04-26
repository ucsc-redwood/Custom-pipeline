
#pragma once
#include "application.hpp"
#include "app_params.hpp"


class Conv2d : public ApplicationBase {
    public:
    Conv2d() : ApplicationBase() {};
    ~Conv2d() {};
    void        submit(const int queue_idx);
	void 		 cleanup(VkPipeline *cov2d_pipeline);
    void compute_constant(AppParams params);

	void run(const int logical_block,
	const int queue_idx,
    float *input_data,
	float *weighted_data,
	float *bias_data,
    float *output_data,
    int weight_output_channels,
	VkBuffer input_buffer,
    VkBuffer weighted_buffer,
    VkBuffer bias_buffer,
    VkBuffer output_buffer,
	const int n);

    private:
	VkShaderModule shaderModule;
	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};
	struct PushConstant {
    int weight_input_channels;
	int weight_height;
    int weight_width;
    int input_height;
    int input_width;
	int bias_number_of_elements;
    int kernel_size;
    int stride;
    int padding;
	bool relu;
	} conv2d_push_constant;
};


void Conv2d::submit(const int queue_idx) {
	vkResetFences(singleton.device, 1, &fence);
	const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
	VkSubmitInfo computeSubmitInfo {};
	computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
	computeSubmitInfo.commandBufferCount = 1;
	computeSubmitInfo.pCommandBuffers = &commandBuffer;
	vkQueueSubmit(singleton.queues[queue_idx], 1, &computeSubmitInfo, fence);
	vkWaitForFences(singleton.device, 1, &fence, VK_TRUE, UINT64_MAX);
}

void Conv2d::cleanup(VkPipeline *pipeline) {
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
}

void Conv2d::compute_constant( AppParams params){
    conv2d_push_constant.weight_input_channels = params.weight_input_channels;
	conv2d_push_constant.weight_height = params.weight_height;
	conv2d_push_constant.weight_width = params.weight_width;
    conv2d_push_constant.input_height = params.input_height;
    conv2d_push_constant.input_width = params.input_width;
	conv2d_push_constant.bias_number_of_elements = params.bias_number_of_elements;
    conv2d_push_constant.kernel_size = params.kernel_size;
    conv2d_push_constant.stride = params.stride;
    conv2d_push_constant.padding = params.padding;
	conv2d_push_constant.relu = params.relu;
    }

void Conv2d::run(const int logical_block,
	const int queue_idx,
    float *input_data,
	float *weighted_data,
	float *bias_data,
    float *output_data,
    int weight_output_channels,
	VkBuffer input_buffer,
    VkBuffer weighted_buffer,
    VkBuffer bias_buffer,
    VkBuffer output_buffer,
	const int n){
    int thread_per_block_x = 16;
    int thread_per_block_y = 16;
    int output_height = (conv2d_push_constant.input_height + 2 * conv2d_push_constant.padding - conv2d_push_constant.kernel_size) / conv2d_push_constant.stride + 1;
    int output_width = (conv2d_push_constant.input_width + 2 * conv2d_push_constant.padding - conv2d_push_constant.kernel_size) / conv2d_push_constant.stride + 1;
    int num_block_x = weight_output_channels;
    int num_block_y = (output_height + thread_per_block_y -1)/ thread_per_block_y;
    int num_block_z = (output_width + thread_per_block_x -1)/ thread_per_block_x;
    VkPipeline pipeline;

	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
    VkDescriptorSetLayoutBinding input_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
    VkDescriptorSetLayoutBinding weighted_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);
    VkDescriptorSetLayoutBinding bias_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2, 1);
    VkDescriptorSetLayoutBinding output_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3, 1);

	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
        input_layoutBinding, weighted_layoutBinding, bias_layoutBinding, output_layoutBinding
	};
	// create descriptor 
	create_descriptor_set_layout(set_layout_bindings, &descriptorLayout[0], &descriptorSetLayouts[0]);

	// initialize pipeline_layout and attach descriptor set layout to pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = init_pipeline_layout(1, descriptorSetLayouts);
	//add push constant to the pipeline layout
	VkPushConstantRange push_constant = init_push_constant(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant));
	add_push_constant(&pipelineLayoutCreateInfo, &push_constant, 1);
	vkCreatePipelineLayout(singleton.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);
	// allocate descriptor sets
	allocate_descriptor_sets(1, descriptorSetLayouts, descriptorSets);
	std::cout << "allocate descriptor sets"<<std::endl;

	// update descriptor sets, first we need to create write descriptor, then specify the destination set, binding number, descriptor type, and number of descriptors(buffers) to bind
    VkDescriptorBufferInfo input_bufferDescriptor = { input_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet input_descriptor_write  = create_descriptor_write(descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &input_bufferDescriptor);
    VkDescriptorBufferInfo weighted_bufferDescriptor = { weighted_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet weighted_descriptor_write = create_descriptor_write(descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &weighted_bufferDescriptor);
    VkDescriptorBufferInfo bias_bufferDescriptor = { bias_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet bias_descriptor_write = create_descriptor_write(descriptorSets[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &bias_bufferDescriptor);
    VkDescriptorBufferInfo output_bufferDescriptor = { output_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet output_descriptor_write = create_descriptor_write(descriptorSets[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &output_bufferDescriptor);
	std::cout <<"create descriptor writes"<<std::endl;


	std::vector<VkWriteDescriptorSet> descriptor_writes = {
        input_descriptor_write, weighted_descriptor_write, bias_descriptor_write, output_descriptor_write
    };
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	std::cout<<"update descriptor sets"<<std::endl;
	//create pipeline 
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("conv2d.spv", &shaderModule);
	create_pipeline(&shader_stage,&pipelineLayout, &pipeline);
	std::cout << "load shader"<<std::endl;

	// allocate the command buffer, specify the number of commands within a command buffer.
	allocate_command_buffer(1);
	
	// record command buffer, which involves binding the pipeline and descriptor sets,
	//specify the descriptor sets it would be using, and the number of logical blocks.

	VkCommandBufferBeginInfo cmdBufInfo {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	std::cout <<"begin command buffer"<<std::endl;
	// preparation
	vkBeginCommandBuffer(commandBuffer, &cmdBufInfo);
    VkBufferMemoryBarrier input_barrier = create_buffer_barrier(&input_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier weighted_barrier = create_buffer_barrier(&weighted_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier bias_barrier = create_buffer_barrier(&bias_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    VkBufferMemoryBarrier output_barrier = create_buffer_barrier(&output_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);

    create_pipeline_barrier(&input_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&weighted_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&bias_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&output_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &conv2d_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, num_block_x, num_block_y, num_block_z);

    output_barrier = create_buffer_barrier(&output_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    create_pipeline_barrier(&output_barrier, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);


	vkEndCommandBuffer(commandBuffer);


	// create fence
	create_fence();

	// submit the command buffer, fence and flush
	submit(queue_idx);



	vkQueueWaitIdle(singleton.queues[queue_idx]);
	std::cout <<"end command buffer"<<std::endl;

	cleanup(&pipeline);
}
// void conv2d(float* input_data, int input_channels, int input_height, int input_width,
//             float* weight_data, int output_channels, int kernel_height, int kernel_width,
//             float* bias_data, int stride, int padding, float* output_data) {
//     int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
//     int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

//     // Allocate device memory
//     float *d_input_data, *d_weight_data, *d_bias_data, *d_output_data;
//     cudaMalloc(&d_input_data, input_channels * input_height * input_width * sizeof(float));
//     cudaMalloc(&d_weight_data, output_channels * input_channels * kernel_height * kernel_width * sizeof(float));
//     cudaMalloc(&d_bias_data, output_channels * sizeof(float));
//     cudaMalloc(&d_output_data, output_channels * output_height * output_width * sizeof(float));

//     // Copy data to device
//     cudaMemcpy(d_input_data, input_data, input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_weight_data, weight_data, output_channels * input_channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_bias_data, bias_data, output_channels * sizeof(float), cudaMemcpyHostToDevice);

//     // Define block and grid sizes
//     dim3 blockSize(16, 16, 1);
//     dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x, (output_height + blockSize.y - 1) / blockSize.y, output_channels);

//     // Launch the kernel
//     conv2dKernel<<<gridSize, blockSize>>>(d_input_data, input_channels, input_height, input_width,
//                                           d_weight_data, output_channels, kernel_height, kernel_width,
//                                           d_bias_data, stride, padding,
//                                           d_output_data, output_height, output_width);

//     // Copy the result back to host
//     cudaMemcpy(output_data, d_output_data, output_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_input_data);
//     cudaFree(d_weight_data);
//     cudaFree(d_bias_data);
//     cudaFree(d_output_data);
// }