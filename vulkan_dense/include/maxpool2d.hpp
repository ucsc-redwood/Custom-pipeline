
#pragma once
#include "application.hpp"


class Maxpool2d : public ApplicationBase {
    public:
    Maxpool2d() : ApplicationBase() {};
    ~Maxpool2d() {};
    void        submit(const int queue_idx);
	void 		 cleanup(VkPipeline *pipeline);
    void compute_constant(int input_channels,
    int input_height,
    int input_width,
    int pool_size,
    int stride);

	void run(const int logical_block,
	const int queue_idx,
    float *input_data,
    float *output_data,
	VkBuffer input_buffer,
    VkBuffer output_buffer,
	const int n);
    private:
	VkShaderModule shaderModule;
	VkDescriptorSetLayout descriptorSetLayouts[1] = {VkDescriptorSetLayout{}};
	VkDescriptorSet descriptorSets[1] = {VkDescriptorSet{}};
	VkDescriptorSetLayoutCreateInfo descriptorLayout[1] = {VkDescriptorSetLayoutCreateInfo{}};
	struct PushConstant {
        int input_channels;
        int input_height;
        int input_width;
        int pool_size;
        int stride;
	} maxpool2d_push_constant;
};


void Maxpool2d::submit(const int queue_idx) {
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

void Maxpool2d::cleanup(VkPipeline *pipeline) {
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
}

void Maxpool2d::compute_constant(
    int input_channels,
    int input_height,
    int input_width,
    int pool_size,
    int stride){
    maxpool2d_push_constant.input_channels = input_channels;
    maxpool2d_push_constant.input_height = input_height;
    maxpool2d_push_constant.input_width = input_width;
    maxpool2d_push_constant.pool_size = pool_size;
    maxpool2d_push_constant.stride = stride;
    }

void Maxpool2d::run(const int logical_block,
	const int queue_idx,
    float *input_data,
    float *output_data,
	VkBuffer input_buffer,
    VkBuffer output_buffer,
	const int n){
    int output_height = (maxpool2d_push_constant.input_height - maxpool2d_push_constant.pool_size) / maxpool2d_push_constant.stride + 1;
    int output_width = (maxpool2d_push_constant.input_width - maxpool2d_push_constant.pool_size) / maxpool2d_push_constant.stride + 1;
	int total_channels = maxpool2d_push_constant.input_channels;
    int thread_per_block_x = 16;
	int thread_per_block_y = 16;
	int num_blocks_x = total_channels;
	int num_blocks_y = (output_height + thread_per_block_y - 1) / thread_per_block_y;
	int num_blocks_z = (output_width + thread_per_block_x - 1) / thread_per_block_x;
    VkPipeline pipeline;

	// create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
	};

	create_descriptor_pool(poolSizes, 1);

	// create layout binding
    VkDescriptorSetLayoutBinding input_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0, 1);
    VkDescriptorSetLayoutBinding output_layoutBinding = build_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1, 1);

	std::vector<VkDescriptorSetLayoutBinding> set_layout_bindings = {
        input_layoutBinding, output_layoutBinding
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
    VkDescriptorBufferInfo output_bufferDescriptor = { output_buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet output_descriptor_write = create_descriptor_write(descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &output_bufferDescriptor);
	std::cout <<"create descriptor writes"<<std::endl;


	std::vector<VkWriteDescriptorSet> descriptor_writes = {
        input_descriptor_write, output_descriptor_write
    };
	vkUpdateDescriptorSets(singleton.device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, NULL);
	std::cout<<"update descriptor sets"<<std::endl;
	//create pipeline 
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("maxpool2d.spv", &shaderModule);
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
    VkBufferMemoryBarrier output_barrier = create_buffer_barrier(&output_buffer, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);

    create_pipeline_barrier(&input_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    create_pipeline_barrier(&output_barrier, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &maxpool2d_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, num_blocks_x, num_blocks_y, num_blocks_z);

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

// void maxpool2d(float* input_data, int input_channels, int input_height, int input_width,
//                int pool_size, int stride, float* output_data) {
//     int output_height = (input_height - pool_size) / stride + 1;
//     int output_width = (input_width - pool_size) / stride + 1;

//     // Allocate memory on the device
//     float* d_input_data;
//     float* d_output_data;
//     cudaMalloc(&d_input_data, input_channels * input_height * input_width * sizeof(float));
//     cudaMalloc(&d_output_data, input_channels * output_height * output_width * sizeof(float));

//     // Copy data from host to device
//     cudaMemcpy(d_input_data, input_data, input_channels * input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);

//     // Define block and grid sizes
//     dim3 blockSize(16, 16, 1);
//     dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x,
//                   (output_height + blockSize.y - 1) / blockSize.y,
//                   input_channels);

//     // Launch the kernel
//     maxpool2dKernel<<<gridSize, blockSize>>>(d_input_data, input_height, input_width,
//                                              pool_size, stride, d_output_data,
//                                              output_height, output_width, input_channels);

//     // Copy result back to host
//     cudaMemcpy(output_data, d_output_data, input_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_input_data);
//     cudaFree(d_output_data);
// }