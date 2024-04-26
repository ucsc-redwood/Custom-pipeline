
#pragma once
#include "application.hpp"


class LinearLayer : public ApplicationBase {
    public:
    LinearLayer() : ApplicationBase() {};
    ~LinearLayer() {};
    void        submit(const int queue_idx);
	void 		 cleanup(VkPipeline *pipeline);
    void compute_constant(int input_size,
    int output_size);

	void run(const int logical_block,
	const int queue_idx,
    float *input_data,
    float *weighted_data,
    float *bias_data,
    float *output_data,
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
        int input_size;
        int output_size;
	} linear_layer_push_constant;
};


void LinearLayer::submit(const int queue_idx) {
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

void LinearLayer::cleanup(VkPipeline *pipeline) {
		vkDestroyDescriptorSetLayout(singleton.device, descriptorSetLayouts[0], nullptr);
		vkDestroyPipeline(singleton.device, *pipeline, nullptr);
		vkDestroyShaderModule(singleton.device, shaderModule, nullptr);
}

void LinearLayer::compute_constant(
    int input_size,
    int output_size){
    linear_layer_push_constant.input_size = input_size;
    linear_layer_push_constant.output_size = output_size;  
}

void LinearLayer::run(const int logical_block,
	const int queue_idx,
    float *input_data,
    float *weighted_data,
    float *bias_data,
    float *output_data,
	VkBuffer input_buffer,
    VkBuffer weighted_buffer,
    VkBuffer bias_buffer,
    VkBuffer output_buffer,    
	const int n){
    
    int thread_per_block = 256;
    int block_per_grid = (linear_layer_push_constant.output_size + thread_per_block - 1) / thread_per_block;
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
	VkPipelineShaderStageCreateInfo shader_stage = load_shader("linearlayer.spv", &shaderModule);
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

	vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant), &linear_layer_push_constant);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, descriptorSets, 0, 0);
	vkCmdDispatch(commandBuffer, block_per_grid, 1, 1);

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
