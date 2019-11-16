#include <iostream>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include "vae.hpp"

torch::Tensor vae_loss(torch::Tensor pred, torch::Tensor target, torch::Tensor mu, torch::Tensor log_var)
{
	// Reconstruction error :
	// std::cerr << pred << '\n';
	// std::cerr << target.view({-1, 28 * 28}) << '\n';

	// std::cerr << pred.size(0) << '\n';
	// std::cerr << pred.size(1) << '\n';
	// std::cerr << pred.size(2) << '\n';
	// std::cerr << pred.size(3) << '\n';
	// std::cerr << pred << '\n';

	// cv::Mat input_data = cv::Mat(28, 28, CV_32FC1, pred.data_ptr<float>());
	// cv::imwrite("pred.png", input_data * 255);
	auto reconstruction = torch::binary_cross_entropy(pred, target.view({-1, 28 * 28}), {}, torch::Reduction::Sum);
	auto kl_divergence = -0.5 * torch::sum(1 + log_var - torch::pow(mu, 2) - torch::exp(log_var));
	
	return reconstruction + kl_divergence;
}

int main() {
	auto net = std::make_shared<VAE>();

	// Create multi-threaded data loader for MNIST data
	// Make sure to enter absolute path to the data directory for no errors later on
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			std::move(torch::data::datasets::MNIST("../mnist/").map(torch::data::transforms::Stack<>())), 1);
			// std::move(torch::data::datasets::MNIST("../mnist/").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
				// torch::data::transforms::Stack<>())), 1);
	torch::optim::Adam optimizer(net->parameters(), 1e-3);

	net->train();

	for(size_t epoch=1; epoch<=10; ++epoch) 
	{
		size_t batch_index = 0;
		// Iterate data loader to yield batches from the dataset
		for (auto& batch: *data_loader) {
			// Reset gradients
			optimizer.zero_grad();
			// Execute the model
			auto [prediction, mu, std] = net->forward(batch.data);
			// Compute loss value
			torch::Tensor loss = vae_loss(prediction, batch.data, mu, std);
			// std::cerr << loss.item<float>() << '\n';
			// return 0;
			// Compute gradients
			loss.backward();
			// Update the parameters
			optimizer.step();

			// Output the loss and checkpoint every 100 batches
			if (++batch_index % 100 == 0) {
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index 
					<< " | Loss: " << loss.mean().item<float>() << std::endl;
				torch::save(net, "net.pt");
			}
		}
	}
}
