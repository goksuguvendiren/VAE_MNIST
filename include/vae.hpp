#include <torch/torch.h>

struct VAE : public torch::nn::Module
{
    VAE()
    {
        fc1  = register_module("fc1", torch::nn::Linear(784, 400));
        fc21 = register_module("fc21", torch::nn::Linear(400, 20));
        fc22 = register_module("fc22", torch::nn::Linear(400, 20));

        fc3  = register_module("fc3", torch::nn::Linear(20, 400));
        fc4  = register_module("fc4", torch::nn::Linear(400, 784));
    }

    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        auto mu = fc21->forward(x);
        auto var_log = fc22->forward(x);
        return {mu, var_log};
    }

    torch::Tensor decode(torch::Tensor x)
    {
        x = torch::relu(fc3->forward(x));
        return torch::sigmoid(fc4->forward(x));
    }

    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var)
    {
        auto std = torch::exp(log_var * 0.5);
        auto eps = torch::randn_like(std);

        return mu + eps * std;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = x.view({-1, 28 * 28});
        auto [mu, log_var] = encode(x);
        auto z = reparameterize(mu, log_var);
        return {decode(z), mu, log_var};
    }

    torch::nn::Linear fc1{nullptr}, fc21{nullptr}, fc22{nullptr}, fc3{nullptr}, fc4{nullptr};
};