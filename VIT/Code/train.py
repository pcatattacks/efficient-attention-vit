import torch
from torch import nn, optim

from utils import (
    save_experiment,
    save_checkpoint,
    topk_accuracies,
    count_parameters,
    measure_inference_latency,
    get_model_flops,
)
from data import prepare_data
from vit import ViTForClassfication


config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.output_dir = output_dir  # root for experiments/results

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        
        train_losses, test_losses = [], []
        top1_accuracies, top5_accuracies = [], []
        epoch_times = []
        peak_memories = []
        
        for i in range(epochs):
            # reset peak memory stats per epoch if on CUDA
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
        
            start_time = time.time()
        
            train_loss = self.train_epoch(trainloader)
            (top1, top5), test_loss = self.evaluate(testloader)
        
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
        
            if self.device.startswith("cuda"):
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)  # MB
            else:
                peak_mem = None
            peak_memories.append(peak_mem)
        
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            top1_accuracies.append(top1)
            top5_accuracies.append(top5)
        
            print(
                f"Epoch: {i+1}, "
                f"Train loss: {train_loss:.4f}, "
                f"Test loss: {test_loss:.4f}, "
                f"Top-1: {top1:.4f}, Top-5: {top5:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
        
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1,output_dir=self.output_dir)
        
        
        # global metrics (once per experiment)
        cfg = self.model.config
        input_size = (cfg["num_channels"], cfg["image_size"], cfg["image_size"])
        
        param_count = count_parameters(self.model)
        latency_mean_ms, latency_std_ms = measure_inference_latency(
            self.model, self.device, input_size=input_size
        )
        flops_macs, _ = get_model_flops(self.model, self.device, input_size=input_size)
        
        print(f"\nFinal metrics for {self.exp_name}:")
        print(f"  Params: {param_count:,}")
        if flops_macs is not None:
            print(f"  FLOPs (MACs): {flops_macs:.3e}")
        print(f"  Inference latency: {latency_mean_ms:.3f} ± {latency_std_ms:.3f} ms / image")
        
        # Save the experiment
        save_experiment(
            self.exp_name,
            config,
            self.model,
            train_losses,
            test_losses,
            top1_accuracies,
            epoch_times=epoch_times,
            top5_accuracies=top5_accuracies,
            peak_memories=peak_memories,
            param_count=param_count,
            inference_latency_ms={
                "mean": latency_mean_ms,
                "std": latency_std_ms,
            },
            flops=flops_macs,
            output_dir=self.output_dir,
        )
        save_summary_dataframe(
            self.exp_name,
            train_losses,
            test_losses,
            top1_accuracies,
            top5_accuracies,
            epoch_times,
            peak_memories,
            param_count,
            {"mean": latency_mean_ms, "std": latency_std_ms},
            flops_macs,
            output_dir=self.output_dir,          # NEW
        )

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        total_top1 = 0.0
        total_top5 = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                logits, _ = self.model(images)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                accs = topk_accuracies(logits, labels, topk=(1, 5))
                total_top1 += accs[1] * len(images)
                total_top5 += accs[5] * len(images)
                n_samples += len(images)

        avg_loss = total_loss / n_samples
        top1 = total_top1 / n_samples
        top5 = total_top5 / n_samples

        return (top1, top5), avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")  # NEW
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    args = parse_args()
    # Training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    output_dir = args.output_dir  # NEW
    # Load the CIFAR10 dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device,output_dir=output_dir)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == "__main__":
    main()