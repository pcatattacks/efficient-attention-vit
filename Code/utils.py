import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import time  # NEW
from vit import ViTForClassfication


def topk_accuracies(logits, labels, topk=(1,)):
    """
    Compute top-k accuracies for a batch.
    Returns a dict {k: accuracy_in_[0,1]}.
    """
    maxk = max(topk)
    batch_size = labels.size(0)

    # (B, num_classes) -> (B, maxk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, B)
    correct = pred.eq(labels.view(1, -1).expand_as(pred))  # (maxk, B)

    accs = {}
    for k in topk:
        # correct[:k] has shape (k, B)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accs[k] = (correct_k / batch_size).item()
    return accs


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(model, device, input_size=(3, 32, 32),
                              num_warmup=10, num_iters=50):
    """
    Measure average forward-pass latency (ms / image) on a single image.
    """
    model.eval()
    dummy = torch.randn(1, *input_size, device=device)

    times_ms = []
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(dummy)
        # Timed runs
        for _ in range(num_iters):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.time()
            times_ms.append((end - start) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    return mean_ms, std_ms


def get_model_flops(model, device, input_size=(3, 32, 32)):
    """
    Estimate FLOPs using ptflops (if installed).
    Returns (macs, params) as raw numbers (not strings), or (None, None) if unavailable.

    Install with:
        pip install ptflops
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("ptflops not installed; skipping FLOPs computation.")
        return None, None

    model = model.to(device)
    model.eval()

    # ptflops uses MACs (multiply-accumulates). Often reported as "FLOPs".
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            input_res=input_size,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
    return macs, params


def save_checkpoint(experiment_name, model, epoch, output_dir="."):
    exp_root = os.path.join(output_dir, "experiments", experiment_name)
    os.makedirs(exp_root, exist_ok=True)
    cpfile = os.path.join(exp_root, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)


def save_experiment(
    experiment_name,
    config,
    model,
    train_losses,
    test_losses,
    accuracies,
    epoch_times=None,
    top5_accuracies=None,
    peak_memories=None,
    param_count=None,
    inference_latency_ms=None,
    flops=None,
    output_dir=".",             # NEW
):
    exp_root = os.path.join(output_dir, "experiments", experiment_name)
    os.makedirs(exp_root, exist_ok=True)

    configfile = os.path.join(exp_root, "config.json")
    with open(configfile, "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    jsonfile = os.path.join(exp_root, "metrics.json")
    with open(jsonfile, "w") as f:
        data = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "accuracies": accuracies,
        }
        if top5_accuracies is not None:
            data["top5_accuracies"] = top5_accuracies
        if epoch_times is not None:
            data["epoch_times"] = epoch_times
        if peak_memories is not None:
            data["peak_memories_mb"] = peak_memories
        if param_count is not None:
            data["param_count"] = param_count
        if inference_latency_ms is not None:
            data["inference_latency_ms"] = inference_latency_ms
        if flops is not None:
            data["flops_macs"] = flops

        json.dump(data, f, sort_keys=True, indent=4)

    save_checkpoint(experiment_name, model, "final", output_dir=output_dir)

    
def save_summary_dataframe(
    exp_name,
    train_losses,
    test_losses,
    top1_accuracies,
    top5_accuracies,
    epoch_times,
    peak_memories,
    param_count,
    inference_latency_ms,
    flops,
    output_dir=".",           # NEW
):
    
    final_train_loss = train_losses[-1]
    final_test_loss = test_losses[-1]
    final_top1 = top1_accuracies[-1]
    final_top5 = top5_accuracies[-1]

    df = pd.DataFrame([{
        "experiment": exp_name,
        "params": param_count,
        "flops_macs": flops,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "final_top1_accuracy": final_top1,
        "final_top5_accuracy": final_top5,
        "avg_epoch_time_sec": float(np.mean(epoch_times)),
        "std_epoch_time_sec": float(np.std(epoch_times)),
        "peak_memory_mb": max([m for m in peak_memories if m is not None] or [None]),
        "inference_latency_mean_ms": inference_latency_ms["mean"],
        "inference_latency_std_ms": inference_latency_ms["std"],
    }])

    results_root = os.path.join(output_dir, "results")
    os.makedirs(results_root, exist_ok=True)
    outfile = os.path.join(results_root, f"{exp_name}_summary.csv")
    df.to_csv(outfile, index=False)
    print(f"\nSaved summary DataFrame → {outfile}")

    return df


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    model.eval()
    # Load random images
    num_images = 30
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = torch.stack([test_transform(image) for image in raw_images])
    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    logits, attention_maps = model(images, output_attentions=True)
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()