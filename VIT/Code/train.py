import torch
from torch import nn, optim
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import (
    save_experiment,
    save_checkpoint,
    topk_accuracies,
    count_parameters,
    measure_inference_latency,
    get_model_flops,
    save_summary_dataframe,
)

from data import prepare_data_imagenette, prepare_data_cifar10
from vit import ViTForClassfication
import time
import argparse


# ---------------------------
# Default CONFIG (updated dynamically depending on dataset)
# ---------------------------

config = {
    "patch_size": 8,          
    "hidden_size": 192,
    "num_hidden_layers": 8,
    "num_attention_heads": 3,
    "intermediate_size": 4 * 192,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 160,          # overwritten by dataset loader if needed
    "num_classes": 10,          # overwritten for CIFAR-10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
    "attention_type": "linformer",
    "linformer_k": 256,
}


class Trainer:
    """
    Simple trainer with:
    - LR scheduler (optional)
    - Best-checkpoint saving (by lowest test loss)
    - Early stopping (optional)
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        exp_name,
        device,
        output_dir,
        scheduler=None,
        early_stopping_patience=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.output_dir = output_dir

        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        train_losses, test_losses = [], []
        top1_accuracies, top5_accuracies = [], []
        epoch_times = []
        peak_memories = []

        # ---- best model tracking ----
        best_test_loss = float("inf")
        best_epoch = -1
        best_state_dict = None
        epochs_no_improve = 0
        min_delta = 1e-4  # minimum improvement to reset patience

        for i in range(epochs):
            # reset peak memory stats per epoch if on CUDA
            if str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)

            start_time = time.time()

            train_loss = self.train_epoch(trainloader)
            (top1, top5), test_loss = self.evaluate(testloader)

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            if str(self.device).startswith("cuda"):
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

            # ---- best model logic (by test loss) ----
            if test_loss < best_test_loss - min_delta:
                best_test_loss = test_loss
                best_epoch = i + 1
                epochs_no_improve = 0

                # store a copy of weights on CPU to avoid GPU bloat
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

                # optional: save a "best" checkpoint
                save_checkpoint(
                    self.exp_name,
                    self.model,
                    epoch=i + 1,
                    output_dir=self.output_dir,
                )
                print(f"\t[Best model updated at epoch {i+1} with val loss {test_loss:.4f}]")
            else:
                epochs_no_improve += 1

            # ---- early stopping ----
            if (
                self.early_stopping_patience is not None
                and epochs_no_improve >= self.early_stopping_patience
            ):
                print(
                    f"\nEarly stopping triggered at epoch {i+1} "
                    f"(no improvement for {self.early_stopping_patience} epochs)."
                )
                break

            # ---- LR scheduler step (after validation) ----
            if self.scheduler is not None:
                self.scheduler.step()

            # optional periodic saving of non-best checkpoints
            if (
                save_model_every_n_epochs > 0
                and (i + 1) % save_model_every_n_epochs == 0
                and (i + 1) != epochs
            ):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1, output_dir=self.output_dir)

        # ---- restore best weights before evaluating global metrics ----
        if best_state_dict is not None:
            print(f"\nRestoring best model from epoch {best_epoch} (val loss={best_test_loss:.4f}).")
            self.model.load_state_dict(best_state_dict)

        # global metrics (once per experiment, on best model)
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
        print(f"  Best epoch: {best_epoch}, Best val loss: {best_test_loss:.4f}")

        # Save the experiment summary (curves etc — still use ALL recorded epochs)
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
        summary_df = save_summary_dataframe(
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
            output_dir=self.output_dir,
            config=self.model.config,
        )
        return {
            "summary_df": summary_df,
            "best_epoch": best_epoch,
            "best_val_loss": best_test_loss,
            "param_count": param_count,
            "flops_macs": flops_macs,
            "latency_ms": {
                "mean": latency_mean_ms,
                "std": latency_std_ms,
            },
        }

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch

            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(images)

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0.0
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


# =============================================================
# ARGS
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "imagenette"], default="imagenette")
    parser.add_argument("--data-root", type=str, required=False, default=None)   # for Imagenette
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--sweep",action="store_true",help="Run a multi-config sweep instead of a single config.",)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


# =============================================================
# MAIN
# =============================================================

def main():
    args = parse_args()

    # --------------------- LOAD DATA (ONCE) ---------------------
    if args.dataset == "cifar10":
        trainloader, testloader, classes = prepare_data_cifar10(
            batch_size=args.batch_size
        )
        config["image_size"] = 32
        config["num_classes"] = 10

    elif args.dataset == "imagenette":
        if args.data_root is None:
            raise ValueError("Must provide --data-root when using Imagenette.")

        trainloader, testloader, classes = prepare_data_imagenette(
            root=args.data_root,
            batch_size=args.batch_size,
            image_size=config["image_size"],
        )
        config["num_classes"] = len(classes)

    base_image_size = config["image_size"]

    # =========================================================
    # MODE 1: SWEEP
    # =========================================================
    if args.sweep:
        patch_sizes = [8,10,16]           # must divide image_size=160
        linformer_ks = [32, 64, 128, 256]

        sweep_settings = []

        for ps in patch_sizes:
            # full attention
            sweep_settings.append({
                "name": f"full_ps{ps}",
                "attention_type": "full",
                "patch_size": ps,
                "linformer_k": None,
            })

            # linformer with various k
            for k in linformer_ks:
                sweep_settings.append({
                    "name": f"linf_ps{ps}_k{k}",
                    "attention_type": "linformer",
                    "patch_size": ps,
                    "linformer_k": k,
                })

        all_summary_dfs = []

        for setting in sweep_settings:
            print("\n==============================")
            print(f"Running config: {setting}")
            print("==============================\n")

            # Update config for this run
            config["attention_type"] = setting["attention_type"]
            config["patch_size"] = setting["patch_size"]
            config["linformer_k"] = setting["linformer_k"]
            config["seq_len"] = (base_image_size // config["patch_size"]) ** 2 + 1

            # Build model
            model = ViTForClassfication(config)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-2,
            )

            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
            )

            early_stopping_patience = 10

            # Name each sub-experiment: base_exp + suffix
            exp_name = f"{args.exp_name}_{setting['name']}"

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                exp_name=exp_name,
                device=args.device,
                output_dir=args.output_dir,
                scheduler=scheduler,
                early_stopping_patience=early_stopping_patience,
            )

            result = trainer.train(
                trainloader=trainloader,
                testloader=testloader,
                epochs=args.epochs,
                save_model_every_n_epochs=args.save_model_every,
            )

            all_summary_dfs.append(result["summary_df"])

        # ---- Combine all sweep results into one CSV ----
        combined_df = pd.concat(all_summary_dfs, ignore_index=True)
        results_root = os.path.join(args.output_dir, "results")
        os.makedirs(results_root, exist_ok=True)
        combined_path = os.path.join(
            results_root, f"{args.exp_name}_SWEEP_summary.csv"
        )
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined sweep summary → {combined_path}")

        # ---- Simple plots from combined CSV ----
        try:
            # Bar plot: Top-1 vs experiment
            plt.figure(figsize=(10, 5))
            plt.bar(combined_df["experiment"], combined_df["final_top1_accuracy"])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Final Top-1 Accuracy")
            plt.tight_layout()
            bar_path = os.path.join(results_root, f"{args.exp_name}_SWEEP_top1_bar.png")
            plt.savefig(bar_path)
            print(f"Saved bar plot → {bar_path}")

            # Scatter: FLOPs vs Top-1, colored by attention_type
            plt.figure(figsize=(8, 6))
            colors = combined_df["attention_type"].map(
                {"full": "C0", "linformer": "C1"}
            )
            plt.scatter(
                combined_df["flops_macs"],
                combined_df["final_top1_accuracy"],
                c=colors,
            )
            for _, row in combined_df.iterrows():
                plt.annotate(row["experiment"], (row["flops_macs"], row["final_top1_accuracy"]),
                             fontsize=8, alpha=0.7)
            plt.xlabel("FLOPs (MACs)")
            plt.ylabel("Final Top-1 Accuracy")
            plt.tight_layout()
            scatter_path = os.path.join(results_root, f"{args.exp_name}_SWEEP_flops_vs_top1.png")
            plt.savefig(scatter_path)
            print(f"Saved FLOPs vs Top-1 plot → {scatter_path}")

        except Exception as e:
            print("Plotting failed (non-fatal):", e)

        return

    # =========================================================
    # MODE 2: SINGLE RUN (no sweep)
    # =========================================================
    # recompute sequence length AFTER image_size / patch_size are set
    config["seq_len"] = (config["image_size"] // config["patch_size"]) ** 2 + 1

    model = ViTForClassfication(config)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    early_stopping_patience = 10

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        exp_name=args.exp_name,
        device=args.device,
        output_dir=args.output_dir,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
    )

    _ = trainer.train(
        trainloader=trainloader,
        testloader=testloader,
        epochs=args.epochs,
        save_model_every_n_epochs=args.save_model_every,
    )


if __name__ == "__main__":
    main()
