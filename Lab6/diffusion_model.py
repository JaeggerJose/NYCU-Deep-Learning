import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from tqdm import tqdm
import os
from models import ContextUnet
from utils import SpriteDataset, generate_animation
from dataset import Iclevr_Dataset



class DiffusionModel(nn.Module):
    def __init__(self, device=None, dataset_name=None, checkpoint_name=None):
        super(DiffusionModel, self).__init__()
        self.device = self.initialize_device(device)
        self.file_dir = os.path.dirname(__file__)
        self.dataset_name = self.initialize_dataset_name(self.file_dir, checkpoint_name, dataset_name)
        self.checkpoint_name = checkpoint_name
        self.nn_model = self.initialize_nn_model(self.dataset_name, checkpoint_name, self.file_dir, self.device)
        self.create_dirs(self.file_dir)

    def train(self, batch_size=64, n_epoch=32, lr=1e-3, timesteps=500, beta1=1e-4, beta2=0.02,
              checkpoint_save_dir=None, image_save_dir=None, noise_schedule="cosine"):
        """Trains model for given inputs"""
        self.nn_model.train()        
        if noise_schedule == "linear":
            _ , _, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        else:
            _ , _, ab_t = self.get_ddpm_noise_schedule_cosine(timesteps, beta1, beta2, self.device)
        # prepare logging
        log_dir = os.path.join(self.file_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"{self.dataset_name}_train_loss.csv")
        loss_history = []
        epoch_history = []
        if self.dataset_name == "iclevr":
            dataset = Iclevr_Dataset()
        else:
            dataset = self.instantiate_dataset(self.dataset_name, 
                                self.get_transforms(self.dataset_name), self.file_dir)
        dataloader = self.initialize_dataloader(dataset, batch_size, self.checkpoint_name, self.file_dir)
        optim = self.initialize_optimizer(self.nn_model, lr, self.checkpoint_name, self.file_dir, self.device)
        scheduler = self.initialize_scheduler(optim, self.checkpoint_name, self.file_dir, self.device)

        start_epoch = self.get_start_epoch(self.checkpoint_name, self.file_dir)
        for epoch in range(start_epoch, start_epoch + n_epoch):
            ave_loss = 0
            print(f"lr: {optim.param_groups[0]['lr']}")

            for x, c in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
                x = x.to(self.device)
                c = c.to(self.device)
                
                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0], )).to(self.device)
                x_pert = self.perturb_input(x, t, noise, ab_t)

                # predict noise
                pred_noise = self.nn_model(x_pert, t / timesteps, c=c)

                # obtain loss
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                
                # update params
                optim.zero_grad()
                loss.backward()
                optim.step()

                ave_loss += loss.item()/len(dataloader)
            scheduler.step()
            print(f"Epoch: {epoch}, loss: {ave_loss}")
            # log to memory and csv
            loss_history.append(ave_loss)
            epoch_history.append(epoch)
            if not os.path.exists(csv_path):
                with open(csv_path, 'w') as f:
                    f.write('epoch,loss\n')
            with open(csv_path, 'a') as f:
                f.write(f"{epoch},{ave_loss}\n")
            # save curve png
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(epoch_history, loss_history, marker='o')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.title(f'{self.dataset_name} train loss')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, f"{self.dataset_name}_loss_curve.png"))
                plt.close()
            except Exception:
                pass
            if epoch % 5 == 0:
                self.generate(n_samples=32, n_images_per_row=8, timesteps=timesteps, beta1=beta1, beta2=beta2, epoch=epoch)
            if epoch % 10 == 0:
                self.save_tensor_images(x, x_pert, self.get_x_unpert(x_pert, t, pred_noise, ab_t), 
                                        epoch, self.file_dir, image_save_dir)
                self.save_checkpoint(self.nn_model, optim, scheduler, epoch, ave_loss, 
                                    timesteps, beta1, beta2, self.device, self.dataset_name,
                                    dataloader.batch_size, noise_schedule, self.file_dir, checkpoint_save_dir)

    @torch.no_grad()
    def sample_ddpm(self, n_samples, context=None, timesteps=None, 
                    beta1=None, beta2=None, save_rate=20, inference_transform=lambda x: x, noise_schedule=None):
        """Returns the final denoised sample x0,
        intermediate samples xT, xT-1, ..., x1, and
        times tT, tT-1, ..., t1
        """
        if all([timesteps, beta1, beta2]):
            schedule = noise_schedule or "cosine"
            if schedule == "linear":
                a_t, b_t, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
            else:
                a_t, b_t, ab_t = self.get_ddpm_noise_schedule_cosine(timesteps, beta1, beta2, self.device)
        else:
            timesteps, a_t, b_t, ab_t = self.get_ddpm_params_from_checkpoint(self.file_dir,
                                                                             self.checkpoint_name, 
                                                                             self.device)
        
        self.nn_model.eval()
        samples = torch.randn(n_samples, self.nn_model.in_channels, 
                              self.nn_model.height, self.nn_model.width, 
                              device=self.device)
        intermediate_samples = [samples.detach().cpu()] # samples at T = timesteps
        t_steps = [timesteps] # keep record of time to use in animation generation
        for t in range(timesteps, 0, -1):
            print(f"Sampling timestep {t}", end="\r")
            if t % 50 == 0: print(f"Sampling timestep {t}")

            z = torch.randn_like(samples) if t > 1 else 0
            pred_noise = self.nn_model(samples, 
                                       torch.tensor([t/timesteps], device=self.device)[:, None, None, None], 
                                       context)
            samples = self.denoise_add_noise(samples, t, pred_noise, a_t, b_t, ab_t, z)
            
            if t % save_rate == 1 or t < 8:
                intermediate_samples.append(inference_transform(samples.detach().cpu()))
                t_steps.append(t-1)
        return intermediate_samples[-1], intermediate_samples, t_steps

    @torch.no_grad()
    def sample_with_sampler(self, n_samples, context=None, sampler="ddpm", timesteps=None,
                            beta1=None, beta2=None, save_rate=20, ddim_steps=20, noise_schedule=None):
        """Generic sampling entry; supports 'ddpm' and 'ddim'. Returns (x0, intermediates, t_steps_or_None)."""
        sampler = sampler.lower()
        if sampler == "ddpm":
            return self.sample_ddpm(n_samples=n_samples, context=context, timesteps=timesteps,
                                    beta1=beta1, beta2=beta2, save_rate=save_rate, noise_schedule=noise_schedule)
        elif sampler == "ddim":
            # use sampling_functions.sample_ddim
            if all([timesteps, beta1, beta2]):
                schedule = noise_schedule or "cosine"
                if schedule == "linear":
                    a_t, b_t, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
                else:
                    a_t, b_t, ab_t = self.get_ddpm_noise_schedule_cosine(timesteps, beta1, beta2, self.device)
            else:
                timesteps, a_t, b_t, ab_t = self.get_ddpm_params_from_checkpoint(self.file_dir,
                                                                                 self.checkpoint_name,
                                                                                 self.device)
            from sampling_functions import sample_ddim
            self.nn_model.eval()
            samples, intermediate = sample_ddim(n_sample=n_samples,
                                                height=self.nn_model.height,
                                                width=self.nn_model.width,
                                                nn_model=self.nn_model,
                                                timesteps=timesteps,
                                                ab_t=ab_t,
                                                device=self.device,
                                                context=context,
                                                n=ddim_steps)
            # adapt return shape to match ddpm interface
            intermediates = []
            try:
                import torch as _torch
                for arr in intermediate:
                    if isinstance(arr, _torch.Tensor):
                        intermediates.append(arr)
                    else:
                        intermediates.append(_torch.from_numpy(arr))
            except Exception:
                intermediates = []
            return samples, intermediates, None
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    def perturb_input(self, x, t, noise, ab_t):
        """Perturbs given input
        i.e., Algorithm 1, step 5, argument of epsilon_theta in the article
        """
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise
    
    def instantiate_dataset(self, dataset_name, transforms, file_dir, train=True):
        """Returns instantiated dataset for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "iclevr"}, "Unknown dataset"
        
        transform, target_transform = transforms
        if dataset_name=="mnist":
            return MNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="fashion_mnist":
            return FashionMNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="sprite":
            return SpriteDataset(os.path.join(file_dir, "datasets"), transform, target_transform)
        if dataset_name=="cifar10":
            return CIFAR10(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name=="iclevr":
            return Iclevr_Dataset()
            

    def get_transforms(self, dataset_name):
        """Returns transform and target-transform for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10"}, "Unknown dataset"

        if dataset_name in {"mnist", "fashion_mnist", "cifar10"}:
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: 2*(x - 0.5)
            ])
            target_transform = transforms.Compose([
                lambda x: torch.tensor([x]),
                lambda class_labels, n_classes=10: nn.functional.one_hot(class_labels, n_classes).squeeze()
            ])

        if dataset_name=="sprite":
            transform = transforms.Compose([
                transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
                lambda x: 2*x - 1       # range [-1,1]
            ])
            target_transform = lambda x: torch.from_numpy(x).to(torch.float32)
        return transform, target_transform
    
    def get_x_unpert(self, x_pert, t, pred_noise, ab_t):
        """Removes predicted noise pred_noise from perturbed image x_pert"""
        return (x_pert - (1 - ab_t[t, None, None, None]).sqrt() * pred_noise) / ab_t.sqrt()[t, None, None, None]
    
    def initialize_nn_model(self, dataset_name, checkpoint_name, file_dir, device):
        """Returns the instantiated model based on dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "iclevr"}, "Unknown dataset name"

        if dataset_name in {"mnist", "fashion_mnist"}:
            nn_model = ContextUnet(in_channels=1, height=28, width=28, n_feat=64, n_cfeat=10, n_downs=2)
        elif dataset_name=="sprite":
            nn_model = ContextUnet(in_channels=3, height=16, width=16, n_feat=64, n_cfeat=5, n_downs=2)
        elif dataset_name == "cifar10":
            nn_model = ContextUnet(in_channels=3, height=32, width=32, n_feat=64, n_cfeat=10, n_downs=4)
        else:
            nn_model = ContextUnet(in_channels=3, height=64, width=64, n_feat=64, n_cfeat=24, n_downs=4)

        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            nn_model.to(device)
            nn_model.load_state_dict(checkpoint["model_state_dict"])
            return nn_model
        return nn_model.to(device)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, 
                        timesteps, beta1, beta2, device, dataset_name, batch_size, noise_schedule,
                        file_dir, save_dir):
        """Saves checkpoint for given variables"""
        if save_dir is None:
            fpath = os.path.join(file_dir, "checkpoints", f"{dataset_name}_checkpoint_{epoch}.pth")
        else:
            fpath = os.path.join(save_dir, f"{dataset_name}_checkpoint_{epoch}.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "timesteps": timesteps, 
            "beta1": beta1, 
            "beta2": beta2,
            "device": device,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "noise_schedule": noise_schedule
        }
        torch.save(checkpoint, fpath)

    def create_dirs(self, file_dir):
        """Creates directories required for training"""
        dir_names = ["checkpoints", "saved-images", "logs"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    def initialize_optimizer(self, nn_model, lr, checkpoint_name, file_dir, device):
        """Instantiates and initializes the optimizer based on checkpoint availability"""
        optim = torch.optim.Adam(nn_model.parameters(), lr=lr)
        # if checkpoint_name:
        #     checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
        #     optim.load_state_dict(checkpoint["optimizer_state_dict"])
        return optim

    def initialize_scheduler(self, optimizer, checkpoint_name, file_dir, device):
        """Instantiates and initializes scheduler based on checkpoint availability"""
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, 
                                                      end_factor=0.001, total_iters=30)
        # if checkpoint_name:
        #     checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
        #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return scheduler
    
    def get_start_epoch(self, checkpoint_name, file_dir):
        """Returns starting epoch for training"""
        if checkpoint_name:
            start_epoch = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["epoch"] + 1
        else:
            start_epoch = 0
        return start_epoch
    
    def save_tensor_images(self, x_orig, x_noised, x_denoised, cur_epoch, file_dir, save_dir):
        """Saves given tensors as a single image"""
        if save_dir is None:
            fpath = os.path.join(file_dir, "saved-images", f"x_orig_noised_denoised_{cur_epoch}.png")
        else:
            fpath = os.path.join(save_dir, f"x_orig_noised_denoised_{cur_epoch}.png")
        # ensure directory exists
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        inference_transform = lambda x: (x + 1)/2
        save_image([make_grid(inference_transform(img.detach())) for img in [x_orig, x_noised, x_denoised]], fpath)

    def get_ddpm_noise_schedule(self, timesteps, beta1, beta2, device):
        """Returns ddpm noise schedule variables, a_t, b_t, ab_t
        b_t: \beta_t
        a_t: \alpha_t
        ab_t \bar{\alpha}_t
        """
        b_t = torch.linspace(beta1, beta2, timesteps+1, device=device)
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t
    
    def get_ddpm_noise_schedule_cosine(self, timesteps, beta1, beta2, device):
        """
        Returns DDPM noise schedule variables: a_t, b_t, ab_t
        b_t: \beta_t
        a_t: \alpha_t
        ab_t: \bar{\alpha}_t
        """
        import math
        # Use cosine schedule for beta_t
        t = torch.linspace(0, 1, timesteps+1, device=device)
        b_t = beta1 + 0.5 * (1 - torch.cos(t * math.pi)) * (beta2 - beta1)
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t
    
    def get_ddpm_params_from_checkpoint(self, file_dir, checkpoint_name, device):
        """Returns scheduler variables T, a_t, ab_t, and b_t from checkpoint"""
        checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), torch.device("cpu"))
        T = checkpoint["timesteps"]
        schedule = checkpoint.get("noise_schedule", "cosine")
        if schedule == "linear":
            a_t, b_t, ab_t = self.get_ddpm_noise_schedule(T, checkpoint["beta1"], checkpoint["beta2"], device)
        else:
            a_t, b_t, ab_t = self.get_ddpm_noise_schedule_cosine(T, checkpoint["beta1"], checkpoint["beta2"], device)
        return T, a_t, b_t, ab_t
    
    def denoise_add_noise(self, x, t, pred_noise, a_t, b_t, ab_t, z):
        """Removes predicted noise from x and adds gaussian noise z
        i.e., Algorithm 2, step 4 at the ddpm article
        """
        noise = b_t.sqrt()[t]*z
        denoised_x = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return denoised_x + noise
    
    def initialize_dataset_name(self, file_dir, checkpoint_name, dataset_name):
        """Initializes dataset name based on checkpoint availability"""
        if checkpoint_name:
            return torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["dataset_name"]
        return dataset_name
    
    def initialize_dataloader(self, dataset, batch_size, checkpoint_name, file_dir):
        """Returns dataloader based on batch-size of checkpoint if present"""
        if checkpoint_name:
            batch_size = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["batch_size"]
        return DataLoader(dataset, batch_size, True)
    
    def get_masked_context(self, context, p=0.9):
        "Randomly mask out context"
        return context*torch.bernoulli(torch.ones((context.shape[0], 1))*p)
    
    def save_generated_samples_into_folder(self, n_samples, context, folder_path, sampler="ddpm", ddim_steps=20, **kwargs):
        """Save DDPM generated inputs into a specified directory"""
        samples, _, _ = self.sample_with_sampler(n_samples, context, sampler=sampler, ddim_steps=ddim_steps, **kwargs)
        for i, sample in enumerate(samples):
            save_image(sample, os.path.join(folder_path, f"image_{i}.png"))
    
    def save_dataset_test_images(self, n_samples):
        """Save dataset test images with specified number"""
        folder_path = os.path.join(self.file_dir, f"{self.dataset_name}-test-images")
        os.makedirs(folder_path, exist_ok=True)

        dataset = self.instantiate_dataset(self.dataset_name, 
                            (transforms.ToTensor(), None), self.file_dir, train=False)
        dataloader = DataLoader(dataset, 1, True)
        for i, (image, _) in enumerate(dataloader):
            if i == n_samples: break
            save_image(image, os.path.join(folder_path, f"image_{i}.jpeg"))

    def initialize_device(self, device):
        """Initializes device based on availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def get_custom_context(self, n_samples, n_classes, device):
        """Returns custom context in one-hot encoded form"""
        import json
        data = json.load(open('data/test.json'))
        object_dict = json.load(open('data/object.json'))

        labels = torch.zeros(len(data), 24)
        for i in range(len(data)):
            for obj in data[i]:
                labels[i][object_dict[obj]] = 1
                
        # new_label = torch.zeros(len(data), 24*23*22//6)
        # for i in range(len(data)):
        #     index = 0
        #     for obj in range(24):
        #         if labels[i][obj] == 1:
        #             index += pow(2, 23-i)
        #     new_label[i][index] = 1
        # new_label = new_label.to(self.device)
        return labels.to(self.device)

    def _labels_from_json(self, json_path):
        import json
        data = json.load(open(json_path))
        object_dict = json.load(open('data/object.json'))
        labels = torch.zeros(len(data), 24)
        for i in range(len(data)):
            for obj in data[i]:
                labels[i][object_dict[obj]] = 1
        return labels

    def evaluate_on_json(self, json_path, sampler="ddpm", ddim_steps=20):
        """Generate images for given json and compute accuracy using pretrained evaluator."""
        from evaluator import evaluation_model
        labels = self._labels_from_json(json_path).to(self.device)
        n_samples = labels.shape[0]
        x0, _, _ = self.sample_with_sampler(n_samples=n_samples, context=labels, sampler=sampler, ddim_steps=ddim_steps)
        # evaluator expects normalized images; our x0 is in [-1,1], equivalent to Normalize(0.5,0.5) on [0,1]
        images_for_eval = x0
        # move to cuda to match evaluator implementation
        eval_model = evaluation_model(device=self.device.type)
        acc = eval_model.eval(images_for_eval, labels)
        print(f"Accuracy on {os.path.basename(json_path)}: {acc:.4f}")
        return acc

    def save_images_for_conditions(self, json_path, out_dir, sampler="ddpm", ddim_steps=20):
        """Generate and save per-condition PNG images following the order in json file."""
        os.makedirs(out_dir, exist_ok=True)
        labels = self._labels_from_json(json_path).to(self.device)
        n_samples = labels.shape[0]
        x0, _, _ = self.sample_with_sampler(n_samples=n_samples, context=labels, sampler=sampler, ddim_steps=ddim_steps)
        x_vis = (x0 + 1) / 2
        for i in range(n_samples):
            save_image(x_vis[i], os.path.join(out_dir, f"{i:03d}.png"))
        return x0, labels

    def save_grid_for_json(self, json_path, out_png_path, n_images_per_row=8, sampler="ddpm", ddim_steps=20):
        """Generate images for given json and save a grid PNG."""
        labels = self._labels_from_json(json_path).to(self.device)
        n_samples = labels.shape[0]
        x0, _, _ = self.sample_with_sampler(n_samples=n_samples, context=labels, sampler=sampler, ddim_steps=ddim_steps)
        save_image((x0+1)/2, out_png_path, nrow=n_images_per_row)

    def save_denoising_process_grid_for_labels(self, label_list, out_png_path, save_rate=20, n_steps_visual=8):
        """Save a 1xN grid visualizing denoising process for a single set of labels."""
        import json
        object_dict = json.load(open('data/object.json'))
        labels = torch.zeros(1, 24)
        for obj in label_list:
            labels[0][object_dict[obj]] = 1
        labels = labels.to(self.device)
        # run sampling to collect intermediate samples
        _, inter_samples, _ = self.sample_ddpm(n_samples=1, context=labels, save_rate=save_rate)
        # inter_samples is list of tensors [B,C,H,W]; select single sample and pick evenly spaced frames
        frames = inter_samples
        if len(frames) >= n_steps_visual:
            idxs = torch.linspace(0, len(frames)-1, n_steps_visual).round().long().tolist()
            frames = [frames[i] for i in idxs]
        # concatenate horizontally
        row = torch.cat([((f+1)/2)[0] for f in frames], dim=2)
        save_image(row, out_png_path)

    def get_custom_context_from_file(self, json_path):
        """Load a json list of label strings and map to one-hot labels tensor on device"""
        import json
        data = json.load(open(json_path))
        object_dict = json.load(open('data/object.json'))
        labels = torch.zeros(len(data), 24)
        for i in range(len(data)):
            for obj in data[i]:
                labels[i][object_dict[obj]] = 1
        return labels.to(self.device)
    
    def generate(self, n_samples, n_images_per_row, timesteps, beta1, beta2, epoch=1000):
        """Generates x0 and intermediate samples xi via DDPM, 
        and saves as jpeg and gif files for given inputs
        """
        root = os.path.join(self.file_dir, "generated-images")
        os.makedirs(root, exist_ok=True)
        x0, intermediate_samples, t_steps = self.sample_ddpm(n_samples,
                                                             self.get_custom_context(
                                                                 n_samples, self.nn_model.n_cfeat, 
                                                                 self.device),
                                                             timesteps,
                                                             beta1,
                                                             beta2,)
        save_image((x0+1)/2, os.path.join(root, f"{self.dataset_name}_ddpm_images{epoch}.png"), nrow=n_images_per_row)
        # generate_animation(intermediate_samples,
        #                    t_steps, 
        #                    os.path.join(root, f"{self.dataset_name}_ani.gif"),
        #                    n_images_per_row)
        return x0, intermediate_samples


