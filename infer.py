import os
import argparse
import torch

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run joint 2D&3D diffusion inference.')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output images.')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Directory containing test images.')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import necessary modules
    from core.dataloader_inference import joint_diffusion_inference_dataset
    from torch.utils.data.dataloader import DataLoader
    from core.diffusion3d_pipeline import get_2ddiffusion_model, get_3ddiffusion_model, joint_2d_3d_diffusion, save_generation_results
    from core.options import Options
    # Load 2D diffusion model
    dict2ddiffusion_path = os.path.join(args.checkpoints, 'model.safetensors')
    pipe = get_2ddiffusion_model(dict2ddiffusion_path, device)

    # Load 3D diffusion model
    opt = Options()
    dict3ddiffusion_path = os.path.join(args.checkpoints, 'model_1.safetensors')
    diffusion3dgs_model = get_3ddiffusion_model(dict3ddiffusion_path, device, opt)

    # Prepare dataset and dataloader
    context_image_path = [os.path.join(args.test_imgs, i) for i in os.listdir(args.test_imgs)]
    test_dataset = joint_diffusion_inference_dataset(opt, context_image_path, white_bg=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Number of samples: ", len(test_dataset))

    # Create output directory
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)

    # Processing loop
    for i, batch in enumerate(test_dataloader):
        dataset_name = batch['dataset'][0]
        subject_name = batch['subject_name'][0]

        print(f"Processing {dataset_name} - {subject_name}")

        subject_save_folder = os.path.join(save_dir, subject_name)

        if os.path.exists(subject_save_folder + '/gs.ply'):
            continue
            
        WEIGHT_DTYPE = torch.float32
        os.makedirs(subject_save_folder, exist_ok=True)

        gaussians = joint_2d_3d_diffusion(batch, device, pipe, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)

        # Save results
        save_generation_results(subject_save_folder, batch, device, gaussians, diffusion3dgs_model, weight_dtype=WEIGHT_DTYPE)


if __name__ == '__main__':
    main()