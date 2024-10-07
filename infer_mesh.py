import os
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run joint 2D&3D diffusion inference.')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output images.')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Directory containing model checkpoints.')
    parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Directory containing test images.')
    parser.add_argument('--mesh_quality', type=str, default='high', choices=['high', 'low', 'None'], help='Quality of the generated mesh.')
    args = parser.parse_args()

    # Import necessary modules
    from core.dataloader_inference import joint_diffusion_inference_dataset
    from torch.utils.data.dataloader import DataLoader
    from core.tsdf_mesh import generate_tsdf_mesh
    from core.options import Options

    # Prepare dataset and dataloader
    opt = Options()
    context_image_path = [os.path.join(args.test_imgs, i) for i in os.listdir(args.test_imgs)]
    test_dataset = joint_diffusion_inference_dataset(opt, context_image_path, white_bg=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    save_dir = args.output

    # Processing loop
    for i, batch in enumerate(test_dataloader):
        dataset_name = batch['dataset'][0]
        subject_name = batch['subject_name'][0]

        print(f"Processing {dataset_name} - {subject_name}")

        subject_save_folder = os.path.join(save_dir, subject_name)

        if os.path.exists(subject_save_folder + '/gs.ply'):
            if args.mesh_quality == 'None' or os.path.exists(os.path.join(subject_save_folder, 'tsdf-rgbd.ply')):
                print(f"{dataset_name} - {subject_name} already processed, skipping")
                continue

        if args.mesh_quality != 'None':
            generate_tsdf_mesh(subject_save_folder, os.path.join(args.checkpoints, 'pifuhd.pt'), quality=args.mesh_quality)

if __name__ == '__main__':
    main()