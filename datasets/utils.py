import random
import matplotlib.pyplot as plt


def visualize_batch(dataloader, num_images=2):
    images, masks = next(iter(dataloader))

    random_indices = random.sample(range(len(images)), num_images)

    # Plot the images and annotations side by side
    fig, axes = plt.subplots(num_images, 2, figsize=(16, 4 * num_images))

    for i, idx in enumerate(random_indices):

        image = images[i].squeeze(0).permute(1, 2, 0)
        mask = masks[i].squeeze(0)

        axes[i, 0].imshow(image, alpha=1)
        # axes[i, 0].imshow(mask, alpha=0.5)
        axes[i, 0].axis(False)

        axes[i, 1].imshow(image, alpha=0.5)
        axes[i, 1].imshow(mask, alpha=0.5)
        axes[i, 1].axis(False)

    plt.show()
