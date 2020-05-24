import argparse
import torch
from datasets.MovingMNIST import MovingMNIST
import matplotlib.pyplot as plt
import numpy as np

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def draw_predictions(images: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor):
    images = images.cpu()
    gt = gt.cpu()
    pred = pred.cpu()

    img_list = []
    for i in range(len(images)):
        img_list.append(torch.squeeze(images[i]).numpy())

    img_list.append(torch.squeeze(gt).numpy())
    img_list.append(torch.squeeze(pred).numpy())

    show_images(img_list, titles=[''] * 10 + ['gt', 'pred'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='path to torch model to load')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model
    net = torch.load(model_path).to(device)

    dataset = MovingMNIST('./data/MovingMNIST/mnist_test_seq.npy', train=False)

    def to_float(x: torch.Tensor): return x.to(dtype=torch.float32)
    def norm(x: torch.Tensor): return (x - dataset.mean) / dataset.std
    def t(x): return norm(to_float(x))

    dataset.transform = t

    images, gt = dataset[0]

    images = images.to(device)
    gt = gt.to(device)
    with torch.no_grad():
        pred = net(torch.unsqueeze(images, 0))
    draw_predictions(images, gt, pred)


if __name__ == '__main__':
    main()