import cv2
import numpy as np


def load_data(num_samples=500, dataset="mnist"):
    data = []
    if dataset == "mnist":
        data = np.load("fashion_mnist_images.npy")
        num_samples = data.shape[1]
        data = np.reshape(data, (28, 28, num_samples))
        return data
    else:
        path = "./" + dataset + ".mp4"
        video = cv2.VideoCapture(path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_id in np.random.choice(num_frames, size=num_samples, replace=False):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read()
            frame = np.reshape(frame, (frame.shape[0]*3, frame.shape[1]))
            data.append(frame)
        return np.dstack(data)


def main():
    mnist_data = load_data()
    mit_data = load_data(dataset="MIT")
    eagle_data = load_data(dataset="Eagle")
    friends_data = load_data(dataset="Friends")
    np.savez("data.npz", mnist=mnist_data, mit=mit_data, eagle=eagle_data, friends=friends_data)


if __name__ == '__main__':
    main()
