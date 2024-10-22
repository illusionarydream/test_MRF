import numpy as np
import cv2
import matplotlib.pyplot as plt


class MRF_noisy:
    def __init__(self, img, labels, max_iter=20):
        self.origin_val = img  # noisy image
        self.labels = labels  # number of labels
        self.max_iter = max_iter  # maximum number of iterations
        self.height, self.width = img.shape  # image size
        # state value, which is the output of the MRF
        self.state_val = np.zeros((self.height, self.width))
        # * initialize the edges and the messages on the edges
        self.message = self.initialize_message(
            self.height, self.width, self.labels)
        self.temp_message = self.initialize_message(
            self.height, self.width, self.labels)

    def get_neighbours(self, x, y):
        # get the 4 neighbours of the pixel (x, y)
        neighbours = []
        if x > 0:
            neighbours.append((x - 1, y))
        if x < self.height - 1:
            neighbours.append((x + 1, y))
        if y > 0:
            neighbours.append((x, y - 1))
        if y < self.width - 1:
            neighbours.append((x, y + 1))
        return neighbours

    def pix_idx(self, x, y):
        return x * self.width + y

    # *--- Main functions ---*
    def inference(self):
        # inference the state value of the image
        # iterate over the maximum number of iterations
        for i in range(self.max_iter):
            print("Iteration: ", i)
            # update the message
            self.update_message()
            # get the state value
            self.get_state_val()

        return self.state_val

    def get_state_val(self):
        for x in range(self.height):
            for y in range(self.width):
                self.state_val[x, y] = self.get_pixel_state(x, y)
        return self.state_val/labels

    def update_message(self):
        # update the temp messages
        for x in range(self.height):
            for y in range(self.width):
                self.update_pixel_message(x, y)
        # update the message
        for key in self.message.keys():
            self.message[key] = self.temp_message[key]

    def get_pixel_state(self, x, y):
        # get the state of the pixel (x, y)
        # * state(x, y) = argmax_l phi(x, y, l) * PI_{t in N(x, y)} message(t, x)
        max_val = 0.0
        max_label = 0.0
        i_idx = self.pix_idx(x, y)
        i_x, i_y = x, y
        # iterate over the labels
        for label in range(self.labels):
            # calculate the PI
            PI_ans = 1
            for neighbour in self.get_neighbours(i_x, i_y):
                j_idx = self.pix_idx(neighbour[0], neighbour[1])
                PI_ans *= self.message[(j_idx, i_idx)][label]
            # calculate the phi
            phi = self.get_inner_potential(i_x, i_y, label)
            # update the maximum value
            if phi * PI_ans > max_val:
                max_val = phi * PI_ans
                max_label = label

        return max_label

    def update_pixel_message(self, x, y):
        j_x, j_y = x, y
        j_idx = self.pix_idx(j_x, j_y)
        # get neighbours of the pixel (x, y)
        neighbours = self.get_neighbours(x, y)
        # first iterate over the neighbours
        for neighbour in neighbours:
            # get the origin idx and the neighbour idx
            i_x, i_y = neighbour
            i_idx = self.pix_idx(i_x, i_y)
            # second iterate over the labels
            Z = 0
            for neighbour_label in range(self.labels):
                # get the label of the neighbour
                i_label = neighbour_label
                # calculate the label k
                sum = 0
                for k in range(self.labels):
                    j_label = k
                    # calculate the phi
                    phi = self.get_inner_potential(j_x, j_y, j_label)
                    # calculate the psi
                    psi = self.get_outer_potential(j_label, i_label)
                    # calculate the message: j -> i
                    PI_ans = 1
                    for neighbour_t in neighbours:
                        # t is the neighbour of j but not i
                        t_idx = self.pix_idx(neighbour_t[0], neighbour_t[1])
                        if t_idx != i_idx:
                            PI_ans *= self.message[(t_idx, j_idx)][j_label]
                    # update the sum
                    sum += phi * psi * PI_ans
                    Z += sum
                # update the temp_message
                self.temp_message[(j_idx, i_idx)][i_label] = sum
            # normalize the message
            for neighbour_label in range(self.labels):
                self.temp_message[(j_idx, i_idx)][neighbour_label] /= Z

    def initialize_message(self, height, width, labels):
        # initialize the edges and the messages on the edges
        message = {}
        for x in range(height):
            for y in range(width):
                neighbours = self.get_neighbours(x, y)
                for neighbour in neighbours:
                    tuple_ij = (self.pix_idx(x, y), self.pix_idx(
                        neighbour[0], neighbour[1]))
                    message[tuple_ij] = np.ones(labels)
        return message

    def get_inner_potential(self, x, y, label):
        # get the inner potential of the pixel (x, y) with label
        # * phi(x, y, l) = exp(-(I(x, y) - l)^2)
        return np.exp(-np.abs(self.origin_val[x, y] - label)**2)

    def get_outer_potential(self, label, label_):
        # get the outer potential of the pixel (x, y) with label
        # * psi(x, y, l, x', y', l') = exp(-(l - l')^2)
        return np.exp(-np.abs(label - label_)**2)


if __name__ == '__main__':
    img = cv2.imread("image/binary_image.png", cv2.IMREAD_GRAYSCALE)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 128:
                img[i, j] = 0
            else:
                img[i, j] = 15
    labels = 16
    mrf = MRF_noisy(img, labels)
    state_val = mrf.inference()
    plt.imshow(state_val, cmap='gray')
    plt.show()
