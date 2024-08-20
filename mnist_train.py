import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.data import datasets
from mlx.nn.layers.activations import partial
import tensorflow as tf
from functools import partial
import datetime

train_data = datasets.load_mnist(train=True)
test_data = datasets.load_mnist(train=False)


def get_streamed_data(data, batch_size=64, shuffle=False):
    if shuffle:
        stream = data.shuffle().to_stream()
    else:
        stream = data.to_stream()
    return (
        stream.batch(batch_size)
        .key_transform("image", lambda x: x.astype("float32") / 255.0)
        .prefetch(8, 8)
    )


train_stream = get_streamed_data(train_data, shuffle=True)
test_stream = get_streamed_data(test_data)


class Model(nn.Module):
    def __init__(self, input_width, input_height, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc1 = nn.Linear(64 * (input_width // 2) * (input_height // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def __call__(self, x):
        out = self.conv1(x)
        out = nn.relu(out)
        out = self.conv2(out)
        out = nn.relu(out)
        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = nn.relu(out)
        out = self.fc2(out)
        return out


model = Model(28, 28, 10)
mx.eval(model.parameters())

optimizer = optim.Adam(learning_rate=0.00001)
state = [model.state, optimizer.state]


def train_step(model, X, y):
    output = model(X)
    loss = mx.mean(nn.losses.cross_entropy(output, y))
    accurracy = mx.mean(mx.argmax(output, axis=1) == y)
    return loss, accurracy


@partial(mx.compile, inputs=state, outputs=state)
def step(X, y):
    train_step_fn = nn.value_and_grad(model, train_step)
    (loss, accuracy), grads = train_step_fn(model, X, y)
    optimizer.update(model, grads)
    return loss, accuracy


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/mlx_test/" + current_time + "/train"
test_log_dir = "logs/mlx_test/" + current_time + "/test"

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

epochs = 20
for epoch in range(epochs):
    average_loss = []
    average_accuracy = []
    for idx, batch in enumerate(train_stream):
        X, y = mx.array(batch["image"]), mx.array(batch["label"])
        loss, acccuracy = step(X, y)
        mx.eval(state)

        average_loss.append(loss)
        average_accuracy.append(acccuracy)
    train_loss = mx.mean(mx.array(average_loss))
    train_acccuracy = mx.mean(mx.array(average_accuracy))

    with train_summary_writer.as_default():
        tf.summary.scalar("loss", train_loss.item(), step=epoch)
        tf.summary.scalar("accuracy", train_acccuracy.item(), step=epoch)

    print(
        f"Epoch {epoch}, Loss: {train_loss.item()}, Accuracy: {train_acccuracy.item()}"
    )

    test_loss = []
    test_accuracy = []

    for test_batch in test_stream:
        X, y = mx.array(test_batch["image"]), mx.array(test_batch["label"])
        output = model(X)
        loss = mx.mean(nn.losses.cross_entropy(output, y))
        accuracy = mx.mean(mx.argmax(output, axis=1) == y)
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    test_loss = mx.mean(mx.array(test_loss))
    test_accuracy = mx.mean(mx.array(test_accuracy))

    with test_summary_writer.as_default():
        tf.summary.scalar("loss", test_loss.item(), step=epoch)
        tf.summary.scalar("accuracy", test_accuracy.item(), step=epoch)

    print(f"Test Loss: {test_loss.item()}, Test Accuracy: {test_accuracy.item()}")

    train_stream.reset()
    test_stream.reset()
