import torch
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable


def preprocess(x, y, img_side_size=28):
    '''Shape of img during the training process'''
    return torch.Tensor(x.reshape(-1, 1, img_side_size, img_side_size)), y


def get_data(train_ds, valid_ds, batch_size):
    '''return two data dataloader, for train and test'''
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size * 2))


def noise(n, dtype):
    '''Generates n gaussian vectors of size 100'''
    vec = Variable(torch.randn(n, 100))
    return vec.type(dtype)


class WrappedDataLoader:
    ''' Creates classes that we can use to iterate '''

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator(optimizer, discriminator, loss, real_data, fake_data, dtype, WGAN_training):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data).type(dtype)

    one = torch.FloatTensor([1]).type(dtype)
    mone = one * -1
    # Calculate error and backpropagate
    if WGAN_training:
        # compute the gradient of the mean, i.e the mean of the gradients by linearity
        prediction_real.mean().view(1).backward(one)
    else:
        error_real = loss(prediction_real, ones_target(N).type(dtype))
        error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data).type(dtype)
    if WGAN_training:
        # compute the gradient of the mean, i.e the mean of the gradients by linearity
        prediction_fake.mean().view(1).backward(mone)
        
    # Calculate error and backpropagate
    else:
        error_fake = loss(prediction_fake, zeros_target(N).type(dtype))
        error_fake.backward()
        error_discriminator = error_real - error_fake

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    if WGAN_training:
        return prediction_real, prediction_fake
    else:
        return error_real + error_fake, prediction_real, prediction_fake

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
            
def train_generator(optimizer, discriminator, loss, generated_data, dtype, WGAN_training):
    one = torch.FloatTensor([1]).type(dtype)
    N = generated_data.size(0)    # Reset gradients
    optimizer.zero_grad()    
    prediction = discriminator(generated_data).type(dtype)  # Sample noise and generate data 
    if WGAN_training:
        prediction.mean().view(1).backward(one)
    else:
        error = loss(prediction, ones_target(N).type(dtype))
        error.backward()    # Update weights with gradients
    
    optimizer.step()    # Return error
    if WGAN_training:
        return prediction
    else:
        return error
