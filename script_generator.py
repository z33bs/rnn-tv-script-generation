import time

import helper
import problem_unittests as tests
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from workspace_utils import active_session



def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    # Truncate to make only full batches
    n_full_seq = len(words) // sequence_length

    # handle edge case
    # make sure that there is one-word left to
    # attach corresponding label to feature
    if n_full_seq * sequence_length + 1 > len(words):
        n_full_seq -= 1

    # rolling window through words
    features = [words[i:i + sequence_length] for i in range((n_full_seq - 1) * sequence_length)]
    targets = [words[i + sequence_length] for i in range((n_full_seq - 1) * sequence_length)]

    dataset = TensorDataset(torch.LongTensor(features), torch.LongTensor(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader




class RNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()

        # set class variables
        # todo Problematic if save whole rnn and load on different device
        self.train_on_gpu = torch.cuda.is_available()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        # embeddings and lstm_out
        nn_input = nn_input.long()
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # reshape into (batch_size, seq_length, output_size)
        batch_size = nn_input.size(0)
        out = out.view(batch_size, -1, self.output_size)
        out = out[:, -1]  # get last word in sequence

        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        # initialize hidden state with zero weights, and move to GPU if available

        return hidden




def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """

    # move data to GPU, if available
    if torch.cuda.is_available():
        inp, target = inp.cuda(), target.cuda()

    ## perform backpropagation and optimization
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in hidden])

    # zero accumulated gradients
    rnn.zero_grad()

    # get the output from the fwd pass
    output, h = rnn.forward(inp, h)

    # calculate the loss and perform backprop
    # output is scores for each word in vocab, target is index position of correct word in vocab
    loss = criterion(output, target)
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    # nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []

    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        start = time.time()
        begin_loss = 0

        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if (batch_i > n_batches):
                break

            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
            # record loss
            batch_losses.append(loss)

            # running stats
            elapsed_time = float(time.time() - start)
            speed = batch_size * batch_i / elapsed_time if elapsed_time > 0 else 0
            print("{} {:.0f}% \tTraining Loss: {:.6f} --> {:.6f} | {:.6f} \t#/s: {:.1f}".format(
                batch_i, 100. * batch_i / n_batches, batch_losses[0], batch_losses[-1], np.average(batch_losses), speed), end='\r')
            # intermittent save checkpoint & sample
            if (batch_i+(epoch_i-1)*n_batches) % show_every_n_batches == 0:
                torch.save(rnn.state_dict(), 'trained_rnn_sd_'+format(np.average(batch_losses),'.6f').replace('.','_'))
                generated_script = generate(rnn, vocab_to_int['jerry:'], int_to_vocab, token_dict,
                                            vocab_to_int[helper.SPECIAL_WORDS['PADDING']], 100)
                print('\n', generated_script.replace('\n', ' | '), '\n')
                rnn.train()

        # epoch stats
        print('Epoch: {:>4}/{:<4} Loss (beg avg end): {} {} {}\n'.format(
            epoch_i, n_epochs, batch_losses[0], np.average(batch_losses), batch_losses[-1]))
        batch_losses = []

    return rnn


def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)

        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))

        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)

        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if (train_on_gpu):
            p = p.cpu()  # move to cpu

        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()

        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())

        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        if (train_on_gpu):
            current_seq = current_seq.cpu()  # move to cpu
        # the generated word becomes the next "current sequence" and the cycle can continue
        if train_on_gpu:
            current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = ' '.join(predicted)

    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')

    # return all the sentences
    return gen_sentences


def run_tests():
    tests.test_rnn(RNN, train_on_gpu)
    tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)

if __name__ == '__main__':
    # with active_session():

        """
            RUN
        """
        resume = True

        train_on_gpu = torch.cuda.is_available()

        int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
        # Data params
        # Sequence Length
        sequence_length = 20  # of words in a sequence
        # Batch Size
        batch_size = 128

        # data loader - do not change
        train_loader = batch_data(int_text, sequence_length, batch_size)

        # Training parameters
        # Number of Epochs
        num_epochs = 15
        # Learning Rate
        learning_rate = 0.001

        # Model parameters
        # Vocab size
        vocab_size = len(vocab_to_int)
        # Output size
        output_size = vocab_size
        # Embedding Dimension
        embedding_dim = 256
        # Hidden Dimension
        hidden_dim = 256
        # Number of RNN Layers
        n_layers = 2

        # Show stats for every n number of batches
        # Now per x seconds elapsed
        show_every_n_batches = 3000

        # create model and move to gpu if available
        rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.25)

        if resume:
            if train_on_gpu:
                rnn.load_state_dict(torch.load('trained_rnn_sd', map_location=torch.device('cuda')))
            else:
                rnn.load_state_dict(torch.load('trained_rnn_sd', map_location=torch.device('cpu')))
        else:
            print('Starting with new model...')
        if train_on_gpu:
            rnn.cuda()

        generated_script = generate(rnn, vocab_to_int['jerry:'], int_to_vocab, token_dict,
                                        vocab_to_int[helper.SPECIAL_WORDS['PADDING']], 100)
        print('\n', generated_script.replace('\n', ' | '), '\n')

        # defining loss and optimization functions for training
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # training the model
        try:
            trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)
        except KeyboardInterrupt:
            torch.save(rnn.state_dict(), 'trained_rnn_sd')
            print('\nLatest Model Saved')
