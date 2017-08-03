This is a seq2seq encoder-decoder LSTM model made according to Google's paper
href="https://arxiv.org/abs/1506.05869" A Neural Conversational Model.

The model tries to predict the next dialog line using the provided one. It
learns on the

"https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html" Cornell Movie Dialogs corpus.

Unlike simple char RNNs this model is more
sophisticated and theoretically, given enough time and data, can deduce facts
from raw text. Your mileage may vary. This particular network architecture is
based on AdditionRNN but changed to be used with a huge amount of possible
tokens (10-40k) instead of just digits.

Use the get_data.sh script to download, extract and optimize the train data.
It's been only tested on Linux, it could work on OS X or even on Windows 10
in the Ubuntu shell.

Special tokens used:

<unk> - replaces any word or other token that's not in
the dictionary (too rare to be included or completely unknown)
<eos> - end of sentence, used only in the output to
stop the processing; the model input and output length is limited by the
ROW_SIZE constant.
<go> - used only in the decoder input as the first
token before the model produced anything

The architecture is like this:

Input => Embedding Layer => Encoder => Decoder => Output (softmax)

The encoder layer produces a so called "thought vector" that contains a
neurally-compressed representation of the input. Depending on that vector the
model produces different sentences even if they start with the same token.
There's one more input, connected directly to the decoder layer, it's used to
provide the previous token of the output. For the very first output token we
send a special <go> token there, on the next iteration we
use the token that the model produced the last time. On the training stage
everything is simple, we apriori know the desired output so the decoder input
would be the same token set prepended with the <go> token
and without the last <eos> token. Example:
```
Input: "how" "do" "you" "do" "?"
Output: "I'm" "fine" "," "thanks" "!" "<eos>"
Decoder: "<go>" "I'm" "fine" "," "thanks" "!"
```
Actually, the input is reversed as per

"https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf"
Sequence to Sequence Learning with Neural Networks,

the most important words are
usually in the beginning of the phrase and they would get more weight if
supplied last (the model "forgets" tokens that were supplied "long ago", i.e.
they have lesser weight than the recent ones). The output and decoder input
sequence lengths are always equal. The input and output could be of any
length (less than {@link #ROW_SIZE}) so for purpose of batching we mask the
unused part of the row. The encoder and decoder layers work sequentially.
First the encoder creates the thought vector, that is the last activations of
the layer. Those activations are then duplicated for as many time steps as
there are elements in the output so that every output element can have its
own copy of the thought vector. Then the decoder starts working. It receives
two inputs, the thought vector made by the encoder and the token that it
_should have produced_ (but usually it outputs something else so we have our
loss metric and can compute gradients for the backward pass) on the previous
step (or <go> for the very first step). These two vectors are simply
concatenated by the merge vertex. The decoder's output goes to the softmax
layer and that's it.

The test phase is much more tricky. We don't know the decoder input because
we don't know the output yet (unlike in the train phase), it could be
anything. So we can't use methods like outputSingle() and have to do some
manual work. Actually, we can but it would require full restarts of the
entire process, it's super slow and ineffective.

First, we do a single feed forward pass for the input with a single decoder
element, <go>. We don't need the actual activations except
the "thought vector". It resides in the second merge vertex input (named
"dup"). So we get it and store for the entire response generation time. Then
we put the decoder input (<go> for the first iteration) and the thought
vector to the merge vertex inputs and feed it forward. The result goes to the
decoder layer, now with rnnTimeStep() method so that the internal layer state
is updated for the next iteration. The result is fed to the output softmax
layer and then we sample it randomly (not with argMax(), it tends to give a
lot of same tokens in a row). The resulting token is looked up in the
dictionary, printed to the {@link System#out} and then it goes to the next
iteration as the decoder input and so on until we get <eos>.

To continue the training process from a specific batch number, enter it when
prompted; batch numbers are printed after each processed macrobatch. If
you've changed the minibatch size after the last launch, recalculate the
number accordingly, i.e. if you doubled the minibatch size, specify half of
the value and so on.