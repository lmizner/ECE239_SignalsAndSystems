## Building and training a bigram language model
from functools import partial
import math

import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BigramLanguageModel(nn.Module):
    """
    Class definition for a simple bigram language model.

    """

    def __init__(self, config):
        """
        Initialize the bigram language model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.embeddings)
        2. A linear layer that maps embeddings to logits. (self.linear) **set bias to True**
        3. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        super().__init__()
        # ========= TODO : START ========= #

        # Embedding layer
        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim).to(device)
    
        # Linear layer with modified input size
        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias = True).to(device)
    
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout).to(device)

        # ========= TODO : END ========= #

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the bigram language model.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, 2) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, vocab_size) containing the logits.
        """

        # ========= TODO : START ========= #

        # Get embeddings for each token in the input
        x_embedded = self.embeddings(x).to(device)
    
        # Reshape the tensor to have a single dimension for predictions
        x_reshaped = x_embedded.view(-1, x_embedded.size(-1)).to(device)
 
        # Apply dropout
        x_drop = self.dropout(x_reshaped).to(device)
    
        # Pass through linear layer
        logits = self.linear(x_drop).to(device)
    
        return logits

        # ========= TODO : END ========= #

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.
        We will perform multinomial sampling which is very similar to greedy sampling
        but instead of taking the token with the highest probability, we sample the next token from a 
        multinomial distribution.

        Args:
        context : List[int]
            A list of integers (tokens) representing the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Output:
        List[int]
            A list of integers (tokens) representing the generated tokens.
        """

        ### ========= TODO : START ========= ###

        # Convert context to tensor
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)

        # Initialize list to store generated tokens
        generated_tokens = context_tensor.clone().detach().tolist()[0]

        # Loop until maximum number of new tokens is reached
        for _ in range(max_new_tokens):
            with torch.no_grad():    
                # Forward pass to get logits
                logits = self.forward(context_tensor)
        
            # Apply softmax to logits
            probabilities = F.softmax(logits[:, -1], dim=-1)

            # Sample the next token from the multinomial distribution
            next_token = torch.multinomial(probabilities, 1).item()
        
            # Append the sampled token to the generated tokens
            generated_tokens.append(next_token)
        
            # Convert the sampled tokens to a tensor
            sampled_tokens_tensor = torch.tensor([next_token], dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)
        
            # Update the context tensor by concatenating with the sampled tokens
            context_tensor = torch.cat((context_tensor[:, 1:], sampled_tokens_tensor), dim=1)

        # Convert the list of generated tokens to a tensor
        generated_tokens_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)

        return generated_tokens_tensor


        ### ========= TODO : END ========= ###


class SingleHeadAttention(nn.Module):
    """
    Class definition for Single Head Causal Self Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)

    """

    def __init__(
        self,
        input_dim,
        output_key_query_dim=None,
        output_value_dim=None,
        dropout=0.1,
        max_len=512,
    ):
        """
        Initialize the Single Head Attention Layer.

        The model should have the following layers:
        1. A linear layer for key. (self.key) **set bias to False**
        2. A linear layer for query. (self.query) **set bias to False**
        3. A linear layer for value. (self.value) # **set bias to False**
        4. A dropout layer. (self.dropout)
        5. A causal mask. (self.causal_mask) This should be registered as a buffer.

         NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        if output_key_query_dim:
            self.output_key_query_dim = output_key_query_dim
        else:
            self.output_key_query_dim = input_dim

        if output_value_dim:
            self.output_value_dim = output_value_dim
        else:
            self.output_value_dim = input_dim

        causal_mask = None  # You have to implement this, currently just a placeholder

        # ========= TODO : START ========= #

        # self.key = ...
        # self.query = ...
        # self.value = ...
        # self.dropout = ...

        # causal_mask = ...

        # Initialize linear layers
        self.key = nn.Linear(input_dim, output_key_query_dim, bias = False)
        self.query = nn.Linear(input_dim, output_key_query_dim, bias = False)
        self.value = nn.Linear(input_dim, output_value_dim, bias = False)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize causal mask
        causal_mask = torch.triu(torch.ones(max_len, max_len) == 1).transpose(0, 1)

        
        
        # ========= TODO : END ========= #

        self.register_buffer("causal_mask", causal_mask)  # Registering as buffer to avoid backpropagation

    def forward(self, x):
        """
        Forward pass of the Single Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, output_value_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        # Step 1: Apply linear transformations to the input tensor
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Step 2: Apply dropout
        Q = self.dropout(Q)
        K = self.dropout(K)
        V = self.dropout(V)

        dk = Q.size(-1) 

        # Step 3: Calculate dot product of Query and Key tensors
        dot_product = torch.matmul(Q, K.transpose(-2, -1))

        # Step 4: Scale dot product by sqrt(d_k)
        scaled_dot_product = dot_product / (dk ** 0.5)

        # Step 5: Apply causal mask
        scaled_dot_product = scaled_dot_product.masked_fill(self.causal_mask[:scaled_dot_product.size(-2), :scaled_dot_product.size(-1)], -1e9)

        # Step 6: Apply softmax to obtain attention weights
        attention_weights = F.softmax(scaled_dot_product, dim=-1)

        # Step 7: Multiply attention weights by Value tensor to get output
        output = torch.matmul(attention_weights, V)

        return output

        # ========= TODO : END ========= #


class MultiHeadAttention(nn.Module):
    """
    Class definition for Multi Head Attention Layer.

    As in Attention is All You Need (https://arxiv.org/pdf/1706.03762)
    """

    def __init__(self, input_dim, num_heads, dropout=0.1) -> None:
        """
        Initialize the Multi Head Attention Layer.

        The model should have the following layers:
        1. Multiple SingleHeadAttention layers. (self.head_{i}) Use setattr to dynamically set the layers.
        2. A linear layer for output. (self.out) **set bias to True**
        3. A dropout layer. (self.dropout) Apply dropout to the output of the out layer.

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        # ========= TODO : START ========= #

        # self.head_{i} = ... # Use setattr to implement this dynamically, this is used as a placeholder
        # self.out = ...
        # self.dropout = ...

        head_dim = input_dim // num_heads

        # Initialize attention heads
        for i in range(num_heads):
            setattr(self, f'head_{i}', SingleHeadAttention(input_dim, output_key_query_dim=head_dim, output_value_dim=head_dim, dropout=dropout))

        # Initialize output layer
        self.out = nn.Linear(input_dim, input_dim, bias=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Multi Head Attention Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        # Initialize list to store outputs of attention heads
        head_outputs = []

        # Iterate over each attention head
        for i in range(self.num_heads):
            # Get attention head and apply it to input
            head_attention = getattr(self, f'head_{i}')
            head_output = head_attention(x)
            head_outputs.append(head_output)

        # Concatenate outputs of attention heads
        concatenated_heads = torch.cat(head_outputs, dim=-1)

        # Apply linear transformation
        output = self.out(concatenated_heads)

        # Apply dropout
        output = self.dropout(output)

        return output


        # ========= TODO : END ========= #


class FeedForwardLayer(nn.Module):
    """
    Class definition for Feed Forward Layer.
    """

    def __init__(self, input_dim, feedforward_dim=None, dropout=0.1):
        """
        Initialize the Feed Forward Layer.

        The model should have the following layers:
        1. A linear layer for the feedforward network. (self.fc1) **set bias to True**
        2. A GELU activation function. (self.activation)
        3. A linear layer for the feedforward network. (self.fc2) ** set bias to True**
        4. A dropout layer. (self.dropout)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = input_dim * 4

        # ========= TODO : START ========= #

        # self.fc1 = ...
        # self.activation = ...
        # self.fc2 = ...
        # self.fc2 = ...
        # self.dropout = ...

        self.fc1 = nn.Linear(input_dim, feedforward_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(feedforward_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Feed Forward Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        ### ========= TODO : START ========= ###

        # Apply first linear layer
        x = self.fc1(x)

        # Apply GELU activation function
        x = self.activation(x)

        # Apply second linear layer
        x = self.fc2(x)

        # Apply dropout
        x = self.dropout(x)

        return x

        ### ========= TODO : END ========= ###


class LayerNorm(nn.Module):
    """
    LayerNorm module as in the paper https://arxiv.org/abs/1607.06450

    Note : Variance computation is done with biased variance.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True) -> None:
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(tuple(self.normalized_shape)))
            self.beta = nn.Parameter(torch.zeros(tuple(self.normalized_shape)))

    def forward(self, input):
        """
        Forward pass of the LayerNorm Layer.

        Args:
        input : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        # Calculate mean and variance along the last dimension
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)

        # Normalize input using mean and variance
        output = (input - mean) / torch.sqrt(var + self.eps)

        # Apply elementwise affine transformation if enabled
        if self.elementwise_affine:
            output = output * self.gamma + self.beta

        return output

        # ========= TODO : END ========= #


class TransformerLayer(nn.Module):
    """
    Class definition for a single transformer layer.
    """

    def __init__(self, input_dim, num_heads, feedforward_dim=None):
        super().__init__()
        """
        Initialize the Transformer Layer.
        We will use prenorm layer where we normalize the input before applying the attention and feedforward layers.

        The model should have the following layers:
        1. A LayerNorm layer. (self.norm1)
        2. A MultiHeadAttention layer. (self.attention)
        3. A LayerNorm layer. (self.norm2)
        4. A FeedForwardLayer layer. (self.feedforward)

        NOTE : PLEASE KEEP OF EACH LAYER AS PROVIDED BELOW TO FACILITATE TESTING.
        """

        # ========= TODO : START ========= #

        # self.norm1 = ...
        # self.attention = ...
        # self.norm2 = ...
        # self.feedforward = ...


        # Layer normalization for the first sub-layer (before attention)
        self.norm1 = LayerNorm(input_dim)

        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(input_dim, num_heads)

        # Layer normalization for the second sub-layer (after attention)
        self.norm2 = LayerNorm(input_dim)

        # Feed forward layer
        self.feedforward = FeedForwardLayer(input_dim, feedforward_dim)


        # ========= TODO : END ========= #

    def forward(self, x):
        """
        Forward pass of the Transformer Layer.

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, num_tokens, token_dim) containing the output tokens.
        """

        # ========= TODO : START ========= #

        # Residual connection for the first sub-layer
        residual1 = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual1

        # Residual connection for the second sub-layer
        residual2 = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = x + residual2

        return x

        # ========= TODO : END ========= #


class MiniGPT(nn.Module):
    """
    Putting it all together: GPT model
    """

    def __init__(self, config) -> None:
        super().__init__()
        """
        Putting it all together: our own GPT model!

        Initialize the MiniGPT model.

        The model should have the following layers:
        1. An embedding layer that maps tokens to embeddings. (self.vocab_embedding)
        2. A positional embedding layer. (self.positional_embedding) We will use learnt positional embeddings. 
        3. A dropout layer for embeddings. (self.embed_dropout)
        4. Multiple TransformerLayer layers. (self.transformer_layers)
        5. A LayerNorm layer before the final layer. (self.prehead_norm)
        6. Final language Modelling head layer. (self.head) We will use weight tying (https://paperswithcode.com/method/weight-tying) and set the weights of the head layer to be the same as the vocab_embedding layer.

        NOTE: You do not need to modify anything here.
        """

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embed_dim
        )
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.embed_dim, config.num_heads, config.feedforward_size
                )
                for _ in range(config.num_layers)
            ]
        )

        # prehead layer norm
        self.prehead_norm = LayerNorm(config.embed_dim)

        self.head = nn.Linear(
            config.embed_dim, config.vocab_size
        )  # Language modelling head

        if config.weight_tie:
            self.head.weight = self.vocab_embedding.weight

        # precreate positional indices for the positional embedding
        pos = torch.arange(0, config.context_length, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x):
        """
        Forward pass of the MiniGPT model.

        Remember to add the positional embeddings to your input token!!

        Args:
        x : torch.Tensor
            A tensor of shape (batch_size, seq_len) containing the input tokens.

        Output:
        torch.Tensor
            A tensor of shape (batch_size, seq_len, vocab_size) containing the logits.
        """

        ### ========= TODO : START ========= ###

        # Get the batch size and sequence length
        batch_size, seq_len = x.size()

        # Embedding the input tokens
        x = self.vocab_embedding(x)

        # Adding positional embeddings to the input tokens
        pos = self.pos[:seq_len].unsqueeze(0).expand(batch_size, -1)
        pos_embedding = self.positional_embedding(pos)
        x = x + pos_embedding

        # Applying dropout to the embeddings
        x = self.embed_dropout(x)

        # Passing the embedded and positional-encoded tokens through multiple Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        # Applying LayerNorm before the final layer
        x = self.prehead_norm(x)

        # Passing the output through the final linear layer (head) to obtain logits
        logits = self.head(x)

        return logits
 

        ### ========= TODO : END ========= ###

    def _init_weights(self, module):
        """
        Weight initialization for better convergence.

        NOTE : You do not need to modify this function.
        """

        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # GPT-2 style FFN init
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, context, max_new_tokens=100):
        """
        Use the model to generate new tokens given a context.

        Please copy the generate function from the BigramLanguageModel class you had implemented earlier.
        """

        ### ========= TODO : START ========= ###

        # Convert context to tensor
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)

        # Initialize list to store generated tokens
        generated_tokens = context_tensor.clone().detach().tolist()[0]

        # Loop until maximum number of new tokens is reached
        for _ in range(max_new_tokens):
            with torch.no_grad():    
                # Forward pass to get logits
                logits = self.forward(context_tensor)
        
            # Apply softmax to logits
            probabilities = F.softmax(logits[:, -1], dim=-1)

            # Sample the next token from the multinomial distribution
            next_token = torch.multinomial(probabilities, 1).item()
        
            # Append the sampled token to the generated tokens
            generated_tokens.append(next_token)
        
            # Convert the sampled tokens to a tensor
            sampled_tokens_tensor = torch.tensor([next_token], dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)
        
            # Update the context tensor by concatenating with the sampled tokens
            context_tensor = torch.cat((context_tensor[:, 1:], sampled_tokens_tensor), dim=1)

        # Convert the list of generated tokens to a tensor
        generated_tokens_tensor = torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)

        return generated_tokens_tensor


        ### ========= TODO : END ========= ###
