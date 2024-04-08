using Flux
using JLD2

struct ChessNet
    conv_layers::Int
    conv_neurons::Int
    model
end

function split_heads(x, policy_head, value_head)
	policy = policy_head(x)
	value = value_head(x)
	return policy, value
end


struct CombinedHeads
    policy_head::Chain
    value_head::Chain
end

function (ch::CombinedHeads)(x)
    return split_heads(x, ch.policy_head, ch.value_head)
end

function policy_function(x)
    reshaped_x = reshape(x, :, size(x, 4))
    reshaped_x = (transpose(reshaped_x) |> softmax) 
    return reshaped_x
end


function combined_heads_function(x)
    return split_heads(x, policy_head, value_head)
end


function ChessNet(conv_layers::Int, conv_neurons::Int)
    layers = []
    
    # Input layer
    push!(layers, Conv((3, 3), 18=>conv_neurons, pad=(1,1), stride=(1,1)))
    push!(layers, relu)
    
    # Convolutional layers
    for _ = 1:conv_layers-1
		push!(layers, Conv((3, 3), conv_neurons=>conv_neurons, pad=(1,1), stride=(1,1)))
		push!(layers, BatchNorm(conv_neurons))
		push!(layers, relu)
        push!(layers, Dropout(0.1))
    end
    
    # Flatten layer
    push!(layers, Flux.flatten)

    # Fully connected layers
    push!(layers, Dense(conv_neurons*8*8, 256, relu))
   	push!(layers, Dense(256, 256, relu))

    # Splitting into two heads: policy and value
    policy_head = Chain(Dense(256, 128, relu), Dense(128, 4096), softmax)
    
    value_head = Chain(Dense(256, 128, relu), Dense(128, 1, tanh))
    
    combined_heads = CombinedHeads(policy_head, value_head)
    
    model = Chain(layers..., combined_heads)

    return ChessNet(conv_layers, conv_neurons, model)
end


function ChessNet()
    layers = []

    push!(layers, Conv((3, 3), 18=>32, pad=(1,1), stride=(1,1)))
    push!(layers, relu)
    
    push!(layers, Conv((3, 3), 32=>64, pad=(1,1), stride=(1,1)))
    push!(layers, BatchNorm(64))
    push!(layers, relu)
    push!(layers, Dropout(0.1))
    
    push!(layers, Conv((3, 3), 64=>128, pad=(1,1), stride=(1,1)))
    push!(layers, BatchNorm(128))
    push!(layers, relu)
    push!(layers, Dropout(0.1))

    push!(layers, Conv((3, 3), 128=>128, pad=(2,2), stride=(1,1), dilation=(2,2)))
    push!(layers, BatchNorm(128))
    push!(layers, relu)
    push!(layers, Dropout(0.1))

    push!(layers, Conv((3, 3), 128=>256, pad=(2,2), stride=(1,1), dilation=(2,2)))
    push!(layers, BatchNorm(256))
    push!(layers, relu)
    
    push!(layers, Flux.flatten)

    policy_head = Chain(Dense(256, 128, relu), Dropout(0.1), Dense(128, 256, relu), Dense(256, 4096), softmax)

    value_head = Chain(Dense(256, 128, relu), Dropout(0.1), Dense(128, 64, relu), Dense(64, 1, tanh))

    combined_heads = CombinedHeads(policy_head, value_head)

    model = Chain(layers..., combined_heads)

    return ChessNet(4, 32, model)
end

