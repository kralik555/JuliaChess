using Flux
using JLD2

struct ValueNet
    model
end

struct PolicyNet
    model
end

function ValueNet()
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
    push!(layers, MeanPool((2, 2), pad=(0,0), stride=(2,2)))
    
    push!(layers, Flux.flatten)

    push!(layers, Dense(4096, 256, relu))

    push!(layers, Dense(256, 128, relu))
    push!(layers, Dense(128, 64, relu))
    push!(layers, Dense(64, 1, tanh))

    model = Chain(layers...)
    return ValueNet(model)
end

function PolicyNet()
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
    push!(layers, MeanPool((2, 2), pad=(0,0), stride=(2,2)))
    
    push!(layers, Flux.flatten)

    push!(layers, Dense(4096, 4096, softmax))
    model = Chain(layers...)
    return PolicyNet(model)
end

function train_batch_two_models(value_net::ValueNet, policy_net::PolicyNet, tensors, move_distros, game_values, opt)
	tensors = Float32.(tensors)
	move_distros = Float32.(move_distros)
	game_values = Float32.(game_values)
	game_values = reshape(game_values, 1, :)
	move_distros = permutedims(move_distros, (2, 1))
    
    value_data_loader = DataLoader((tensors, game_values), batchsize=128, shuffle=true)
    policy_data_loader = DataLoader((tensors, move_distros), batchsize=128, shuffle=true)

    for (x_batch, y_moves_batch) in value_data_loader
        x_batch = Float32.(x_batch)
        Flux.train!(mse, Flux.params(value_net.model), [(x_batch, y_moves_batch)], opt)
    end
    for (x_batch, y_moves_batch) in policy_data_loader
        x_batch = Float32.(x_batch)
        Flux.train!(kldivergence, Flux.params(policy_net.model), [(x_batch, y_moves_batch)], opt)
    end
    return value_net, policy_net
end

function train_on_chunks(value_net::ValueNet, policy_net::PolicyNet, chunks_path::String, num_epochs)
    opt = Adam(0.001)
	for epoch in 1:num_epochs
		println("Epoch ", epoch)
		files = readdir(chunks_path)
		num_chunks = size(files)[1]
		for i in 1:num_chunks
			println("Chunk ", i)
			tensors = []
			move_distros = []
			game_values = []
			chunk = deserialize("$(chunks_path)chunk_$i.bin")
			k = collect(keys(chunk))
			for key in k
				board = fromfen(key)
				tensor = create_input_tensors(board)
				move_distr = Vector(chunk[key][1])
				value = chunk[key][3] / chunk[key][2]
				push!(tensors, tensor)
				push!(move_distros, reshape(move_distr, 1, :))
				push!(game_values, value)
			end
			tensors = permutedims(cat(tensors..., dims=4), (2, 3, 1, 4))
			move_distros = vcat(move_distros...)
			println("Got to training!")
            value_net, policy_net = train_batch_two_models(value_net, policy_net, tensors, move_distros, game_values)
		end

		model_save_path = "../models/value_model_$(epoch).jld2"
		JLD2.@save model_save_path value_net

		model_save_path = "../models/policy_model_$(epoch).jld2"
		JLD2.@save model_save_path policy_net
	end
end
