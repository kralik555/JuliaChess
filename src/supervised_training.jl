using Flux
using Chess
using JLD2
using Flux.Data: DataLoader
include("data_reader.jl")
include("model.jl")

# ==============================================

function train_batch(model::ChessNet, tensors, move_distros, game_values)
	function loss(x, y_moves, y_value)
		y_pred_moves, y_pred_value = model.model(x)
 		move_loss = Flux.crossentropy(y_pred_moves, y_moves)
        value_loss = Flux.mse(y_pred_value, y_value)
        return move_loss + value_loss
	end

	opt = Adam(0.001)
	
	tensors = Float32.(tensors)
	move_distros = Float32.(move_distros)
	game_values = Float32.(game_values)
	game_values = reshape(game_values, 1, :)
	move_distros = permutedims(move_distros, (2, 1))

	data_loader = DataLoader((tensors, move_distros, game_values), batchsize=128, shuffle=true)
	
	println("Data loader created!")

	for (x_batch, y_move_batch, y_value_batch) in data_loader
		x_batch = Float32.(x_batch)
		Flux.train!(loss, Flux.params(model.model), [(x_batch, y_move_batch, y_value_batch)], opt)
	end
	return model
end


function train_on_dict(model::ChessNet, file_path::String, num_epochs::Int)
	for epoch in 1:num_epochs
		println("Epoch ", epoch)
		files = readdir(file_path)
		num_chunks = size(files)[1]
		for i in 1:num_chunks
			println("Chunk ", i)
			tensors = []
			move_distros = []
			game_values = []
			chunk = deserialize("$(file_path)chunk_$i.bin")
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
			model = train_batch(model, tensors, move_distros, game_values)
		end

		model_save_path = "../models/supervised_model_$(epoch).jld2"
		JLD2.@save model_save_path model                                            
	end
end

if abspath(PROGRAM_FILE) == @__FILE__
	net = ChessNet(8, 128)
	train_on_dict(net, "../data/move_distros/", 3)
end
