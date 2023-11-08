using Flux
using JLD2
using Flux.Data: DataLoader
using BSON
using ONNX
using HDF5
include("data_reader.jl")
include("model.jl")

# ==============================================
function train_model(model::ChessNet, file_path::String, batch_size::Int)
	data_array = load_data(file_path)
	println(length(data_array))

	num_chunks = ceil(Int, length(data_array) / (50 * batch_size))

	function loss(x, y_move, y_value)                                           
 		y_pred_move, y_pred_value = model.model(x)                                  
        current_batch_size = size(x, 4)
		reshaped_policy = reshape(y_pred_move, 4096, current_batch_size)        
 		reshaped_value = reshape(y_pred_value, current_batch_size)                  
 		y_move_onehot = Flux.onehotbatch(y_move, 1:4096)                            
                                                                             
 		move_loss = Flux.crossentropy(reshaped_policy, y_move_onehot)               
        value_loss = Flux.mse(reshaped_value, y_value)                          
        return move_loss + value_loss                                           
 	end

	opt = Adam(0.001)

	for chunk_index in 1:num_chunks
		start_index = (chunk_index - 1) * 50 * batch_size + 1
		end_index = min(chunk_index * batch_size * 50, length(data_array))
		println(start_index, " - ", end_index)

		current_chunk = data_array[start_index:end_index]

		tensors = cat([x[1] for x in current_chunk]..., dims=4)
		moves = [x[2] for x in current_chunk]
		game_values = [x[3] for x in current_chunk]
		tensors = permutedims(tensors, (2, 3, 1, 4))

		data_loader = DataLoader((tensors, moves, game_values), batchsize=batch_size, shuffle=true)

		for (x_batch, y_move_batch, y_value_batch) in data_loader               
 			x_batch = Float32.(x_batch)                                             
            Flux.train!(loss, Flux.params(model.model), [(x_batch, y_move_batch, y_value_batch)],     opt)
		end
		model_save_path = "models/model_high_rating.jld2"
		JLD2.@save model_save_path model
	end
    model_save_path = "models/model_high_rating.jld2"                    
	JLD2.@save model_save_path model                                            
                                                                                    
    return model
end



if abspath(PROGRAM_FILE) == @__FILE__
	net = ChessNet(8, 128)
	train_model(net, "data/arrays/games_data_1.jld2", 128)
end
