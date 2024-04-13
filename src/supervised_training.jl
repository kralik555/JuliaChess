using Flux
using Chess
using Chess.UCI
using JLD2
using Flux.Data: DataLoader
using LinearAlgebra
include("data_reader.jl")
include("model.jl")
include("board_class.jl")

# ==============================================

function train_batch(model::ChessNet, tensors, move_distros, game_values, opt)
	function loss(x, y_moves, y_value)
		y_pred_moves, y_pred_value = model.model(x)
 		move_loss = Flux.kldivergence(y_pred_moves, y_moves)
        value_loss = Flux.mse(y_pred_value, y_value)
        return move_loss + value_loss
	end

	
	tensors = Float32.(tensors)
	move_distros = Float32.(move_distros)
	game_values = Float32.(game_values)
	game_values = reshape(game_values, 1, :)
	move_distros = permutedims(move_distros, (2, 1))

	data_loader = DataLoader((tensors, move_distros, game_values), batchsize=128, shuffle=true)
	
	for (x_batch, y_move_batch, y_value_batch) in data_loader
		x_batch = Float32.(x_batch)
		Flux.train!(loss, Flux.params(model.model), [(x_batch, y_move_batch, y_value_batch)], opt)
	end
	return model
end


function train_on_dict(model::ChessNet, file_path::String, num_epochs::Int, opt)
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
			model = train_batch(model, tensors, move_distros, game_values, opt)
		end

		model_save_path = "../models/supervised_model_$(epoch).jld2"
		JLD2.@save model_save_path model                                            
	end
end


function change_value(value::Float64)
	value /= 1500
	if value > 1
		value = 1
	elseif value < -1
		value = -1
	end
	return value
end

function change_policy(policy::Vector{Float64})
	policy /= 1500
	for p in policy
		if p > 1
			p = 1
		elseif p < -1
			p = -1
		end
	end
	exp_policy = exp.(policy .- maximum(policy))
	return exp_policy / sum(exp_policy)
end

function train_with_stockfish(model::ChessNet, stockfish_path::String)
	engine = runengine(stockfish_path)
	positions = Vector{String}()
	values = Vector{Float64}()
	policies = Vector{SparseVector{Float64}}()
	opt = Adam(0.001)

	for i in 1:10_000
		board = startboard()
		setboard(engine, board)
        println(i)
		while !isterminal(board)
			push!(positions, fen(board))
			engine_move_values = mpvsearch(board, engine, depth=8)
			move_values = Vector{Float64}()
			move_indexes = Vector{Int}()
			for move_value in engine_move_values
				move = move_value.pv[1]
				value = move_value.score.value
				if move_value.score.ismate == true
					value = 1500
					if sidetomove(board) == BLACK
						value = -1500
					end
				end
				push!(move_indexes, encode_move(tostring(move)))
				push!(move_values, value)
			end
			position_value = 0.0
			if sidetomove(board) == WHITE
				position_value = maximum(move_values)
			else
				position_value = minimum(move_values)
			end
			position_value = change_value(position_value)
			changed_move_values = change_policy(move_values)
			# policy and value changed to be between 0 and 1 and policy to make a probability distribution
			policy = SparseVector(4096, move_indexes, changed_move_values)
			domove!(board, rand(moves(board)))
			push!(values, position_value)
			push!(policies, policy)
			if length(positions) == 128
				position_tensors = []
				for position in positions
					push!(position_tensors, create_input_tensors(fromfen(position)))
				end
				v_policies = Vector{Any}()
				for policy in policies
					v_policy = Vector(policy)
					push!(v_policies, reshape(v_policy, 1, :))
				end
				v_policies = vcat(v_policies...)
				position_tensors = permutedims(cat(position_tensors..., dims=4), (2, 3, 1, 4))
				model = train_batch(model, position_tensors, v_policies, values, opt)
				println(i)
				positions = Vector{String}()
				values = Vector{Float64}()
				policies = Vector{SparseVector{Float64}}()
			end
		end
		if i % 1000 == 0
			model_save_path = "../models/sp_stockfish_$(div(i, 1000)).jld2"
			JLD2.@save model_save_path model
		end
	end
	quit(engine)
end

if abspath(PROGRAM_FILE) == @__FILE__
	net = ChessNet()
	train_with_stockfish(net, "../stockfish/stockfish.exe")
end
