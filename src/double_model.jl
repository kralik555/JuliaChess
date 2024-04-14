using Flux
using JLD2
using Chess
using Chess.UCI
using SparseArrays
using Flux.Data: DataLoader
using LinearAlgebra
include("board_class.jl")

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

    push!(layers, Dense(4096, 4096))
    push!(layers, softmax)
    model = Chain(layers...)
    return PolicyNet(model)
end

function train_batch_two_models(value_net::ValueNet, policy_net::PolicyNet, tensors, move_distros, game_values, opt_value, opt_policy)
	function value_loss(x, y_true)
        y_pred_values = value_net.model(x)
        value_loss = Flux.mse(y_pred_values, y_true)
        print("Value loss: ", value_loss)
        return value_loss
    end

    function policy_loss(x, y_true)
        y_pred_moves = policy_net.model(x)
        move_loss = Flux.kldivergence(y_pred_moves, y_true)
        print("\tPolicy loss: ", move_loss, "\n")
        return move_loss
    end

    tensors = Float32.(tensors)
	move_distros = Float32.(move_distros)
	game_values = Float32.(game_values)
	game_values = reshape(game_values, 1, :)
	move_distros = permutedims(move_distros, (2, 1))
    
    
    value_data_loader = DataLoader((tensors, game_values), batchsize=128, shuffle=true)
    policy_data_loader = DataLoader((tensors, move_distros), batchsize=128, shuffle=true)

    for (x_batch, y_values_batch) in value_data_loader
        x_batch = Float32.(x_batch)
        Flux.train!(value_loss, Flux.params(value_net.model), [(x_batch, y_values_batch)], opt_value)
    end
    for (x_batch, y_moves_batch) in policy_data_loader
        x_batch = Float32.(x_batch)
        Flux.train!(policy_loss, Flux.params(policy_net.model), [(x_batch, y_moves_batch)], opt_policy)
    end
    return value_net, policy_net
end


function train_with_stockfish(value_model::ValueNet, policy_model::PolicyNet, stockfish_path::String)
	engine = runengine(stockfish_path)
	positions = Vector{String}()
	values = Vector{Float64}()
	policies = Vector{SparseVector{Float64}}()
	opt_value = Adam(0.01)
    opt_policy = Adam(0.01)

	for i in 1:20_000
		board = startboard()
		setboard(engine, board)
        println(i)
        if i == 1000
            opt_value.eta = 0.001
            opt_policy.eta = 0.001
        elseif i == 4000
            opt_value.eta = 0.0001
            opt_policy.eta = 0.0001
        end
        played_moves = 0
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
					if move_value.score.value == -1
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
                move_values *= -1
			end
			position_value = change_value(position_value)
			changed_move_values = change_policy(move_values, board)
			# policy and value changed to be between 0 and 1 and policy to make a probability distribution
			policy = SparseVector(4096, move_indexes, changed_move_values)
			# choose move baesd on stockfish?
            domove!(board, rand(moves(board)))
			push!(values, position_value)
			push!(policies, policy)
			if length(positions) >= 128
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
				value_model, policy_model = train_batch_two_models(value_model, policy_model, position_tensors, v_policies, values, opt_value, opt_policy)
				JLD2.@save "../models/double_models/stockfish_value_$(div(i, 1000)).jld2" value_model

				JLD2.@save "../models/double_models/stockfish_policy_$(div(i, 1000)).jld2" policy_model
				positions = Vector{String}()
				values = Vector{Float64}()
				policies = Vector{SparseVector{Float64}}()
			end
            played_moves += 1
            if played_moves >= 100
                break
            end
		end
	end
	quit(engine)
end


if abspath(PROGRAM_FILE) == @__FILE__
    value_net = ValueNet()
    policy_net = PolicyNet()
    train_with_stockfish(value_net, policy_net, "../stockfish/stockfish.exe")
end
